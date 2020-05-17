import os
import time
import tensorflow as tf
import numpy as np

from model import evaluate, evaluate_chop, resolve_single, resolve_chop_single
from model import common

from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.optimizers.schedules import ExponentialDecay


class Trainer:
    def __init__(self,
                 model,
                 loss,
                 learning_rate,
                 checkpoint_path,
                 args):

        self.now = None
        self.loss = loss
        self.learning_rate = learning_rate
        self.args = args

        if checkpoint_path:
            self.ckpt_path = checkpoint_path
        else:
            self.ckpt_path = os.path.join(self.args.exp_dir, self.args.exp_name, 'ckpt')        

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=model)

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                         directory=self.ckpt_path,
                                         max_to_keep=100)

        self.restore(self.ckpt_path)


    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, save_best_only=False):
        loss_mean = Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(self.args.num_iter - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            loss = self.train_step(lr, hr)
            loss_mean(loss)

            loss_value = loss_mean.result()
            loss_mean.reset_states()

            lr_value = ckpt.optimizer._decayed_lr('float32').numpy()

            duration = time.perf_counter() - self.now                
            self.now = time.perf_counter()               
            
            if step % self.args.log_freq == 0:
                tf.summary.scalar('loss', loss_value, step=step)
                tf.summary.scalar('lr', lr_value, step=step)

            if step % self.args.print_freq == 0:
                 print(f'{step}/{self.args.num_iter}: loss = {loss_value.numpy():.3f} , lr = {lr_value:.6f} ({duration:.2f}s)')

            if step % self.args.valid_freq == 0:
                psnr_value = self.evaluate(valid_dataset)
                ckpt.psnr = psnr_value
                tf.summary.scalar('psnr', psnr_value, step=step)

                print(f'{step}/{self.args.num_iter}: loss = {loss_value.numpy():.3f}, lr = {lr_value:.6f}, PSNR = {psnr_value.numpy():3f}')

            if step % self.args.save_freq == 0:
                # save weights only
                save_path = self.ckpt_path + '/weights-' + str(step) + '.h5'
                self.checkpoint.model.save_weights(filepath=save_path, save_format='h5')

                # save ckpt (weights + other train status)
                ckpt_mgr.save(checkpoint_number=step)

            

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)
            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(loss_value, self.checkpoint.model.trainable_variables)
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))

        return loss_value

    def evaluate(self, dataset):
        return evaluate_chop(self.checkpoint.model, dataset)

    def test(self, img):
        return resolve_chop_single(self.checkpoint.model, img)

    def restore(self, ckpt_path):
        if os.path.isdir(ckpt_path) is False:
            self.checkpoint.restore(ckpt_path).expect_partial()
            print(f'Model restored from checkpoint path {ckpt_path}')
        else:
            if self.checkpoint_manager.latest_checkpoint:
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                print(f'Model checkpoint restored at step {self.checkpoint.step.numpy()}')
            else:
                print('No Model restored')


class EUSRTrainer(Trainer):
    def __init__(self,
                 model,
                 ckpt_path,
                 args):
        super().__init__(model=model,
                         loss=MeanAbsoluteError(),
                         learning_rate=ExponentialDecay(args.lr_init, args.lr_decay_step, args.lr_decay_ratio, staircase=True),
                         checkpoint_path=ckpt_path,
                         args=args)

    def train(self, train_dataset, valid_dataset, save_best_only=True):
        super().train(train_dataset, valid_dataset, save_best_only)



class GANTrainer:
    def __init__(self,
                 generator,
                 discriminator,
                 loss_img,
                 learning_rate,
                 checkpoint_path,
                 args):

        self.now = None
        # self.loss = loss
        self.learning_rate = learning_rate
        self.args = args

        if checkpoint_path:
            self.ckpt_path = checkpoint_path
        else:
            self.ckpt_path = os.path.join(self.args.exp_dir, self.args.exp_name, 'ckpt')        

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                              psnr=tf.Variable(-1.0),
                                              optimizer=Adam(learning_rate),
                                              model=generator)

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                         directory=self.ckpt_path,
                                         max_to_keep=100)

        self.restore(self.ckpt_path)


        # loss functions
        self.loss_img = loss_img
        self.loss_L1 = MeanAbsoluteError()
        self.loss_L2 = MeanSquaredError()
        self.loss_BCE = BinaryCrossentropy(from_logits=False)

        # generator / discriminator
        self.generator = self.checkpoint.model
        self.discriminator = discriminator
        self.optimizer_G = Adam(learning_rate=learning_rate)
        self.optimizer_D = Adam(learning_rate=learning_rate)


    @property
    def model(self):
        return self.checkpoint.model

    def train(self, train_dataset, valid_dataset, save_best_only=False):
        loss_img_mean= Mean()
        loss_dmse_mean = Mean()
        loss_dct_mean = Mean()
        loss_adv_mean= Mean()
        loss_G_mean= Mean()
        loss_D_mean= Mean()

        ckpt_mgr = self.checkpoint_manager
        ckpt = self.checkpoint

        self.now = time.perf_counter()

        for lr, hr in train_dataset.take(self.args.num_iter - ckpt.step.numpy()):
            ckpt.step.assign_add(1)
            step = ckpt.step.numpy()

            sr, loss_img, loss_dmse, loss_dct, loss_adv, loss_G, loss_D = self.train_step(lr, hr)

            loss_img_mean(loss_img)
            loss_dmse_mean(loss_dmse)
            loss_dct_mean(loss_dct)
            loss_adv_mean(loss_adv)
            loss_G_mean(loss_G)
            loss_D_mean(loss_D)

            loss_img_val = loss_img_mean.result()
            loss_dmse_val = loss_dmse_mean.result()
            loss_dct_val = loss_dct_mean.result()
            loss_adv_val = loss_adv_mean.result()
            loss_G_val = loss_G_mean.result()
            loss_D_val = loss_D_mean.result()

            loss_img_mean.reset_states()
            loss_dmse_mean.reset_states()
            loss_dct_mean.reset_states()
            loss_adv_mean.reset_states()
            loss_G_mean.reset_states()
            loss_D_mean.reset_states()

            lr_value = ckpt.optimizer._decayed_lr('float32').numpy()

            if step % self.args.log_freq == 0:
                tf.summary.scalar('loss/loss_img', loss_img_val, step=step)
                tf.summary.scalar('loss/loss_dmse', loss_dmse_val, step=step)
                tf.summary.scalar('loss/loss_dct', loss_dct_val, step=step)
                tf.summary.scalar('loss/loss_adv', loss_adv_val, step=step) 
                tf.summary.scalar('loss/loss_G', loss_G_val, step=step)  
                tf.summary.scalar('loss/loss_D', loss_D_val, step=step)                                 

                tf.summary.scalar('lr', lr_value, step=step)

                tf.summary.image('input', np.array(tf.cast(lr, tf.uint8)), step=step)
                tf.summary.image('output', np.clip(np.array(tf.cast(dct_sr, tf.uint8)),0,255), step=step)
                tf.summary.image('truth', np.clip(np.array(tf.cast(dct_hr,tf.uint8)),0,255), step=step)                


            if step % self.args.print_freq == 0:
                duration = time.perf_counter() - self.now
                print(f'{step}/{self.args.num_iter}: loss_img = {loss_img_val.numpy():.3f} , lr = {lr_value:.6f} ({duration:.2f}s)')

            if step % self.args.valid_freq == 0:
                # Compute PSNR on validation dataset
                psnr_value = self.evaluate(valid_dataset)
                tf.summary.scalar('psnr', psnr_value.numpy(), step=step)
                print(f'{step}/{self.args.num_iter}: loss_img = {loss_img_val.numpy():.3f}, lr = {lr_value:.6f}, PSNR = {psnr_value.numpy():3f}')

                if save_best_only and psnr_value <= ckpt.psnr:
                    # skip saving checkpoint, no PSNR improvement
                    continue
                ckpt.psnr = psnr_value

            if step % self.args.save_freq == 0:
                ckpt_mgr.save()

            self.now = time.perf_counter()

    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape_G, tf.GradientTape() as tape_D:
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr = self.checkpoint.model(lr, training=True)

            hr_out = self.discriminator(hr, training=True)
            sr_out = self.discriminator(sr, training=True)

            loss_img = self.loss_img(hr, sr)
            loss_dmse = self.loss_dmse(hr, sr)
            loss_dct = self.loss_dct(hr, sr)

            loss_G = loss_img + 0.1 * loss_dmse + 0.001 * loss_dct + 1 * loss_adv

            loss_adv = self.loss_BCE(tf.ones_like(sr_out), sr_out)
            loss_disc = self.loss_BCE(tf.ones_like(hr_out), hr_out)+ self.loss_BCE(tf.zeros_like(sr_out), sr_out)

            loss_D = loss_disc

        gradients_G = tape_G.gradient(loss_G, self.generator.trainable_variables)
        gradients_D = tape_D.gradient(loss_D, self.discriminator.trainable_variables)

        self.optimizer_G.apply_gradients(zip(gradients_G, self.generator.trainable_variables))
        self.optimizer_D.apply_gradients(zip(gradients_D, self.discriminator.trainable_variables))

        return sr, loss_img, loss_dmse, loss_dct, loss_adv, loss_G, loss_D

    def evaluate(self, dataset):
        return evaluate_chop(self.checkpoint.model, dataset)

    def test(self, img):
        return resolve_chop_single(self.checkpoint.model, img)

    def restore(self, ckpt_path):
        if os.path.isdir(ckpt_path) is False:
            self.checkpoint.restore(ckpt_path).expect_partial()
            print(f'Model restored from checkpoint path {ckpt_path}')
        else:
            if self.checkpoint_manager.latest_checkpoint:
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                print(f'Model checkpoint restored at step {self.checkpoint.step.numpy()}')
            else:
                print('No Model restored')


    @tf.function
    def loss_dmse(self, hr, sr):
        hr_dx, hr_dy = tf.image.image_gradients(hr)
        sr_dx, sr_dy = tf.image.image_gradients(sr)
        x_val = self.loss_L1(hr_dx, sr_dx)
        y_val = self.loss_L1(hr_dy, sr_dy)
        return x_val + y_val

    @tf.function
    def loss_dct(self, hr, sr):
        dct_hr = tf.signal.dct(hr)
        dct_sr = tf.signal.dct(sr)
        result = self.loss_L2(dct_hr, dct_sr)
        return result


class EUSRPCLTrainer(GANTrainer):
    def __init__(self,
                 generator,
                 discriminator,
                 ckpt_path,
                 args):
        super().__init__(generator=generator,
                         discriminator=discriminator,
                         loss_img=MeanAbsoluteError(),
                         learning_rate=ExponentialDecay(args.init_lr, args.decay_step, args.decay_ratio, staircase=True),
                         checkpoint_path=ckpt_path,
                         args=args)

    def train(self, train_dataset, valid_dataset, save_best_only=True):
        super().train(train_dataset, valid_dataset, save_best_only)

