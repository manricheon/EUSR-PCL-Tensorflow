import os
import matplotlib.pyplot as plt
from importlib import import_module

import tensorflow as tf
from tensorflow.python.keras.models import Model

from dataset import Dataset
from trainer import MAMNetTrainer
from trainer import EUSRTrainer, EUSRPCLTrainer
from options import args


# GPU number and memory growth setting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[args.gpu_id],'GPU')
        tf.config.experimental.set_memory_growth(gpus[args.gpu_id],True)
    except RuntimeError as e:
	    print(e)


def main():

	# model setting
	model_module = import_module("model." + args.model_name)
	model = model_module.create_generator(args)
	model.summary(line_length=120)
	print('Model created')

	# training
	if args.is_train:
		# experiment dir setting
		exp_dir = os.path.join(args.exp_dir, args.exp_name)
		ckpt_dir = os.path.join(exp_dir, 'ckpt')
		log_dir = os.path.join(exp_dir, 'log')		
		os.makedirs(exp_dir, exist_ok=True)
		os.makedirs(ckpt_dir, exist_ok=True)
		os.makedirs(log_dir, exist_ok=True)
		print('Experiment setting created')

		# dataset load
		dataset_train = Dataset(args, subset='train')
		dataset_valid = Dataset(args, subset='valid')
		train_ds = dataset_train.dataset(batch_size=args.num_batch, patch_size=args.patch_size, random_transform=True)
		valid_ds = dataset_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)
		print('Dataset loaded')

		# trainer and train
		train_summary_writer = tf.summary.create_file_writer(log_dir)
		with train_summary_writer.as_default():
			if args.is_gan:
				discriminator = model_module.create_discriminator(args)
				discriminator.summary(line_length=120)
				print('Discriminator Load Done')

				trainer = EUSRPCLTrainer(generator=model, discriminator=discriminator,
					ckpt_path=args.ckpt_path, args=args)
				print('Trainer created')
				trainer.train(train_ds,	valid_ds, save_best_only=False)
				print('Training Done')

			else:
				trainer = EUSRTrainer(model=model, ckpt_path=args.ckpt_path, args=args)
				print('Trainer created')
				trainer.train(train_ds,	valid_ds, save_best_only=False)
				print('Training Done')


	if args.is_test:
		print('Test a trained model using inference.py script :)')

if __name__ == '__main__':
	main()

