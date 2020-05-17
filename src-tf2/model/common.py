import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


# ---------------------------------------
#  Resolve output
# ---------------------------------------
def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]

def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch

# ---------------------------------------
#  Resolve output via chop-forward
# ---------------------------------------

def resolve_chop_single(model, lr, scale=1, shave=10):
    sr_batch = resolve_chop(model, tf.expand_dims(lr,axis=0), scale, shave)[0]
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch

def resolve_chop(model, x, scale=2, shave=10, chopsize=1000000):
    b, h, w, c = x.shape
    h_half, w_half = h//2, w//2
    h_size, w_size = h_half + shave, w_half + shave
    
    input1 = x[:,0:h_size,0:w_size,:]
    input2 = x[:,0:h_size,(w-w_size):w,:]
    input3 = x[:,(h-h_size):h,0:w_size,:]
    input4 = x[:,(h-h_size):h,(w-w_size):w,:]

    outputPatch = []
    
    if w*h < chopsize:
        output1 = model(tf.cast(input1, tf.float32))
        outputPatch.append(output1)
        output2 = model(tf.cast(input2, tf.float32))
        outputPatch.append(output2)     
        output3 = model(tf.cast(input3, tf.float32))
        outputPatch.append(output3)     
        output4 = model(tf.cast(input4, tf.float32))
        outputPatch.append(output4)     
    else:
        output1 = resolve_chop(model, input1, scale, shave, chopsize)
        outputPatch.append(output1)
        output2 = resolve_chop(model, input2, scale, shave, chopsize)
        outputPatch.append(output2)
        output3 = resolve_chop(model, input3, scale, shave, chopsize)
        outputPatch.append(output3)             
        output4 = resolve_chop(model, input4, scale, shave, chopsize)
        outputPatch.append(output4)

    h, w = h*scale, w*scale
    h_half, w_half = h_half*scale, w_half*scale
    h_size, w_size = h_size*scale, w_size*scale
    shave = shave*scale
    
    upper = np.concatenate((outputPatch[0][:,0:h_half,0:w_half,:], outputPatch[1][:,0:h_half,w_size-w+w_half:w_size,:]), axis=2)
    lower = np.concatenate((outputPatch[2][:,h_size-h+h_half:h_size,0:w_half,:], outputPatch[3][:,h_size-h+h_half:h_size,w_size-w+w_half:w_size,:]), axis=2)
    output = np.concatenate((upper, lower), axis=1)

    return output


# ---------------------------------------
#  Evaluation
# ---------------------------------------

def evaluate(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


def evaluate_chop(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve_chop(model, lr)
        sr = tf.clip_by_value(sr, 0, 255)
        sr = tf.round(sr)
        sr = tf.cast(sr, tf.uint8)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


# ---------------------------------------
#  Normalization
# ---------------------------------------

def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------
def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)


# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------
def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)