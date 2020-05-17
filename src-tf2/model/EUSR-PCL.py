import math
import tensorflow as tf
from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, concatenate
from tensorflow.python.keras.models import Model

from model.common import normalize, denormalize
from model.base_model import *


def create_generator(args, parent=False):
    return EUSR(args)

def create_discriminator(args, parent=False):
    return discriminator(args)


def EUSR(args):
    # init
    scale = args.scale
    num_channels = 3
    num_feats = args.num_feats
    num_res = args.num_res


    # model
    x_in = Input(shape=(None, None, num_channels), name='Input')
    x = Lambda(normalize, name='Norm')(x_in)

    # first conv
    x = Conv2D(num_feats, 3, padding='same', name='ConvIn')(x)
    x = scale_specific_moudle(x, num_feats, name='SSM')

    # residual module
    x = res_module(x, num_feats=num_feats, num_res=num_res, args=args, name='RM')

    # scale_specific_upsampler
    x = scale_specific_upsampler(x, num_feats=num_feats, scale=scale, name='SSU')

    # last conv
    x = Conv2D(num_channels, 3, padding='same', name='ConvOut')(x)    
    x = Lambda(denormalize, name='Denorm')(x)

    return Model(x_in, x, name="EUSR")



############################# Discriminator
def discriminator(args):
    num_filters=32
    HR_SIZE=192

    x_in = Input(shape=(HR_SIZE, HR_SIZE, 3), name='input')
    x = Lambda(normalize_m11, name='norm')(x_in)

    x = discriminator_block(x, num_filters, batchnorm=False, name=0)
    x = discriminator_block(x, num_filters, strides=2, name=1)

    x = discriminator_block(x, num_filters * 2, name=2)
    x = discriminator_block(x, num_filters * 2, strides=2, name=3)

    x = discriminator_block(x, num_filters * 4, name=4)
    x = discriminator_block(x, num_filters * 4, strides=2, name=5)

    x = discriminator_block(x, num_filters * 8, name=6)
    x = discriminator_block(x, num_filters * 8, strides=2, name=7)

    x = discriminator_block(x, num_filters * 16, name=8)
    x = discriminator_block(x, num_filters * 16, strides=2, name=9)

    x = Flatten()(x)

    x = Dense(1024, name='fc')(x)
    x = LeakyReLU(alpha=0.2, name='lrelu')(x)
    x = Dense(1, activation='sigmoid', name='sigmoid')(x)

    return Model(x_in, x, name='discriminator')

