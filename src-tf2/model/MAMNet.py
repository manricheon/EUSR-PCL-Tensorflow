import math
import tensorflow as tf
from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, concatenate
from tensorflow.python.keras.models import Model

from model.common import normalize, denormalize
from model.base_model import *


def create_generator(args, parent=False):
    return MAMNet(args)


def MAMNet(args):
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

    # residual module
    x = res_module(x, num_feats=num_feats, num_res=num_res, args=args, name='RM')

    # scale_specific_upsampler
    x = scale_specific_upsampler(x, num_feats=num_feats, scale=scale, name='SSU')

    # last conv
    x = Conv2D(num_channels, 3, padding='same', name='ConvOut')(x)    
    x = Lambda(denormalize, name='Denorm')(x)

    return Model(x_in, x, name="MAMNet")

