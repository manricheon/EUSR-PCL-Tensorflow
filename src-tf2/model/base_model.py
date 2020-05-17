import math
import tensorflow as tf
from tensorflow.python.keras.layers import Add, Conv2D, Lambda, concatenate, ReLU, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import GlorotUniform

from model.common import pixel_shuffle


def conv(x, num_feats, kernel_size=3, padding='same', activation=None, kernel_initializer=None, args=None, name=None):
    if args.is_init_res:
        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.0001/args.num_res, mode='fan_in', distribution='truncated_normal', seed=None)
    elif args.is_init_he:
        kernel_initializer = tf.keras.initializers.VarianceScaling()
    return Conv2D(num_feats, kernel_size, padding=padding, activation=activation, kernel_initializer=kernel_initializer, name=name)(x)

def res_block(x_in, num_feats, kernel_size, args, name):
    rb_name = name
    x = conv(x_in, num_feats, kernel_size, padding='same', activation='relu', args=args, name=rb_name+'/conv1')
    x = conv(x, num_feats, kernel_size, padding='same', args=args,name=rb_name+'/conv2')
    if args.is_MAM:
        x = mam(x, num_feats=num_feats, ratio=16, args=args, name=rb_name+'/mam')
    x = Add(name=rb_name+'/add')([x_in, x])
    return x

def res_module(x_in, num_feats, num_res, args, name):
    rm_name = name
    x = x_in
    for i in range(num_res):
        x = res_block(x, num_feats, 3, args=args, name=rm_name+'/resb'+str(i+1))
    x = conv(x, num_feats, 3, padding='same', args=args, name=rm_name+'/conv')
    x = Add(name=rm_name+'/add')([x_in, x])
    return x

def upsampler(x, scale, num_feats, name):
    up_name = name
    _scale = int(math.log(scale,2))
    for i in range(_scale):
        x = Conv2D(num_feats*4, 3, padding='same', name=up_name+'/up'+str(i+1)+'/conv')(x)
        x = Lambda(pixel_shuffle(scale=2), name=up_name+'/up'+str(i+1)+'/pixel_shuffle')(x)
    return x

def upsampler_eusr(x, scale, num_feats, name):
    up_name = name
    _scale = int(math.log(scale,2))
    for i in range(_scale):
        x_list = list()
        for j in range(4):
            x = res_module(x, num_feats, num_res=1, name=up_name+'/up'+str(i+1)+'/rm'+str(j+1))
            x_list.append(x)
        x = concatenate(x_list, axis=3, name=up_name+'/up'+str(i+1)+'/concat')
        x = Lambda(pixel_shuffle(scale=2), name=up_name+'/up'+str(i+1)+'/pixel_shuffle')(x)
    return x

def scale_specific_upsampler(x, num_feats, scale, name):
    ssu_name = name
    x = upsampler_eusr(x, scale, num_feats, name=ssu_name+'/upsampler')
    return x

def scale_specific_moudle(x, num_feats, name):
    ssm_name = name
    x = res_block(x, num_feats, kernel_size=5, scaling=1, name=ssm_name+'/resb1')
    x = res_block(x, num_feats, kernel_size=5, scaling=1, name=ssm_name+'/resb2')
    return x

def mam(x, num_feats, ratio, args, name):
    mam_name = name

    modulation_map_CSI = 0.0
    modulation_map_ICD = 0.0
    modulation_map_CSD = 0.0

    if args.is_CSI or args.is_ICD:
        _, tmp_var = tf.nn.moments(x, axes=[1,2], keepdims=True, name=mam_name+'/m1')
        if args.is_std_norm:
            mean_var, var_var = tf.nn.moments(tmp_var, axes=-1, keepdims=True, name=mam_name+'/m2')
            tmp_var = (tmp_var - mean_var) / tf.sqrt(var_var + 1e-5)

    if args.is_CSI:
        modulation_map_CSI = tmp_var

    if args.is_ICD:
        tmp = Dense(num_feats//ratio, activation=ReLU(), kernel_initializer=tf.keras.initializers.VarianceScaling(), name=mam_name+'/ICD_dense1')(tmp_var)
        modulation_map_ICD = Dense(num_feats, name=mam_name+'/ICD_dense2')(tmp)

    if args.is_CSD:
        init_w = tf.keras.initializers.GlorotUniform()
        init_b = tf.zeros_initializer()

        W = tf.Variable(init_w(shape=(3,3,num_feats,1)))
        b = tf.Variable(init_b(shape=(num_feats)))

        modulation_map_CSD = tf.nn.depthwise_conv2d(x, filter=W, strides=[1,1,1,1], padding='SAME', name=mam_name+'/CSD_up') + b

    modulation_map = tf.sigmoid(modulation_map_CSI+modulation_map_ICD+modulation_map_CSD, name=mam_name+'/sigmoid')

    return modulation_map * x


def discriminator_block(x_in, num_filters, strides=1, batchnorm=True, momentum=0.8, name=''):
    db_name = "DB"+str(name)

    x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same', name=db_name+'/conv')(x_in)
    if batchnorm:
        x = BatchNormalization(momentum=momentum, name=db_name+'/bn')(x)
    return LeakyReLU(alpha=0.2, name=db_name+'/lrelu')(x)