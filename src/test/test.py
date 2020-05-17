import argparse
import os

import numpy as np
import tensorflow as tf


# params
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='model.pb', help='path of the model file (.pb)')
parser.add_argument('--input_path', default='LR', help='base path of low resolution (input) images')
parser.add_argument('--output_path', default='SR', help='base path of super resolution (output) images')
parser.add_argument('--use_gpu', action='store_true', help='enable GPU utilization (default: disabled)')
args = parser.parse_args()


def main():
    if (not args.use_gpu):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # load and build graph
    with tf.Graph().as_default():

        # decode
        model_input_path = tf.placeholder(tf.string, [])
        decoded_image = tf.read_file(model_input_path)
        decoded_image = tf.image.decode_png(decoded_image, channels=3, dtype=tf.uint8)
        decoded_image = tf.cast(decoded_image, tf.float32)

        # process SR
        model_input = tf.placeholder(tf.float32, [None, None, 3])
        flag_scale = tf.constant(4, tf.float32)

        with tf.gfile.GFile(args.model_name, 'rb') as f:
            model_graph_def = tf.GraphDef()
            model_graph_def.ParseFromString(f.read())
        
        model_output = tf.import_graph_def(model_graph_def, name='model', input_map={'sr_input:0': [model_input], 'sr_flag_scale:0': flag_scale}, return_elements=['sr_output:0'])[0]

        model_output = model_output[0, :, :, :]
        model_output = tf.round(model_output)
        model_output = tf.clip_by_value(model_output, 0, 255)
        model_output = tf.cast(model_output, tf.uint8)

        # encode
        model_output_path = tf.placeholder(tf.string, [])
        encode_target_image = tf.placeholder(tf.uint8, [None, None, 3])
        encoded_image = tf.image.encode_png(encode_target_image)
        write_op = tf.write_file(model_output_path, encoded_image)

        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True
        ))
        sess.run(init)
    
    # get image path list
    image_path_list = []
    for root, subdirs, files in os.walk(args.input_path):
        for filename in files:
            if (filename.lower().endswith('.png')):
                input_path = os.path.join(args.input_path, filename)
                output_path = os.path.join(args.output_path, filename)
                
                image_path_list.append([input_path, output_path])
    print('Found %d images' % (len(image_path_list)))

    # iterate
    for input_path, output_path in image_path_list:
        print('- %s -> %s' % (input_path, output_path))

        # decode
        image = sess.run([decoded_image], feed_dict={model_input_path:input_path})[0]

        # chop forward
        scale = 4
        shave = 10
        h, w, c = image.shape
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        input1 = image[0:h_size, 0:w_size]
        input2 = image[0:h_size, (w - w_size):w]
        input3 = image[(h - h_size):h, 0:w_size]
        input4 = image[(h - h_size):h, (w - w_size):w]
        tmp1 = sess.run([model_output], feed_dict={model_input:input1})[0]
        tmp2 = sess.run([model_output], feed_dict={model_input:input2})[0]
        tmp3 = sess.run([model_output], feed_dict={model_input:input3})[0]
        tmp4 = sess.run([model_output], feed_dict={model_input:input4})[0]
        tmp_image = np.zeros([image.shape[0]*scale, image.shape[1]*scale, 3])
        h, w = h * scale, w * scale
        h_half, w_half = h_half * scale, w_half * scale
        h_size, w_size = h_size * scale, w_size * scale
        tmp_image[0:h_half, 0:w_half] = tmp1[0:h_half, 0:w_half]
        tmp_image[0:h_half, w_half:] = tmp2[0:h_half, (w_size - w + w_half):w_size]
        tmp_image[h_half:, 0:w_half] = tmp3[(h_size - h + h_half):h_size, 0:w_half]
        tmp_image[h_half:, w_half:] = tmp4[(h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        # encode
        sess.run([write_op], feed_dict={model_output_path:output_path, encode_target_image:tmp_image})

    print('Done')


if __name__ == '__main__':
    main()