import os
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE

# dataset for div2k

class Dataset:
    def __init__(self,
                 args,
                 subset='train'):

        _scales = [2, 4, 8]

        if args.scale in _scales:
            self.scale = args.scale
        else:
            raise ValueError(f'scale must be in ${_scales}')

        if subset == 'train':
            self.image_ids = range(1, 801)
        elif subset == 'valid':
            self.image_ids = range(801, 901)
        else:
            raise ValueError("subset must be 'train' or 'valid'")

        self.name = args.dataset_name
        self.subset = subset
        self.images_dir = os.path.join(args.dataset_dir, args.dataset_name)


    def __len__(self):
        return len(self.image_ids)


    def dataset(self, batch_size=16, patch_size=48, repeat_count=None, random_transform=True):
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))
        if random_transform:
            ds = ds.shuffle(len(self.images_ids), reshuffle_each_iteration=True)
            ds = ds.map(lambda lr, hr: random_crop(lr, hr, patch_size=patch_size, scale=self.scale), num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def hr_dataset(self):
        image_dir_hr = os.path.join(self.images_dir, f'{self.name}_{self.subset}_HR')
        image_list_hr = [os.path.join(image_dir_hr, f'{image_id:04}.png') for image_id in self.image_ids]

        ds = self._images_dataset(image_list_hr)
        return ds

    def lr_dataset(self):
        image_dir_lr = os.path.join(self.images_dir, f'{self.name}_{self.subset}_LR', f'x{self.scale}')        
        image_list_lr = [os.path.join(image_dir_lr, f'{image_id:04}.png') for image_id in self.image_ids]

        ds = self._images_dataset(image_list_lr)
        return ds

    @staticmethod
    def _images_dataset(image_files):
        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        ds = ds.map(lambda x: tf.image.decode_png(x, channels=3), num_parallel_calls=AUTOTUNE)
        return ds


# -----------------------------------------------------------
#  Transformations
# -----------------------------------------------------------

def random_crop(lr_img, hr_img, patch_size=48, scale=2):
    hr_crop_size = patch_size * scale
    lr_crop_size = patch_size
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped

def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (lr_img, hr_img),
                   lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))

def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)

