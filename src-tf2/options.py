import argparse

# Parameter setting
parser = argparse.ArgumentParser()


# Device option
parser.add_argument('--gpu_id', default=-1, type=int, help='gpu id')

# Directory
parser.add_argument('--dataset_dir', default='../dataset/', type=str, help='training dataset path')
parser.add_argument('--dataset_name', default='DIV2K', type=str, help='dataset name')

parser.add_argument('--exp_dir', default='../exp/', type=str, help='experiment root dir')
parser.add_argument('--exp_name', default='exp1', type=str, help='expeirment name')

# Model specifications 
parser.add_argument('--model_name', default='EUSR-PCL', type=str, help='model name: EUSR-PCL')

parser.add_argument("--patch_size", default=48, type=int)
parser.add_argument("--num_channels", default=3, type=int)
parser.add_argument("--num_feats", default=64, type=int)
parser.add_argument("--num_res", default=16, type=int)

parser.add_argument('--scale', default=4, type=int, help='SR scale')

# Training specification
parser.add_argument('--is_train', action='store_true', help='training mode')
parser.add_argument('--is_gan', action='store_true', help='GAN training mode')
parser.add_argument('--ckpt_path', default=None, type=str, help='ckpt dir or file; for inference, it must be file')

parser.add_argument("--lr_init", default=0.0001, type=int)
parser.add_argument("--lr_decay_step", default=200000, type=int)
parser.add_argument("--lr_decay_ratio", default=0.5, type=int)

parser.add_argument("--num_iter", default=1000000, type=int)
parser.add_argument("--num_batch", default=16, type=int)
parser.add_argument("--num_train", default=800, type=int)
parser.add_argument("--num_valid", default=100, type=int)

# Log for training
parser.add_argument("--print_freq", default=10, type=int)
parser.add_argument("--log_freq", default=10, type=int)
parser.add_argument("--save_freq", default=50000, type=int)
parser.add_argument("--valid_freq", default=50, type=int)
parser.add_argument("--max_to_keep", default=1000000, type=int)

# Test (inference) setting
# parser.add_argument('--is_test', action='store_true', help='test mode')
parser.add_argument('--test_input', default='/path/test/input/', type=str, help='test input dir')
parser.add_argument('--test_output', default='/path/test/output/', type=str, help='test output dir')


args = parser.parse_args()