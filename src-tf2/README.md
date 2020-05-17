# EUSR-PCL-TF2

## Training

```shell
python main.py
  --gpu_id 0
  --model_name EUSR-PCL
  --dataset_dir <path of the DIV2K dataset>
  --dataset_name DIV2K
  --exp_dir <path of experiments>
  --exp_name <name of experiment> 
  --num_res 16 --num_feats 64 
  --scale <scaling factor>
  --is_train --is_gan
```

## Inference (Test)

``` shell
python inference.py 
  --gpu_id 0 
  --model_name EUSR-PCL
  --test_input <path of input dir>
  --test_output <path of output dir>
  --exp_dir <path of experiments> 
  --exp_name <name of experiment>  
  --num_res 16 --num_feats 64 
  --scale <scaling factor>
```

## Acknowledgement
Many parts are learned and borrowed from useful repositories below. Thanks.
- https://github.com/krasserm/super-resolution
- https://github.com/junhyukk/MAMNet-Tensorflow
- https://github.com/manricheon/MAMNet-Tensorflow-2