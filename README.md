# EUSR-PCL
GAN-based image super-resolution using perceptual content losses (tensorflow implementation)

## Introduction
This repository include a implementation of GAN-based single image super-resolution using perceptual content losses (PCL), which considers both distortion- and perception-based quality of super-resolved images. In the PIRM Challenge on Perceptual Super Resolution at ECCV 2018, we won the 2nd place for Region 1. (Our team, Yonsei-MCML, also won the 2nd place for Region 2 based on [4PP-EUSR](https://github.com/idearibosome/tf-perceptual-eusr) model.)

## Performance of the method
The perceptual index is calculated by two no-reference quality measurements, [Ma](https://arxiv.org/abs/1612.05890) and [NIQE](https://doi.org/10.1109/LSP.2012.2227726). Lower score means better perceptual quality. The detail of this index is explained in the [PIRM Challenge](https://www.pirm2018.org/PIRM-SR.html).

**Final ranking of our method (Yonsei-MCML)**
(please check the details in [PIRM website](https://www.pirm2018.org/PIRM-SR.html))

![Final ranking](https://github.com/manricheon/eusr-pcl-tf/blob/master/figures/ranking_table_pirm.PNG)

**Performance comparison**
(Note that this performance comparison is performed for the paper. please check the details in [our paper](https://arxiv.org/abs/1809.04783))

![Performance comparison](https://github.com/manricheon/eusr-pcl-tf/blob/master/figures/performance_table_bsd.png)

## Usage of testing code
The instructions for the usage of testing code is below. Generating super-resolved images from the pre-trained models can be done by `<test/test.py>`.


## Information

Please cite following papers when you use the code, pre-trained models, or results:
- M. Cheon, J.-H. Kim, J.-H. Choi, J.-S. Lee: Generative adversarial network-based image super-resolution using perceptual content losses. arXiv:1809.04783 (2018) (To appear in ECCV 2018 workshop)
- J.-H. Kim, J.-S. Lee: Deep residual network with enhanced upscaling module for super-resolution. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 913-921 (2018)



## Example images

