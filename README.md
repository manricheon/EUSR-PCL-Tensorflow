# EUSR-PCL
GAN-based image super-resolution using perceptual content losses (tensorflow implementation)

## Introduction
This repository include a implementation of GAN-based single image super-resolution using perceptual content losses (PCL), which considers both distortion- and perception-based quality of super-resolved images. In the PIRM Challenge on Perceptual Super Resolution at ECCV 2018, we won the 2nd place for Region 1. (Our team, [Yonsei-MCML](http://mcml.yonsei.ac.kr/), also won the 2nd place for Region 2 based on [4PP-EUSR model](https://github.com/idearibosome/tf-perceptual-eusr) by @dearibosome.)

Please cite following papers when you use the code, pre-trained models, or results:
- M. Cheon, J.-H. Kim, J.-H. Choi, J.-S. Lee: [Generative adversarial network-based image super-resolution using perceptual content losses](https://arxiv.org/abs/1809.04783). arXiv:1809.04783 (2018) (To appear in ECCV 2018 workshop)
- J.-H. Kim, J.-S. Lee: [Deep residual network with enhanced upscaling module for super-resolution](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w13/html/Kim_Deep_Residual_Network_CVPR_2018_paper.html). In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pp. 913-921 (2018) 

## Performance of the method
The perceptual index is calculated by two no-reference quality measurements, [Ma](https://arxiv.org/abs/1612.05890) and [NIQE](https://doi.org/10.1109/LSP.2012.2227726). Lower score means better perceptual quality. The detail of this index is explained in the [PIRM Challenge](https://www.pirm2018.org/PIRM-SR.html).

**Final ranking of our method (Yonsei-MCML)**
(please check the details in [PIRM website](https://www.pirm2018.org/PIRM-SR.html))

![Final ranking](https://github.com/manricheon/eusr-pcl-tf/blob/master/figures/ranking_table_pirm.PNG)


## Usage of testing code
The instructions for the usage of testing code is below. Generating super-resolved images from the pre-trained models can be done by `<test/test.py>`. Now, we only support x4 super-resolution for the challenge.

1. Download and copy the trained model available in Downloads section to the `<test/>` folder.
2. Place the low-resolution images (PNG files) to the `<test/LR/>` folder.
3. Run `<python test.py>`
4. The super-resolved images will be available on the `<test/LR/>` folder.

## Usage of training code
The training code will be upaded soon.


## Downloads
Pre-trained models (for the PIRM Challenge)
- Download PIRM Challenge version : [eusr-pcl_pirm.pb](https://drive.google.com/open?id=1TBJB7-aZEag-tAO6oVu9kpOuwdl0NTaW)
