# EUSR-PCL
GAN-based image super-resolution using perceptual content losses (tensorflow implementation)

## Introduction
This repository include a implementation of GAN-based single image super-resolution using perceptual content losses (PCL), which considers both distortion- and perception-based quality of super-resolved images. In the PIRM Challenge on Perceptual Super Resolution at ECCV 2018, we won the 2nd place for Region 1. (Our team, Yonsei-MCML, also won the 2nd place for Region 2 based on [4PP-EUSR](https://github.com/idearibosome/tf-perceptual-eusr))

## Performance of the method
The perceptual index is calculated by two no-reference quality measurements, [Ma](https://arxiv.org/abs/1612.05890) and [NIQE](https://doi.org/10.1109/LSP.2012.2227726). Lower score means better perceptual quality. The detail of this index is explained in the [PIRM Challenge](https://www.pirm2018.org/PIRM-SR.html).

Performance comparison

![Performance comparison](https://github.com/manricheon/eusr-pcl-tf/blob/master/figures/performance_table_bsd.png)

Final ranking of our method (Yonsei-MCML) in PRIM Challenge

![Final ranking](https://github.com/manricheon/eusr-pcl-tf/blob/master/figures/ranking_table_pirm.PNG)

## Usage of testing code



## Example images

