# SAR2Opt-Heterogeneous-Dataset

![Author](https://img.shields.io/badge/Author-MarsZYT-orange.svg)

This is an overview of our proposed heterogeneous remote sensing images "SAR2Opt" dataset.  

Sar2Opt dataset can be used as a benchmark in __change detection__ and __image transaltion__ on remote sensing images.

![Examples](imgs/Honeyview_sar2opt.png)  


## Description of dataset
We manually selected ground points on each pair of SAR-optical images to perform fine registration. All the pathces were cropped in size of 600*600 pixels after registration. If you want to use a pre-trained model, you could resize the patches first.

## Dataset
You can get the dataset from:  
- [Google Drive](https://drive.google.com/file/d/1XB9pWq-tVdxQsbVALxbYIF0Em90J4kkR/view?usp=sharing)  
- [BaiduDisk](https://pan.baidu.com/s/1xQ1nc2aPFdJ99SI2upl5Tg) (hy8d)


## Image-to-image translation results on __SAR2Opt dataset__
Here are some translated results on our SAR2Opt dataset with well-known GAN-based methods, which have been included in our GRSL paper under reviewing.

Baselines we referenced here are:
- [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [BicycleGAN](https://github.com/junyanz/BicycleGAN)
- [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [MUNIT](https://github.com/NVlabs/MUNIT)
- [NICE-GAN](https://github.com/alpc91/NICE-GAN-pytorch)
- [CUT](https://github.com/taesungp/contrastive-unpaired-translation)  

We are grateful to the authors who have shared their codes kindly.

![results](imgs/Honeyview_translated_results.png)

## Citation
__If you find this dataset valuable in your projects, please cite our paper below:__

```
Coming soon...
```
