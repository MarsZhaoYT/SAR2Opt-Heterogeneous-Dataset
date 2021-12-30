# Sar2Opt-Heterogeneous-Dataset

![Author](https://img.shields.io/badge/Author-MarsZYT-orange.svg)

This is an overview of our proposed heterogeneous remote sensing images "Sar2Opt" dataset.  

Sar2Opt dataset can be used as a benchmark in __change detection__ and __image transaltion__ on remote sensing images.

![Examples](https://github.com/MarsZhaoYT/Sar2Opt-Heterogeneous-Dataset/blob/main/imgs/Honeyview_sar2opt.png)  


## Description of dataset
We manually selected ground points on each pair of SAR-optical images to perform fine registration. All the pathces were cropped in size of 600*600 pixels after registration. If you want to use a pre-trained model, you could resize the patches first.

## Dataset
You can get the dataset from:  
- [Google Drive]()  
- [BaiDuDisk]()


## Image-to-image translation results on __Sar2Opt dataset__
Here are some translated results on our Sar2Opt dataset with well-known GAN-based methods, which have been included with our GRSL paper under reviewing.

Baselines used here are: 
- [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [BicycleGAN](https://github.com/junyanz/BicycleGAN)
- [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [MUNIT](https://github.com/NVlabs/MUNIT)
- [NICE-GAN](https://github.com/alpc91/NICE-GAN-pytorch)
- [CUT](https://github.com/taesungp/contrastive-unpaired-translation)  

We are grateful to the authors who have open sourced their codes kindly.

![results](https://github.com/MarsZhaoYT/Sar2Opt-Heterogeneous-Dataset/blob/main/imgs/Honeyview_translated_results.png)

## Citation
__If you find this dataset valuable in your projects, please cite our paper below:__

```

```
