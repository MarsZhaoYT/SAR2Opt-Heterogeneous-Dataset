from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse


def cal_SSIM(x, y):
    return ssim(x, y, multichannel=True)


def get_dirImage(real_dir, fake_dir):
    real_image_paths = sorted(glob(os.path.join(real_dir, "*.*")))
    fake_image_paths = sorted(glob(os.path.join(fake_dir, "*.*")))
    return real_image_paths, fake_image_paths


def cal_batch_SSIM(real_paths, fake_paths):
    assert len(real_paths) == len(fake_paths)

    SSIMs = []
    for i in tqdm(range((len(real_paths)))):

        img1 = Image.open(real_paths[i])
        img2 = Image.open(fake_paths[i])

        tmp_ssim = cal_SSIM(np.array(img1), np.array(img2))
        SSIMs.append(tmp_ssim)
    return SSIMs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_root', type=str, default='/workspace/Image_translation_codes/pytorch-CycleGAN-and-pix2pix/results/SEN1_2_20%_spring_cyclegan/test_latest')
    parser.add_argument('--fake_subdir', type=str, default='fake')
    parser.add_argument('--real_subdir', type=str, default='real')

    # 只需要修改此处
    # root = r'D:\ZYT\Codes\1.Heterogeneous_CD\Image_Translation\pytorch-CycleGAN-and-pix2pix\results\sar2opt_pix2pix\test_latest'

    opt = parser.parse_args() 
    real_dir = opt.dir_root + '/' + opt.real_subdir
    fake_dir = opt.dir_root + '/' + opt.fake_subdir

    real_paths, fake_paths = get_dirImage(real_dir, fake_dir)
    print(len(real_paths))

    SSIMs = cal_batch_SSIM(real_paths, fake_paths)

    print("mean SSIM: ", np.mean(SSIMs))

