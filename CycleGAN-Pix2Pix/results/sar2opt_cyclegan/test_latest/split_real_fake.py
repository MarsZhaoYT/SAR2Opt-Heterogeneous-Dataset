import os
from shutil import copy

def check_dir_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

def split_by_name(dir_path, fakeA_save_path, fakeB_save_path):
    check_dir_exists(fakeA_save_path)
    check_dir_exists(fakeB_save_path)

    if not os.path.exists(fakeA_save_path):
        os.mkdir(fakeA_save_path)
    
    if not os.path.exists(fakeB_save_path):
        os.mkdir(fakeB_save_path)

    for file in os.listdir(dir_path):
        tmp_file = os.path.join(os.path.abspath(dir_path), file)
        if 'fake_A' in file:
            copy(tmp_file, fakeA_save_path + os.sep + file)
        elif 'fake_B' in file:
            copy(tmp_file, fakeB_save_path + os.sep + file)

if __name__ == '__main__':
    dir_path = 'Image_translation_codes/pytorch-CycleGAN-and-pix2pix/results/sar2opt_cyclegan/test_latest/images'
    fakeA_path = 'Image_translation_codes/pytorch-CycleGAN-and-pix2pix/results/sar2opt_cyclegan/test_latest/fakeA'
    fakeB_path = 'Image_translation_codes/pytorch-CycleGAN-and-pix2pix/results/sar2opt_cyclegan/test_latest/fakeB'
    split_by_name(dir_path, fakeA_path, fakeB_path)
