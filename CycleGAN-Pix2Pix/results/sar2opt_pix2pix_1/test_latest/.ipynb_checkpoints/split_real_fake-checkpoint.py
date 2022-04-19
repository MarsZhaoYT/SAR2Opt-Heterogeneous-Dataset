import os
from shutil import copyfile

def check_dir_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

def split_by_name(dir_path, real_save_path, fake_save_path):
    # check_dir_exists(real_save_path)
    # check_dir_exists(fake_save_path)

    if not os.path.exists(real_save_path):
        os.mkdir(real_save_path)
    
    if not os.path.exists(fake_save_path):
        os.mkdir(fake_save_path)

    for file in os.listdir(dir_path):
        tmp_file = os.path.join(dir_path, file)
        print(tmp_file)
        if 'real' in str(file):
            copyfile(tmp_file, real_save_path)
        else:
            copyfile(tmp_file, fake_save_path)


if __name__ == '__main__':
    dir_path = '/workspace/Image_translation_codes/pytorch-CycleGAN-and-pix2pix/results/sar2opt_pix2pix/test_latest/images'
    real_path = 'real'
    fake_path = 'fake'
    split_by_name(dir_path, real_path, fake_path)