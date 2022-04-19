import os
from shutil import copy

def check_dir_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

def split_by_name(dir_path, real_save_path, fake_save_path):
    check_dir_exists(real_save_path)
    check_dir_exists(fake_save_path)

    if not os.path.exists(real_save_path):
        os.mkdir(real_save_path)
    
    if not os.path.exists(fake_save_path):
        os.mkdir(fake_save_path)

    for file in os.listdir(dir_path):
        tmp_file = os.path.join(os.path.abspath(dir_path), file)
        if 'output' in file:
            copy(tmp_file, fake_save_path + os.sep + file)
        else:
            copy(tmp_file, real_save_path + os.sep + file)

if __name__ == '__main__':
    dir_path = 'Image_translation_codes/MUNIT/results/SEN1_2_20%_winter/images/'
    real_path = '/workspace/Image_translation_codes/MUNIT/results/SEN1_2_20%_winter/real'
    fake_path = '/workspace/Image_translation_codes/MUNIT/results/SEN1_2_20%_winter/fake'
    split_by_name(dir_path, real_path, fake_path)
