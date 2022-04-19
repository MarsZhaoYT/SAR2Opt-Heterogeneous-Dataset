import os
from shutil import copy

def check_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_by_name(dir_path, real_save_path, fake_save_path):
    check_dir_exists(real_save_path)
    check_dir_exists(fake_save_path)

    if not os.path.exists(real_save_path):
        os.makedirs(real_save_path)
    
    if not os.path.exists(fake_save_path):
        os.makedirs(fake_save_path)

    for file in os.listdir(dir_path):
        tmp_file = os.path.join(os.path.abspath(dir_path), file)
        if 'real_B' in file:
            copy(tmp_file, real_save_path + os.sep + file)
        elif 'fake_B' in file:
            copy(tmp_file, fake_save_path + os.sep + file)

if __name__ == '__main__':
    source_root = 'Image_translation_codes/ASGIT/results/re-SEN1-2_20%_spring/test_latest/images'
    save_root = 'Image_translation_codes/ASGIT/results/re-SEN1-2_20%_spring'
    
    
    # tmp_source_path = source_root + '_' + str(i) + '/images/'
    
    real_path = save_root + '/real/'
    fake_path = save_root + '/fake/'

    split_by_name(source_root, real_path, fake_path)
