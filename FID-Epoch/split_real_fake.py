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
    source_root = '/workspace/Image_translation_codes/ASGIT/checkpoint/SEN1-2_10%_spring/test'
    save_root = '/workspace/Image_translation_codes/FID-Epoch/AttnCyclegan/SEN1-2_10%_spring/images/epoch'
    
    for i in range(5, 101, 5):
        tmp_source_path = source_root + '_' + str(i) + '/images/'
        
        tmp_real_path = save_root + str(i) + '/real/'
        tmp_fake_path = save_root + str(i) + '/fake/'

        split_by_name(tmp_source_path, tmp_real_path, tmp_fake_path)
