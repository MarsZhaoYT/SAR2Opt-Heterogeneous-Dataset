import cv2
import os
import numpy as np


class Combine_Image():
    def __init__(self, path_AB):
        self.path_AB = path_AB

        if not os.path.exists(path_AB):
            os.makedirs(path_AB)

    def combine_image(self, path_A, path_B):
        img_A = cv2.imread(path_A)
        img_B = cv2.imread(path_B)
        img_AB = np.concatenate([img_A, img_B], 1)
        filename = path_A.split('/')[-1]
        save_path = os.path.join(self.path_AB, filename)

        cv2.imwrite(save_path, img_AB)

    def get_subfile_name(self, path):
        lst = []
        for file in os.listdir(path):
            lst.append(os.path.join(path, file))
        return lst

    def run(self, phase='train'):
        dataset_A = os.path.join(dataset_path, phase) + 'A'
        dataset_B = os.path.join(dataset_path, phase) + 'B'

        list_A = self.get_subfile_name(dataset_A)
        list_B = self.get_subfile_name(dataset_B)
        list_A.sort(key=lambda x: (int(x.split('_')[-2]), int(x.split('_p')[-1].split('.')[0])))
        list_B.sort(key=lambda x: (int(x.split('_')[-2]), int(x.split('_p')[-1].split('.')[0])))
        zipped_AB = zip(list_A, list_B)

        for path in list(zipped_AB):
            path_A = path[0]
            path_B = path[1]
            print(path_A)
            print(path_B)
            print('------------')
            self.combine_image(path_A, path_B)


if __name__ == '__main__':
    dataset_path = 'Image_translation_codes/pytorch-CycleGAN-and-pix2pix/datasets/SEN1_2/Splited_by_percent/10%_train/ROIs1158_spring_10%_train'
    combined_path = 'Image_translation_codes/pytorch-CycleGAN-and-pix2pix/datasets/combined_SEN1_2/10%_train/ROIs1158_spring_10%/test'
    Combine_Image(combined_path).run(phase='test')
