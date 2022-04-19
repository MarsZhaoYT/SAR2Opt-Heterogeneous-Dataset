
dataset_root='Image_translation_codes/NICE-GAN-pytorch/dataset/'
dataset_name='SEN1_2/Splited_by_percent/20%_train/ROIs2017_winter_20%_train'

dataset=$dataset_root$dataset_name

for ((i=1; i<=5; i++));
do
    echo "Processing"$i"th epoch..."

    python Image_translation_codes/NICE-GAN-pytorch/main.py \
            --dataset $dataset \
            --phase test
                   
done