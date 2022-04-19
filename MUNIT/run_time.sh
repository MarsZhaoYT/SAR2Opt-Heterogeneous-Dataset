# set -ex

config_file='Image_translation_codes/MUNIT/configs/SEN1_2_20%_summer.yaml'
input_folder='Image_translation_codes/MUNIT/datasets/SEN1_2/20%_train/ROIs1868_summer_20%_train/testA/'
output_folder='Image_translation_codes/MUNIT/results/SEN1_2_20%_summer/images/'
checkpoint='Image_translation_codes/MUNIT/outputs/SEN1_2_20%_summer/checkpoint/gen_00190000.pt'

for ((i=1; i<=5; i++));
do
    echo "Processing"$i"th epoch..."
    
    python Image_translation_codes/MUNIT/test.py \
                --config $config_file \
                --input_folder $input_folder \
                --output_folder $output_folder \
                --checkpoint $checkpoint \
                --a2b 1 
                   
done