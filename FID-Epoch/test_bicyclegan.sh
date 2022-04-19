set -ex
modelDir="/workspace/Image_translation_codes/BicycleGAN/checkpoint/combined_SEN1_2_10%_spring/"
inputDataRoot="/workspace/Image_translation_codes/DATASET/combined_SEN1_2/10%_train/ROIs1158_spring/"
resultsDirRoot="/workspace/Image_translation_codes/FID-Epoch/cyclegan/SEN1_2_10%_spring/images/"
# outputFolder="/workspace/image_translation/FID-Epoch/MUNITresults/SEN1_2_10%_spring/images/"

RESULTS_DIR='results/combined_SEN1_2_10%_spring'

# dataset
CLASS='combined_SEN1_2_10%_spring_bicycle_gan'
DIRECTION='AtoB' # from domain A to domain B
LOAD_SIZE=286 # scale images to this size
CROP_SIZE=256 # then crop to this size
INPUT_NC=3  # number of channels in the input image

# misc
GPU_ID=0   # gpu id
NUM_TEST=5000 # number of input images duirng test
NUM_SAMPLES=1 # number of samples per input images

for ((i=5; i<=80; i+=5));
  do
      tmpOutputFolder="/workspace/Image_translation_codes/FID-Epoch/bicyclegan/SEN1_2_10%_spring/epoch"$i"/images/"
          
      CUDA_VISIBLE_DEVICES=${GPU_ID} python /workspace/Image_translation_codes/BicycleGAN/test.py \
            --dataroot $inputDataRoot \
            --dataset_phase train \
            --epoch $i \
            --results_dir $tmpOutputFolder \
            --checkpoints_dir /workspace/Image_translation_codes/BicycleGAN/checkpoint/combined_SEN1_2_10%_spring/ \
            --name ${CLASS} \
            --direction ${DIRECTION} \
            --load_size ${LOAD_SIZE} \
            --crop_size ${CROP_SIZE} \
            --input_nc ${INPUT_NC} \
            --num_test ${NUM_TEST} \
            --n_samples ${NUM_SAMPLES} \
            --center_crop \
            --no_flip

  done

  # /workspace/Image_translation_codes/BicycleGAN/checkpoint/combined_SEN1_2_10%_spring/combined_SEN1_2_10%_spring_bicycle_gan/5_net_D.pth