set -ex
# models
RESULTS_DIR='results/combined_SEN1_2_20%_winter'

# dataset
CLASS='combined_SEN1_2_20%_winter_bicycle_gan'
DIRECTION='AtoB' # from domain A to domain B
LOAD_SIZE=286 # scale images to this size
CROP_SIZE=256 # then crop to this size
INPUT_NC=3  # number of channels in the input image

# misc
GPU_ID=0   # gpu id
NUM_TEST=506 # number of input images duirng test
NUM_SAMPLES=1 # number of samples per input images


# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ./datasets/combined_SEN1_2/20%_train/ROIs2017_winter \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir /workspace/Image_translation_codes/BicycleGAN/checkpoint/combined_SEN1_2_20%_winter/ \
  --name ${CLASS} \
  --direction ${DIRECTION} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --input_nc ${INPUT_NC} \
  --num_test ${NUM_TEST} \
  --n_samples ${NUM_SAMPLES} \
  --center_crop \
  --no_flip
