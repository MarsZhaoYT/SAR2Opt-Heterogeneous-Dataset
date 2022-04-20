set -ex
MODEL='bicycle_gan'
# dataset details
CLASS='sar2opt'  # facades, day2night, edges2shoes, edges2handbags, maps
NZ=8
NO_FLIP='--no_flip'
DIRECTION='AtoB'
LOAD_SIZE=256
CROP_SIZE=256
INPUT_NC=3
NITER=200
NITER_DECAY=200

# training
GPU_ID=1
DISPLAY_ID=$((GPU_ID*10+1))
CHECKPOINTS_DIR=./checkpoint/${CLASS}/
NAME=${CLASS}_${MODEL}

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot ./datasets/combined_opt-sar \
  --name ${NAME} \
  --model ${MODEL} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --nz ${NZ} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --use_dropout