set -ex
modelDir="/workspace/Image_translation_codes/pytorch-CycleGAN-and-pix2pix/checkpoint/SEN1_2_10%_spring_cyclegan/"
inputDataRoot="/workspace/Image_translation_codes/DATASET/SEN1_2/Splited_by_percent/10%_train/ROIs1158_spring_10%_train/"
resultsDirRoot="/workspace/Image_translation_codes/FID-Epoch/Cyclegan/SEN1_2_10%_spring/images/"
# outputFolder="/workspace/image_translation/FID-Epoch/MUNITresults/SEN1_2_10%_spring/images/"

for ((i=15; i<101; i+=5));
  do
      tmpOutputFolder="/workspace/Image_translation_codes/FID-Epoch/Cyclegan/SEN1_2_10%_spring/epoch"$i"/images/"
      
      echo "Loading model of Epoch:"$tmpModel
      python /workspace/Image_translation_codes/pytorch-CycleGAN-and-pix2pix/test.py --dataroot $inputDataRoot \
            --name SEN1_2_10%_spring_cyclegan \
            --model cycle_gan \
            --epoch $i \
            --dataset_phase train \
            --results_dir $tmpOutputFolder        

  done