set -ex
modelDir="/workspace/Image_translation_codes/pytorch-CycleGAN-and-pix2pix/checkpoint/SEN1_2_10%_spring_pix2pix/"
inputDataRoot="/workspace/Image_translation_codes/DATASET/combined_SEN1_2/10%_train/ROIs1158_spring/"
resultsDirRoot="/workspace/Image_translation_codes/FID-Epoch/pix2pix/SEN1_2_10%_spring/images/"
# outputFolder="/workspace/image_translation/FID-Epoch/MUNITresults/SEN1_2_10%_spring/images/"

for ((i=5; i<101; i+=5));
  do
      tmpOutputFolder="/workspace/Image_translation_codes/FID-Epoch/pix2pix/SEN1_2_10%_spring/epoch"$i"/images/"
      
      # echo "Loading model of Epoch:"$tmpModel
      python /workspace/Image_translation_codes/pytorch-CycleGAN-and-pix2pix/test.py 
            --dataroot $inputDataRoot \
            --name SEN1_2_10%_spring_pix2pix \
            --model pix2pix \
            --epoch $i \
            --dataset_phase train \
            --results_dir $tmpOutputFolder \
            --direction AtoB     

  done