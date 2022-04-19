set -ex

inputFolder="/workspace/Image_translation_codes/DATASET/SEN1_2/Splited_by_percent/10%_train/ROIs1158_spring_10%_train/"
modelRoot="/workspace/Image_translation_codes/NICE-GAN-pytorch/results/ROIs1158_spring_10%_train/model"
resultRoot="/workspace/Image_translation_codes/FID-Epoch/NICE-GAN/SEN1_2_10%_spring"
# outputFolder="/workspace/image_translation/FID-Epoch/MUNITresults/SEN1_2_10%_spring/images/"

for i in {10..19}
  do
    if (($i<10)); then
      tmpModel=$modelRoot"/params_00"$i"0000.pt"
      tmpOutputFolder=$resultRoot"/epoch"$i"/"
      echo "Loading model of Epoch:"$tmpModel
      
      CUDA_VISIBLE_DEVICES=1 python /workspace/Image_translation_codes/NICE-GAN-pytorch/main.py \
            --dataset $inputFolder \
            --dataset_phase train \
            --result_dir $tmpOutputFolder \
            --checkpoint $tmpModel \
            --phase 'test'
    
    else
      tmpModel=$modelRoot"/params_0"$i"0000.pt"
      tmpOutputFolder=$resultRoot"/epoch"$i"/"
      echo "Loading model:"$tmpModel
      
      CUDA_VISIBLE_DEVICES=1 python /workspace/Image_translation_codes/NICE-GAN-pytorch/main.py \
            --dataset $inputFolder \
            --dataset_phase train \
            --result_dir $tmpOutputFolder \
            --checkpoint $tmpModel \
            --phase 'test'
    fi
  done