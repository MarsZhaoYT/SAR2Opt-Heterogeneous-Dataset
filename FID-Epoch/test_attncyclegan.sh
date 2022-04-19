# 该脚本用于循环调用ASGIT/test.py，对每个epoch的模型文件生成对应的fake optical图像

set -ex
modelDir="/workspace/Image_translation_codes/ASGIT/checkpoint/SEN1-2_10%_spring/"
inputDataRoot="/workspace/Image_translation_codes/ASGIT/datasets/SEN1-2/10%_train/ROIs1158_spring_10%_train/"
resultsDirRoot="/workspace/Image_translation_codes/FID-Epoch/AttnCyclegan/SEN1-2_10%_spring/images/"

# mkdir resultsDirRoot
# outputFolder="/workspace/image_translation/FID-Epoch/MUNITresults/SEN1_2_10%_spring/images/"

for ((i=5; i<101; i+=5));
  do
      # 查看test.py中是否有os.makedirs()操作
      tmpOutputFolder="/workspace/Image_translation_codes/FID-Epoch/AttnCyclegan/SEN1-2_10%_spring/epoch"$i"/images/"
      
      echo "Loading model of Epoch:"$i
     
      CUDA_VISIBLE_DEVICES=1 python /workspace/Image_translation_codes/ASGIT/test.py \
             --netG resnet_9blocks \
             --netD posthoc_attn \
             --model attn_cycle_gan \
             --concat rmult \
             --epoch $i \
             --dataroot $inputDataRoot \
             --name $modelDir \
             --dataset_phase train \
             --results_dir $tmpOutputFolder

  done