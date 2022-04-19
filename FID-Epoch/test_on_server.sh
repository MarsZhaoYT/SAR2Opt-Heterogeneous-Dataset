configPath="/workspace/image_translation/MUNIT/configs/SEN1_2_10%_spring.yaml"
inputFolder="/workspace/image_translation/datasets/SEN1_2/10%_train/ROIs1158_spring_10%_train/trainA"
# outputFolder="/workspace/image_translation/FID-Epoch/MUNITresults/SEN1_2_10%_spring/images/"

for i in {10..15}
  do
    if (($i<10)); then
      tmpModel="/workspace/image_translation/MUNIT/outputs/SEN1_2_10%_spring/checkpoint/gen_000"$i"0000.pt"
      tmpOutputFolder="/workspace/image_translation/FID-Epoch/MUNIT/results/SEN1_2_10%_spring/epoch"$i"/images"
      echo "Loading model of Epoch:"$tmpModel
      python /workspace/image_translation/MUNIT/test.py --config $configPath \
                          --input_folder $inputFolder \
                          --output_folder $tmpOutputFolder \
                          --checkpoint $tmpModel \
                          --a2b 1
    else
      tmpModel="/workspace/image_translation/MUNIT/outputs/SEN1_2_10%_spring/checkpoint/gen_00"$i"0000.pt"
      tmpOutputFolder="/workspace/image_translation/FID-Epoch/MUNIT/results/SEN1_2_10%_spring/epoch"$i"/images"
      echo "Loading model:"$tmpModel
      python /workspace/image_translation/MUNIT/test.py --config $configPath \
                          --input_folder $inputFolder \
                          --output_folder $tmpOutputFolder \
                          --checkpoint $tmpModel \
                          --a2b 1
    fi
  done