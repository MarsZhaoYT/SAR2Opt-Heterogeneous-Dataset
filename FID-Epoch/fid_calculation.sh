set -ex

# trainB训练集
sourceImageFolder="/workspace/Image_translation_codes/ASGIT/datasets/SEN1-2/10%_train/ROIs1158_spring_10%_train/trainB"

# 生成集fakeB的上级目录
targetImageRootFolder="/workspace/Image_translation_codes/FID-Epoch/AttnCyclegan/SEN1-2_10%_spring/images"

out_txt='/workspace/Image_translation_codes/FID-Epoch/output_attncyclegan_SEN1_2_10%_spring.txt'

for ((i=5; i<=100; i+=5))
    do
        # 定义待测试FID的生成集路径
        tmpSourceImageFolder=$targetImageRootFolder"/epoch"$i"/real/"

        # 其他方法用这句
        tmpTargetImageFolder=$targetImageRootFolder"/epoch"$i"/fake/"

        echo "Calculating FID of Epoch"$i

        python /workspace/Image_translation_codes/FID-Epoch/metric/fid_kid.py $tmpSourceImageFolder $tmpTargetImageFolder > $out_txt

    done