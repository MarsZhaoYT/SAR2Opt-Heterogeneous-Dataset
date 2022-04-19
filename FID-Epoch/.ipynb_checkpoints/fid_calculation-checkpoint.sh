set -ex

# trainB训练集
sourceImageFolder="/workspace/Image_translation_codes/DATASET/SEN1_2/Splited_by_percent/10%_train/ROIs1158_spring_10%_train/trainB"

# 生成集fakeB的上级目录
targetImageRootFolder="/workspace/Image_translation_codes/FID-Epoch/AttnCycleGAN/SEN1_2_10%_spring"

for ((i=5; i<=100; i+=5))
    do
        # 定义待测试FID的生成集路径
        # tmpSourceImageFolder=$targetImageRootFolder"/epoch"$i"/real"

        # 其他方法用这句
        # tmpTargetImageFolder=$targetImageRootFolder"/epoch"$i"/fake"

        # AttnCycleGAN用这句
        tmpTargetImageFolder=$targetImageRootFolder"/epoch"

        echo "Calculating FID of Epoch"$i

        python metric/fid_kid.py $sourceImageFolder $tmpTargetImageFolder

    done