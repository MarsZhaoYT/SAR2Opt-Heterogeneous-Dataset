# 修改此处数据集路径
path='./datasets/SEN1-2/10%_train/ROIs1158_spring_10%_train'


# 参数中需要修改的：
# （1）--name：实验结果存储路径
# （2）--data_length：数据集长度
#      10%: spring: 1948   summer: 1220   fall: 2028   winter: 1012
#      20%: spring: 3898   summer: 2440   fall: 4058   winter: 2025
#      30%: spring: 9746   summer: 6102   fall: 10145   winter: 5064

python train_attn_v2.py \
       --netG resnet_9blocks \
       --netD posthoc_attn \
       --model attn_cycle_gan \
       --concat rmult \
       --dataroot $path \
       --name SEN1-2_10%_spring \
       --mask_size 256 \
       --data_length 1948