# 修改此处数据集路径
path='./datasets/SEN1-2/20%_train/ROIs1158_spring_20%_train'

# 修改此处
name='re-SEN1-2_20%_spring'

# 修改此处数据集样本数量
data_length=3898

# （弃用）10%: spring: 1948   summer: 1220   fall: 2028   winter: 1012
# 20%: spring: 3898   summer: 2440   fall: 4058   winter: 2025
# （弃用）50%: spring: 9746   summer: 6102   fall: 10145   winter: 5064
# SAR2Opt: 1450

# train_attn_v2.py中修复了使用visdom存储数据时，训练到epoch=20就自动中断的bug
python train_attn_v2.py \
       --netG resnet_9blocks \
       --netD posthoc_attn \
       --model attn_cycle_gan \
       --concat rmult \
       --dataroot $path \
       --name $name \
       --mask_size 256 \
       --data_length $data_length