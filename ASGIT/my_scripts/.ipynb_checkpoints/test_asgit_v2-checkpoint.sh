# 修改此处数据集路径
path='./datasets/SAR2Opt'

# 修改此处
name='SAR2Opt'

# 修改此处测试集样本数量(test.py无需修改)
# 10%: spring: 488   summer: 1220   fall: 2028   winter: 1012
# 20%: spring: 3898   summer: 2440   fall: 4058   winter: 2025
# 50%: spring: 9746   summer: 6102   fall: 10145   winter: 5064
# data_length=1220


python test.py --netG resnet_9blocks \
               --netD posthoc_attn \
               --model attn_cycle_gan \
               --concat rmult \
               --dataroot $path \
               --name $name
