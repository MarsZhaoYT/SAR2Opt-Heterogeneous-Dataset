path='./datasets/SEN1-2/10%_train/ROIs1158_spring_10%_train'

# train_attn_v2.py中修复了使用visdom存储数据时，训练到epoch=20就自动中断的bug
python train_attn_v2.py \
       --netG resnet_9blocks \
       --netD posthoc_attn \
       --model attn_cycle_gan \
       --concat rmult \
       --dataroot $path \
       --name SEN1-2_10%_spring \
       --mask_size 256