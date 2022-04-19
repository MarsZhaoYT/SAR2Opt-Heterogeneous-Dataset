path='./datasets/SEN1-2/10%_train/ROIs1158_spring_10%_train'

python train_attn.py \
       --netG resnet_9blocks \
       --netD posthoc_attn \
       --model attn_cycle_gan \
       --concat rmult \
       --dataroot $path \
       --name SEN1-2_10%_spring \
       --mask_size 256