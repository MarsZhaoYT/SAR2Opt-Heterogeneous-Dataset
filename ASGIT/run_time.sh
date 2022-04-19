# 修改此处数据集路径
path='/workspace/Image_translation_codes/ASGIT/datasets/SAR2Opt'
# path='./datasets/SAR2Opt'

# 修改此处
name='SAR2Opt'
# name='SAR2Opt'


for ((i=1; i<=5; i++));
do
    echo "Processing"$i"th epoch..."
    
    python /workspace/Image_translation_codes/ASGIT/test.py \
               --netG resnet_9blocks \
               --netD posthoc_attn \
               --model attn_cycle_gan \
               --concat rmult \
               --dataroot $path \
               --name $name
                   
done