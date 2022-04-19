dataroot='Image_translation_codes/contrastive-unpaired-translation/datasets/sar2opt'
name='sar2opt_CUT'


for ((i=1; i<=5; i++));
do
    echo "Processing"$i"th epoch..."
    
    python test.py --dataroot $dataroot \
                    --name $name \
                    --model cut
                   
done