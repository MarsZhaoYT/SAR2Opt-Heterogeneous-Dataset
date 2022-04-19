# set -ex


for ((i=1; i<=5; i++));
do
    echo "Processing"$i"th epoch..."
    # --- pix2pix ---
    # python Image_translation_codes/pytorch-CycleGAN-and-pix2pix/test.py \
    #                --dataroot Image_translation_codes/pytorch-CycleGAN-and-pix2pix/datasets/combined_sar2opt \
    #                --name sar2opt_pix2pix \
    #                --model pix2pix \
    #                --direction AtoB

    # --- cyclegan ---
    python Image_translation_codes/pytorch-CycleGAN-and-pix2pix/test.py \
                   --dataroot Image_translation_codes/pytorch-CycleGAN-and-pix2pix/datasets/sar2opt \
                   --name sar2opt_cyclegan \
                   --model cycle_gan 
                   
done