#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/vqvae-main/run_vqvae.py run \
    --training_subjects="('/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered_CoordConv', 
                          '/nfs/home/apatel/Data/head_neck_data/CT_coordconv',
                          '/nfs/home/apatel/Data/head_neck_data/CT_coordconv',
                          '/nfs/home/apatel/Data/head_neck_data/CT_coordconv')" \
    --validation_subjects="('/nfs/home/apatel/Data/PET_Challenge/processed/CT_1.6_1.6_2.5_clipped_registered_CoordConv_validation', 
                          '/nfs/home/apatel/Data/head_neck_data/CT_coordconv_validation')" \
    --project_directory='/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/' \
    --experiment_name='vqgan_ne2048_dim128_CT_lower_disc_lr_4l' \
    --mode='training' \
    --device=0 \
    --amp=True \
    --deterministic=False \
    --cuda_benchmark=True \
    --seed=3 \
    --epochs=600 \
    --learning_rate=0.0001 \
    --gamma='auto' \
    --log_every=5 \
    --checkpoint_every=5 \
    --eval_every=5 \
    --loss='jukebox_perceptual' \
    --adversarial_component=True \
    --discriminator_network='baseline_discriminator' \
    --discriminator_learning_rate=0.0001 \
    --discriminator_loss='least_square' \
    --generator_loss='least_square' \
    --initial_factor_value=0 \
    --initial_factor_steps=25 \
    --max_factor_steps=50 \
    --max_factor_value=5 \
    --batch_size=3 \
    --normalize=False \
    --eval_batch_size=3 \
    --num_workers=4 \
    --prefetch_factor=4 \
    --starting_epoch=0 \
    --network='baseline_vqvae' \
    --use_subpixel_conv=False \
    --no_levels=4 \
    --no_res_layers=3 \
    --no_channels=256 \
    --codebook_type='ema' \
    --num_embeddings='(2048,)' \
    --embedding_dim='(128,)' \
    --max_decay_epochs=50 \
    --dropout_penultimate=True \
    --dropout_enc=0.0 \
    --dropout_dec=0.0 \
    --act='LEAKYRELU' \
    --apply_coordConv=True \
    --input_has_coordConv=True \
    --cropping_type='without_padding' \
