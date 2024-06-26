#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/vqvae-main/run_vqvae.py run \
    --training_subjects='/nfs/home/apatel/Data/combined_datasets/CT/validation/' \
    --validation_subjects='/nfs/home/apatel/Data/combined_datasets/CT/validation/' \
    --project_directory='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/' \
    --experiment_name='vqgan_ne256_dim64_crop_no_pad_coordconv_CT' \
    --cropping_file='/nfs/home/apatel/Data/combined_multicrop_vis/ct_conditioning_encoding.csv' \
    --mode='extracting' \
    --device=0 \
    --amp=True \
    --deterministic=False \
    --cuda_benchmark=True \
    --seed=3 \
    --epochs=400 \
    --learning_rate=0.0001 \
    --gamma='auto' \
    --log_every=10 \
    --checkpoint_every=10 \
    --eval_every=10 \
    --loss='jukebox_perceptual' \
    --adversarial_component=True \
    --discriminator_network='baseline_discriminator' \
    --discriminator_learning_rate=0.0005 \
    --discriminator_loss='least_square' \
    --generator_loss='least_square' \
    --initial_factor_value=0 \
    --initial_factor_steps=25 \
    --max_factor_steps=50 \
    --max_factor_value=5 \
    --batch_size=1 \
    --normalize=False \
    --eval_batch_size=1 \
    --num_workers=8 \
    --prefetch_factor=8 \
    --starting_epoch=0 \
    --network='baseline_vqvae' \
    --use_subpixel_conv=False \
    --no_levels=3 \
    --no_res_layers=3 \
    --no_channels=256 \
    --codebook_type='ema' \
    --num_embeddings='(256,)' \
    --embedding_dim='(64,)' \
    --max_decay_epochs=50 \
    --dropout_penultimate=True \
    --dropout_enc=0.0 \
    --dropout_dec=0.0 \
    --act='LEAKYRELU' \
    --cropping_type='without_padding' \
    --apply_coordConv=True
