#!/bin/sh

mpirun -np 2 --oversubscribe --allow-run-as-root python3 /nfs/home/apatel/vqvae-main/run_vqvae.py run \
    --training_subjects='("/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/20",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/21",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/22",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/23",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/24",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/25",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/26",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/27",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/28",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/29",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/30",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/31",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/32",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/33",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/34",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/35",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/36",
			"/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/37")' \
    --validation_subjects='/nfs/home/apatel/Data/PET_Challenge/processed/CT_clipped_varying_resolution_and_fov/CT_1.6_1.6_2.5_clipped_registered_CoordConv_validation' \
    --project_directory='/nfs/home/apatel/CT_PET_FDG/CT_Diffusion_MultiRes/' \
    --experiment_name='vqgan_ne256_dim32_CT' \
    --mode='training' \
    --device='ddp' \
    --amp=True \
    --deterministic=False \
    --cuda_benchmark=True \
    --seed=3 \
    --epochs=100 \
    --learning_rate=0.0001 \
    --gamma='auto' \
    --log_every=1 \
    --checkpoint_every=1 \
    --eval_every=1 \
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
    --batch_size=3 \
    --normalize=False \
    --eval_batch_size=3 \
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
    --embedding_dim='(32,)' \
    --max_decay_epochs=50 \
    --dropout_penultimate=True \
    --dropout_enc=0.0 \
    --dropout_dec=0.0 \
    --act='LEAKYRELU' \
    --apply_coordConv=True \
    --input_has_coordConv=True \
    --cropping_type='without_padding' \
