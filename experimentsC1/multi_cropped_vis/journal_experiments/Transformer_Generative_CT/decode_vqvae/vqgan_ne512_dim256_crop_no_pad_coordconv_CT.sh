#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/vqvae-main/run_vqvae.py run \
    --training_subjects='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/training_codes/0' \
    --validation_subjects='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/training_codes/0' \
    --project_directory='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/' \
    --experiment_name='CT_Generative_ne256_dim128' \
    --mode='decoding' \
    --device=0 \
    --amp=True \
    --deterministic=False \
    --cuda_benchmark=True \
    --seed=3 \
    --epochs=200 \
    --learning_rate=0.0001 \
    --gamma='auto' \
    --log_every=5 \
    --checkpoint_every=5 \
    --eval_every=5 \
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
    --embedding_dim='(128,)' \
    --max_decay_epochs=50 \
    --dropout_penultimate=True \
    --dropout_enc=0.0 \
    --dropout_dec=0.0 \
    --act='LEAKYRELU' \
    --apply_coordConv=True \
    --input_has_coordConv=True
