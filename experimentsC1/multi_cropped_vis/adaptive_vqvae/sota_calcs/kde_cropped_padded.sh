#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/vqvae-main/anomaly_detection_conditional.py run \
    --training_subjects='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_ne256_dim128_original/baseline_vqvae/testing_codes_cropped_padded/' \
    --validation_subjects='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_ne256_dim128_original/baseline_vqvae/testing_codes_cropped_padded/' \
    --encoding_conditioning_path='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_ne256_dim128_original/baseline_vqvae/conditioning_cropped_padded_testing.csv' \
    --project_directory='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/' \
    --experiment_name='vqgan_ne256_dim128_original' \
    --vqvae_network_checkpoint='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_ne256_dim128_original/baseline_vqvae/checkpoints/checkpoint_epoch=370.pt' \
    --transformer_network_checkpoint='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_ne256_dim128_original/performer/checkpoints/checkpoint_epoch=80.pt' \
    --infer_mode='zscore_kde' \
    --deterministic=False \
    --cuda_benchmark=True \
    --device=0 \
    --seed=4 \
    --batch_size=1 \
    --eval_batch_size=1 \
    --num_workers=8 \
    --prefetch_factor=8 \
    --starting_epoch=0 \
    --network='performer' \
    --vocab_size_enc=256 \
    --vocab_size=256 \
    --n_embd=256 \
    --n_layers=14 \
    --n_head=8 \
    --emb_dropout=0.0 \
    --ff_dropout=0.0 \
    --attn_dropout=0.0 \
    --threshold=0.01 \
    --num_embeddings='(256,)' \
    --embedding_dim='(128,)' \
    --dropout_penultimate=True \
    --dropout_enc=0.0 \
    --dropout_dec=0.05 \
    --num_passes_dropout=20 \
    --num_passes_sampling=20


