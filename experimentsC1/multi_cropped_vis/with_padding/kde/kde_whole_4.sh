#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/vqvae-main/anomaly_detection_conditional.py run \
    --training_subjects='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_suv_15_jp_do_ne256_dim128_PET/baseline_vqvae/testing_codes/testing_4/' \
    --validation_subjects='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_suv_15_jp_do_ne256_dim128_PET/baseline_vqvae/testing_codes/testing_4/' \
    --encoding_conditioning_path='/nfs/home/apatel/Data/combined_datasets/multi_crop_vis/testing_pet_conditioning.csv' \
    --project_directory='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/' \
    --experiment_name='vqgan_suv_15_jp_do_ne256_dim128_PET' \
    --vqvae_network_checkpoint='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_suv_15_jp_do_ne256_dim128_PET/baseline_vqvae/checkpoints/checkpoint_epoch=1000.pt' \
    --transformer_network_checkpoint='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_suv_15_jp_do_ne256_dim128_PET/performer/checkpoints/checkpoint_epoch=120.pt' \
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
    --num_passes_dropout=4 \
    --num_passes_sampling=60


