#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/vqvae-main/run_transformer.py run \
    --training_subjects="('/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/vqgan_ne256_dim128_PET/baseline_vqvae/training_codes/0',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/vqgan_ne256_dim128_PET/baseline_vqvae/training_codes/1',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/vqgan_ne256_dim128_PET/baseline_vqvae/training_codes/2',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/vqgan_ne256_dim128_PET/baseline_vqvae/training_codes/3',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/vqgan_ne256_dim128_PET/baseline_vqvae/training_codes/4',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/vqgan_ne256_dim128_PET/baseline_vqvae/training_codes/5',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/vqgan_ne256_dim128_PET/baseline_vqvae/training_codes/6',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/vqgan_ne256_dim128_PET/baseline_vqvae/training_codes/7',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/vqgan_ne256_dim128_PET/baseline_vqvae/training_codes/8',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/vqgan_ne256_dim128_PET/baseline_vqvae/training_codes/9',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/vqgan_ne256_dim128_PET/baseline_vqvae/training_codes/10',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/vqgan_ne256_dim128_PET/baseline_vqvae/training_codes/11',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/vqgan_ne256_dim128_PET/baseline_vqvae/training_codes/12',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/vqgan_ne256_dim128_PET/baseline_vqvae/training_codes/13',)" \
    --validation_subjects='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/vqgan_ne256_dim128_PET/baseline_vqvae/training_codes/val' \
    --spatial_encoding_path='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/spatial_encoding_locations_combined.csv' \
    --project_directory='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/' \
    --experiment_name='vqgan_ne256_dim128_PET' \
    --mode='training' \
    --deterministic=False \
    --cuda_benchmark=True \
    --device=0 \
    --seed=4 \
    --epochs=80 \
    --learning_rate=0.001 \
    --gamma='auto' \
    --log_every=1 \
    --checkpoint_every=1 \
    --eval_every=5 \
    --batch_size=1 \
    --eval_batch_size=1 \
    --num_workers=8 \
    --prefetch_factor=8 \
    --starting_epoch=0 \
    --network='performer_pet_spatial' \
    --vocab_size=256 \
    --n_embd=256 \
    --n_layers=12 \
    --n_head=8 \
    --emb_dropout=0.0 \
    --ff_dropout=0.0 \
    --attn_dropout=0.0 \
    --fixed_ordering=False \
    --max_seq_len=21505 \
