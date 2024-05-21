#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/vqvae-main/run_transformer.py run \
    --training_subjects="('/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/training_codes/0',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/training_codes/1',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/training_codes/2',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/training_codes/3',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/training_codes/4',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/training_codes/5',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/training_codes/6',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/training_codes/7',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/training_codes/8',
			'/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/training_codes/9')" \
    --validation_subjects='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/training_codes/10' \
    --spatial_encoding_path='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/spatial_encoding_locations_20.csv' \
    --project_directory='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/' \
    --experiment_name='CT_Generative_ne256_dim128' \
    --mode='training' \
    --deterministic=False \
    --cuda_benchmark=True \
    --device=0 \
    --seed=4 \
    --epochs=200 \
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
    --network='performer_ct_spatial_20' \
    --vocab_size=256 \
    --n_embd=512 \
    --n_layers=16 \
    --n_head=8 \
    --emb_dropout=0.0 \
    --ff_dropout=0.0 \
    --attn_dropout=0.0 \
    --fixed_ordering=False \
    --spatial_conditioning_buckets=8000 \
    --max_seq_len=14742 \
