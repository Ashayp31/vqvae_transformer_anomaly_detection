#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/vqvae-main/anomaly_detection_conditional.py run \
    --training_subjects='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_ne512_dim256_rotations_xyz_crops/baseline_vqvae/testing_rotated_codes_short/' \
    --validation_subjects='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_ne512_dim256_rotations_xyz_crops/baseline_vqvae/testing_rotated_codes_short/' \
    --encoding_conditioning_path='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_ne512_dim256_rotations_xyz_crops/testing_rotated_spatial_encoding_locations.csv' \
    --project_directory='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/' \
    --experiment_name='vqgan_ne512_dim256_rotations_xyz_crops' \
    --vqvae_network_checkpoint='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_ne512_dim256_rotations_xyz_crops/baseline_vqvae/checkpoints/checkpoint_epoch=250.pt' \
    --transformer_network_checkpoint='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_ne512_dim256_rotations_xyz_crops/performer_no_spatial_aug/checkpoints/checkpoint_epoch=103.pt' \
    --infer_mode='kde_zscore' \
    --deterministic=False \
    --cuda_benchmark=True \
    --device=0 \
    --seed=4 \
    --batch_size=1 \
    --eval_batch_size=1 \
    --num_workers=8 \
    --prefetch_factor=8 \
    --starting_epoch=0 \
    --network='performer_no_spatial_aug' \
    --vocab_size_enc=512 \
    --vocab_size=512 \
    --n_embd=256 \
    --n_layers=12 \
    --n_head=8 \
    --emb_dropout=0.0 \
    --ff_dropout=0.0 \
    --attn_dropout=0.0 \
    --threshold=0.005 \
    --num_embeddings='(512,)' \
    --embedding_dim='(256,)' \
    --dropout_penultimate=True \
    --dropout_enc=0.0 \
    --dropout_dec=0.0 \
    --max_seq_len=19489 \
    --fixed_ordering=False \
    --apply_coordConv=True





