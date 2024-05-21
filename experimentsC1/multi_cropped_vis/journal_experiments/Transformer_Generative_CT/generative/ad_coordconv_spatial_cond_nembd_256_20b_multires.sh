#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/vqvae-main/anomaly_detection_conditional.py run \
    --training_subjects='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/training_codes/11' \
    --validation_subjects='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/training_codes/11' \
    --spatial_encoding_path='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/spatial_encoding_locations_20_multires.csv' \
    --project_directory='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/' \
    --experiment_name='CT_Generative_ne256_dim128' \
    --vqvae_network_checkpoint='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/baseline_vqvae/checkpoints/checkpoint_epoch=185.pt' \
    --transformer_network_checkpoint='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis_partial_body_training/CT_Generative_ne256_dim128/performer_ct_spatial_20/checkpoints/checkpoint_65.pt' \
    --infer_mode='anomaly_detection' \
    --deterministic=False \
    --cuda_benchmark=True \
    --device=0 \
    --seed=4 \
    --batch_size=1 \
    --eval_batch_size=1 \
    --num_workers=8 \
    --prefetch_factor=8 \
    --starting_epoch=0 \
    --network='performer_ct_spatial_20' \
    --vocab_size_enc=256 \
    --vocab_size=256 \
    --n_embd=512 \
    --n_layers=16 \
    --n_head=8 \
    --emb_dropout=0.0 \
    --ff_dropout=0.0 \
    --attn_dropout=0.0 \
    --threshold=0.95 \
    --num_embeddings='(256,)' \
    --embedding_dim='(128,)' \
    --dropout_penultimate=True \
    --dropout_enc=0.0 \
    --dropout_dec=0.0 \
    --fixed_ordering=False \
    --spatial_conditioning_buckets=8000 \
    --max_seq_len=14742 \
    --apply_coordConv=True





