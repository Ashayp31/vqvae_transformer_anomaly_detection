#!/bin/sh

CUDA_LAUNCH_BLOCKING=1 python3 /nfs/home/apatel/vqvae-main/anomaly_detection_conditional.py run \
    --training_subjects='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_ne256_dim128_crop_no_pad_coordconv/baseline_vqvae/testing_codes_pet_challenge/' \
    --validation_subjects='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_ne256_dim128_crop_no_pad_coordconv/baseline_vqvae/testing_codes_pet_challenge/' \
    --encoding_conditioning_path='/nfs/home/apatel/Data/combined_multicrop_vis/testing/pet_challenge_testing_conditioning.csv' \
    --spatial_encoding_path='/nfs/home/apatel/Data/combined_multicrop_vis/testing/pet_challenge_testing_conditioning.csv' \
    --cropping_file='/nfs/home/apatel/Data/combined_multicrop_vis/testing/pet_challenge_testing_conditioning.csv' \
    --project_directory='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/' \
    --experiment_name='vqgan_ne256_dim128_crop_no_pad_coordconv' \
    --vqvae_network_checkpoint='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_ne256_dim128_crop_no_pad_coordconv/baseline_vqvae/checkpoints/checkpoint_epoch=400.pt' \
    --transformer_network_checkpoint='/nfs/home/apatel/CT_PET_FDG/multi_crop_vis/vqgan_ne256_dim128_crop_no_pad_coordconv/performer_conditioned_no_pos_emb/checkpoints/checkpoint_epoch=80.pt' \
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
    --network='performer_conditioned_no_pos_emb' \
    --vocab_size_enc=256 \
    --vocab_size=256 \
    --n_embd=256 \
    --n_layers=12 \
    --n_head=8 \
    --emb_dropout=0.0 \
    --ff_dropout=0.0 \
    --attn_dropout=0.0 \
    --threshold=0.005 \
    --num_embeddings='(256,)' \
    --embedding_dim='(128,)' \
    --dropout_penultimate=True \
    --dropout_enc=0.0 \
    --dropout_dec=0.05 \
    --max_seq_len=19489 \
    --fixed_ordering=False \
    --position_emb="none" \
    --apply_coordConv=True





