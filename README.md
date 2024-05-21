<h1 align="center">VQ-VAE and Transformer codebase for unsupervised Anomaly detection</h1>
<p align="center">
</p>


<p align="center">
  <img width="800" height="300" src="https://github.com/Ashayp31/vqvae_transformer_anomaly_detection/assets/62710884/684b0252-dbc5-4a7f-8469-cc3a6501049a">
</p>



## Intro
This codebase contains code for training a VQ-VAE or VQ-GAN model coupled with a performer (linear approximation to full attention in a Transformer) for generative modelling and anomaly detection.
It is based on work published in [1] with further scripts attributed to the work in [2].

[1] [Cross Attention Transformers for Multi-modal Unsupervised Whole-Body PET Anomaly Detection] - https://arxiv.org/abs/2304.07147

[2] [Self-Supervised Anomaly Detection from Anomalous Training Data via Iterative Latent Token Masking] - https://ieeexplore.ieee.org/document/10350925

## Setup

### Install
Create a fresh virtualenv (this codebase was developed and tested with Python 3.8) and then install the required packages:

```pip install -r requirements.txt```

You can also build the docker image using the provided docker file


### Train VQVAE
```bash
python3 run_vqvae.py run \
    --training_subjects=${path_to_training_subjects} \
    --validation_subjects=${path_to_validation_subjects}  \
    --project_directory=${project_directory_path} \
    --experiment_name='vqgan_ne256_dim64_PET' \
    --mode='training' \
    --device=0 \
    --amp=True \
    --deterministic=False \
    --cuda_benchmark=True \
    --seed=3 \
    --epochs=200 \
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
    --num_embeddings='(512,)' \
    --embedding_dim='(8,)' \
    --max_decay_epochs=50 \
    --dropout_penultimate=True \
    --dropout_enc=0.0 \
    --dropout_dec=0.0 \
    --act='LEAKYRELU' \
    --apply_coordConv=True \
    --input_has_coordConv=False \
    --cropping_type='without_padding' \
```

The VQVAE training code is DistributedDataParallel (DDP) compatible. For example to train with 4 GPUs run with:
```bash
mpirun -np 4 --oversubscribe --allow-run-as-root python3 run_vqvae.py run
```



### Train Transformer
```bash
python3 run_transformer.py run \
    --training_subjects=${path_to_training_subjects} \
    --validation_subjects=${path_to_validation_subjects}  \
    --project_directory=${project_directory_path} \
    --encoding_conditioning_path=${path_to_conditoinings} \
    --experiment_name="performer_PET" \
    --mode='training' \
    --deterministic=False \
    --cuda_benchmark=True \
    --device=0 \
    --seed=4 \
    --epochs=25 \
    --learning_rate=0.001 \
    --gamma='auto' \
    --log_every=1 \
    --checkpoint_every=1\
    --eval_every=5 \
    --batch_size=1 \
    --eval_batch_size=1 \
    --num_worker=16 \
    --prefetch_factor=10 \
    --starting_epoch=0 \
    --network='performer' \
    --vocab_size=256 \
    --n_embed=256 \
    --n_layers=22 \
    --n_head=8 \
    --emb_dropout=0.0 \
    --ff_dropout=0.0 \
    --attn_dropout=0.0 \
    --spatial_position_emb="absolute"
```
To train the transformer one can encode images on the fly using the trained VQ-VAE, for this the VQ-VAE checkpoint and network parameters must be included as inputs. Or you can encode the images before and have them in a folder as the "training/validation_subjects path".
Further work using latent token masking can be used by adding a path for the masking files using the input "token_masking_path". 

For the encoding_conditioning_path and token_masking_path a csv file should be provided with 2 columns: "subject" and "conditioning" where the subject is the subject name and conditioning value is the path to the conditioning for that subject.

### Anomaly Detection
```bash
python3 anomaly_detection_conditional.py run \
    --training_subjects=${path_to_training_subjects} \
    --validation_subjects=${path_to_validation_subjects}  \
    --project_directory=${project_directory_path} \
    --encoding_conditioning_path=${path_to_conditoinings} \
    --experiment_name="performer_PET" \    
    --vqvae_network_checkpoint='/results/vqgan_ne256_dim64_PET/baseline_vqvae/checkpoints/checkpoint_epoch=600.pt' \
    --transformer_network_checkpoint='/results/performer_PET/performer/checkpoints/checkpoint_epoch=200.pt' \
    --infer_mode='anomaly_detection' \
    --deterministic=False \
    --cuda_benchmark=True \
    --device=0 \
    --seed=4 \
    --gamma='auto' \
    --batch_size=1 \
    --eval_batch_size=1 \
    --num_workers=8 \
    --prefetch_factor=8 \
    --starting_epoch=0 \
    --network='performer' \
    --vocab_size_enc256 \
    --vocab_size=256 \
    --n_embd=256 \
    --n_layers=22 \
    --n_head=8 \
    --conditioning_type='cross_attend' \
    --emb_dropout=0.0 \
    --ff_dropout=0.0 \
    --attn_dropout=0.0 \
    --threshold=0.01 \
    --num_embeddings='(256,)' \
    --embedding_dim='(128,)' \
    --dropout_penultimate=True \
    --dropout_enc=0.0 \
    --dropout_dec=0.0"
```

To run anomaly detection inference you can use the following script above. For this a likelihood threshold needs to be defined for the likelihood on whether to resample tokens.
You can also use the KDE inference method by changing infer_mode to "kde_zscore" to generate multiple healed representations of the original image.
