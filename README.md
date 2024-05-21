<h1 align="center">VQ-VAE and Transformer codebase for unsupervised Anomaly detection</h1>
<p align="center">
</p>


<p align="center">
  <img width="800" height="300" src="https://github.com/Ashayp31/vqvae_transformer_anomaly_detection/assets/62710884/684b0252-dbc5-4a7f-8469-cc3a6501049a">
</p>



## Intro
This codebase contains code for performing generative modelling of 3D medical imaging with diffusion models in addition to super-resolution tasks. It supports the use of DDPMs in addition to LDMs for higher dimensional data.
This work is the official implementation of [1] applied to 3D CT data. 
This codebase contains the code to perform unsupervised out-of-distribution detection with diffusion models.
It supports the use of DDPMs as well as Latent Diffusion Models (LDM) for dealing with higher dimensional data.
It is based on work published in [1] and [2].

[1] [Resolution and Field of View Invariant Generative Modelling with Latent Diffusion Models]([https://arxiv.org/abs/2211.07740](https://openreview.net/pdf?id=VHfh2J8MQ6))

## Setup

### Install
Create a fresh virtualenv (this codebase was developed and tested with Python 3.8) and then install the required packages:

```pip install -r requirements.txt```

You can also build the docker image
```bash
cd docker/
bash create_docker_image.sh
```


## Run with LDM

### Train VQVAE
```bash
python3 run_vqvae.py run \
    --training_subjects=${path_to_training_subjects} \
    --validation_subjects=${path_to_validation_subjects}  \
    --project_directory=${project_directory_path} \
    --experiment_name='vqgan_ne512_dim8_CT' \
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

The VQVAE training code is DistributedDataParallel (DDP) compatible. For example to train with 4 GPUs run with: mpirun -np 4 --oversubscribe --allow-run-as-root python3 run_vqvae.py run \
