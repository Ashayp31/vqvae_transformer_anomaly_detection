#!/usr/bin/env python3
import os
import numpy as np
from monai.engines import SupervisedEvaluator
from monai.handlers import (
    SegmentationSaver,
)
from typing import Tuple, Union
import deepspeed

from monai.engines.utils import CommonKeys
from ignite.contrib.handlers import ProgressBar
from src.inferer.conditional_transformer import (
    TransformerConditionalInferer,
    LikelihoodMapInferer,
    VQVAETransformerUncertaintyCombinedInferer,
    VQVAETransformerKDEInferer,
    VQVAETransformerKDEZscoreInferer
)
import torch
from src.networks.vqvae.configure import get_vqvae_network
from src.networks.transformers.configure import get_transformer_network
from monai.handlers import CheckpointSaver, LrScheduleHandler, CheckpointLoader
from src.utils.transformer import (
    get_data_flow,
    prepare_batch,
)
from src.utils.vqvae import get_batch_transform
from src.utils.constants import TransformerConditioningType

from src.handlers.general import (
    NpySaver,
)

from fire import Fire

from src.networks.transformers.img2seq_ordering import (
    Ordering,
    OrderingType,
    OrderingTransformations,
    RelativeSpatialPositioning
)

from src.networks.transformers.performer import Performer
from src.utils.general import basic_initialization, log_network_size


def inference(config: dict):

    logger, device = basic_initialization(
        config=config, logger_name="Transformer-Training"
    )



    if config["conditionings"]:
        config["use_continuous_conditioning"] = (
            list(config["use_continuous_conditioning"])
            if type(config["use_continuous_conditioning"]) is tuple
            else [config["use_continuous_conditioning"]]
            * len(config["conditionings"])
        )
    else:
        config["use_continuous_conditioning"] = []

    training_loader, evaluation_loader, training_evaluation_loader = get_data_flow(
        config=config, logger=logger
    )


    _, input_height, input_width = next(iter(training_loader))[
        "quantization"
    ].shape

    embedding_shape = next(iter(training_loader))[
        "quantization"
    ].shape

    config["spatial_shape"] = next(iter(training_loader))["quantization"].shape[1:]

    ordering = Ordering(
        ordering_type=config["ordering_type"],
        spatial_dims=2,
        dimensions=(1, input_height, input_width),
        reflected_spatial_dims=config["reflected_spatial_dims"],
        transpositions_axes=config["transpositions_axes"],
        rot90_axes=config["rot90_axes"],
        transformation_order=config["transformation_order"],
    )
    config["ordering"] = ordering


    config["relative_spatial_pos_attr"] = None

    transformer_network = get_transformer_network(config).to(device)
    log_network_size(network=transformer_network, logger=logger)

    if config["device"] == "ddp":
        transformer_network = torch.nn.parallel.DistributedDataParallel(
            transformer_network,
            device_ids=[config["local_rank"]],
            broadcast_buffers=True,
            find_unused_parameters=True,
            bucket_cap_mb=12.5,
        )


    config["transformer_network"] =  config["network"]
    config["network"] = config["vqvae_network"]
    vqvae_network = get_vqvae_network(config=config)
    #vqvae_checkpoint = torch.load(config["vqvae_network_checkpoint"])["network"]

    if config["infer_mode"] == "anomaly_detection":
        thresholds = [0.0001, 0.00025,0.0005,0.001,0.0025,0.005,0.01]
        #thresholds = [0.00025,0.0005,0.001]
    else:
        thresholds = [0.0001, 0.00025,0.0005,0.001,0.0025,0.005,0.01]
        #thresholds = [0.0001]
        #thresholds = [config["threshold"]]

    for thresh in thresholds:
        config["threshold"] = thresh
        engine = SupervisedEvaluator(
            device=device,
            val_data_loader=evaluation_loader,
            inferer=TransformerConditionalInferer(device=device,
                                                  threshold=config["threshold"],
                                                    embedding_shape=embedding_shape,
                                                  vqvae_net=vqvae_network,
                                                  #vqvae_checkpoint=vqvae_checkpoint,
                                                  clip_encoding= True if config["transformer_network"] == "performer" else False)
            if config["infer_mode"] == "anomaly_detection"
            else LikelihoodMapInferer(device=device, embedding_shape=embedding_shape, clip_encoding=True if config["transformer_network"] == "performer" else False)
            if config["infer_mode"] == "likelihood_map"
            else VQVAETransformerUncertaintyCombinedInferer(device=device,
                                                            threshold=config["threshold"],
                                                            embedding_shape=embedding_shape,
                                                            vqvae_net=vqvae_network,
                                                            num_passes_sampling=config["num_passes_sampling"],
                                                            num_passes_dropout=config["num_passes_dropout"],
                                                            clip_encoding= True if config["transformer_network"] == "performer" else False)
            if config["infer_mode"] == "zscore"
            else VQVAETransformerKDEInferer(device=device,
                                                            threshold=config["threshold"],
                                                            embedding_shape=embedding_shape,
                                                            vqvae_net=vqvae_network,
                                                            num_passes_sampling=config["num_passes_sampling"],
                                                            num_passes_dropout=config["num_passes_dropout"],
                                                            clip_encoding= True if config["transformer_network"] == "performer" else False,
                                                            min_bandwidth = config["min_bandwidth"])
            if config["infer_mode"] == "kde"
            else VQVAETransformerKDEZscoreInferer(device=device,
                                                            threshold=config["threshold"],
                                                            embedding_shape=embedding_shape,
                                                            vqvae_net=vqvae_network,
                                                            num_passes_sampling=config["num_passes_sampling"],
                                                            num_passes_dropout=config["num_passes_dropout"],
                                                            clip_encoding= True if config["transformer_network"] == "performer" else False)
                ,
                non_blocking=True,
                network=transformer_network,
                prepare_batch=lambda batch, pb_device, non_blocking: prepare_batch(
                    batch,
                    transformer_network.module.ordering.get_sequence_ordering()
                    if config["device"] == "ddp"
                    else transformer_network.ordering.get_sequence_ordering(),
                    config["vocab_size"],
                    conditionings=config["conditionings"],
                    conditioned_encoding=config["conditioned_encoding"],
                    vocab_size_enc=config["vocab_size_enc"],
                    device=pb_device,
                    non_blocking=non_blocking,
                    inference=True,
                    original_nii_path=config["original_nii_path"],
                ),
                amp=False,
                val_handlers=[],
            )
    
        #transformer_checkpoint = torch.load(config["transformer_network_checkpoint"], map_location=device)["network"]
        #transformer_network.load_state_dict(transformer_checkpoint).attach(engine)
    
        CheckpointLoader(
            load_path=config["transformer_network_checkpoint"],
            load_dict={"network": transformer_network},
            map_location=device,
        ).attach(engine)
    
        CheckpointLoader(
            load_path=config["vqvae_network_checkpoint"],
            load_dict={"network": vqvae_network},
            map_location=device,
        ).attach(engine)
    
    
        if config["infer_mode"] == "anomaly_detection":
            name_addition = "th_" + str(config["threshold"]).replace(".", "_")
            NpySaver(
                output_dir=config["outputs_directory"],
                output_postfix=name_addition + "_healed",
                dtype=np.dtype(np.float32),
                batch_transform=lambda batch: batch["quantization_meta_dict"],
                output_transform=lambda output: output[CommonKeys.PRED]["resampled"],
            ).attach(engine)
            NpySaver(
                output_dir=config["outputs_directory"],
                output_postfix=name_addition + "_resampling_mask",
                dtype=np.dtype(np.float32),
                batch_transform=lambda batch: batch["quantization_meta_dict"],
                output_transform=lambda output: output[CommonKeys.PRED]["resampling_mask"],
            ).attach(engine)

    
        elif config["infer_mode"] == "likelihood_map":
            name_addition = "llmap"
            NpySaver(
                output_dir=config["outputs_directory"],
                output_postfix=name_addition,
                dtype=np.dtype(np.float32),
                batch_transform=lambda batch: batch["quantization_meta_dict"],
                output_transform=lambda output: output[CommonKeys.PRED]["llmap"],
            ).attach(engine)
        elif config["infer_mode"] == "zscore":
            name_addition = "combined_th_" + str(config["threshold"]).replace(".", "_") + "_llmap_" + str(
                config["use_prior_llmap"]) + "_npd_" + \
                                str(config["num_passes_dropout"]) + "_nps_" + str(
                    config["num_passes_sampling"])
            SegmentationSaver(
                output_dir=config["outputs_directory"],
                output_postfix=name_addition + "_std",
                output_ext=".nii.gz",
                resample=False,
                scale=None,
                dtype=np.dtype(np.float32),
                batch_transform=lambda batch: batch["quantization_meta_dict"],
                output_transform=lambda output: output[CommonKeys.PRED]["std"],
            ).attach(engine)
            SegmentationSaver(
                output_dir=config["outputs_directory"],
                output_postfix=name_addition + "_mean",
                output_ext=".nii.gz",
                resample=False,
                scale=None,
                dtype=np.dtype(np.float32),
                batch_transform=lambda batch: batch["quantization_meta_dict"],
                output_transform=lambda output: output[CommonKeys.PRED]["mean"],
            ).attach(engine)
        elif config["infer_mode"] == "kde":
            if config["conditionings"]:
                condition_add_name = "_conditioned"
            else:
                condition_add_name = ""
            name_addition = "combined_th_" + str(config["threshold"]).replace(".", "_") + "_llmap_" + str(
                config["use_prior_llmap"]) + "_npd_" + \
                                str(config["num_passes_dropout"]) + "_nps_" + str(
                    config["num_passes_sampling"]) + "_minbw_" + str(config["min_bandwidth"]).replace(".", "_") + condition_add_name
            SegmentationSaver(
                output_dir=config["outputs_directory"],
                output_postfix=name_addition + "_kde",
                output_ext=".nii.gz",
                resample=False,
                scale=None,
                dtype=np.dtype(np.float32),
                batch_transform=lambda batch: batch["quantization_meta_dict"],
                output_transform=lambda output: output[CommonKeys.PRED]["kde_result"],
            ).attach(engine)
        else:
            if config["conditionings"]:
                condition_add_name = "_conditioned"
            else:
                condition_add_name = ""
            name_addition = "combined_th_" + str(config["threshold"]).replace(".", "_") + "_npd_" + \
                                str(config["num_passes_dropout"]) + "_nps_" + str(
                    config["num_passes_sampling"]) + condition_add_name
            # SegmentationSaver(
            #     output_dir=config["outputs_directory"],
            #     output_postfix=name_addition + "_result",
            #     output_ext=".nii.gz",
            #     resample=False,
            #     scale=None,
            #     dtype=np.dtype(np.float32),
            #     batch_transform=lambda batch: batch["quantization_meta_dict"],
            #     output_transform=lambda output: output[CommonKeys.PRED]["kde_result"],
            # ).attach(engine)
            NpySaver(
                output_dir=config["outputs_directory"],
                output_postfix=name_addition + "_recons",
                dtype=np.dtype(np.float32),
                batch_transform=lambda batch: batch["quantization_meta_dict"],
                output_transform=lambda output: output[CommonKeys.PRED]["reconstructions"],
            ).attach(engine)
            NpySaver(
                output_dir=config["outputs_directory"],
                output_postfix=name_addition + "_mean",
                dtype=np.dtype(np.float32),
                batch_transform=lambda batch: batch["quantization_meta_dict"],
                output_transform=lambda output: output[CommonKeys.PRED]["mean"],
            ).attach(engine)
            NpySaver(
                output_dir=config["outputs_directory"],
                output_postfix=name_addition + "_std",
                dtype=np.dtype(np.float32),
                batch_transform=lambda batch: batch["quantization_meta_dict"],
                output_transform=lambda output: output[CommonKeys.PRED]["std"],
            ).attach(engine)
            NpySaver(
                output_dir=config["outputs_directory"],
                output_postfix=name_addition + "_resampling_mask",
                dtype=np.dtype(np.float32),
                batch_transform=lambda batch: batch["quantization_meta_dict"],
                output_transform=lambda output: output[CommonKeys.PRED]["resampling_mask"],
            ).attach(engine)
            #SegmentationSaver(
            #    output_dir=config["outputs_directory"],
            #    output_postfix=name_addition + "_recons",
            #    output_ext=".nii.gz",
            #    resample=False,
            #    scale=None,
            #    dtype=np.dtype(np.float32),
                #batch_transform= lambda batch: {'filename_or_obj':  batch["quantization_meta_dict"]["filename_or_obj"]},
            #    batch_transform=get_batch_transform(
            #    no_augmented_extractions=0,
            #    is_nii_based=True,
            #    filename_or_objs_only=True,
            #    mode="extracting"),
            #    output_transform=lambda output: output[CommonKeys.PRED]["reconstructions"],
            #).attach(engine)
            #SegmentationSaver(
            #    output_dir=config["outputs_directory"],
            #    output_postfix=name_addition + "_mean",
            #    output_ext=".nii.gz",
            #    resample=False,
            #    scale=None,
            #    dtype=np.dtype(np.float32),
                #batch_transform= lambda batch: {'filename_or_obj':  batch["quantization_meta_dict"]["filename_or_obj"]},
            #    batch_transform=get_batch_transform(
            #    no_augmented_extractions=0,
            #    is_nii_based=True,
            #    filename_or_objs_only=True,
            #    mode="extracting"),
            #    output_transform=lambda output: output[CommonKeys.PRED]["mean"],
            #).attach(engine)
            #SegmentationSaver(
            #    output_dir=config["outputs_directory"],
            #    output_postfix=name_addition + "_std",
            #    output_ext=".nii.gz",
            #    resample=False,
            #    scale=None,
            #    dtype=np.dtype(np.float32),
                #batch_transform= lambda batch:{'filename_or_obj':  batch["quantization_meta_dict"]["filename_or_obj"]},
            #    batch_transform=get_batch_transform(
            #    no_augmented_extractions=0,
            #    is_nii_based=True,
            #    filename_or_objs_only=True,
            #    mode="extracting"),
            #    output_transform=lambda output: output[CommonKeys.PRED]["std"],
            #).attach(engine)
            #SegmentationSaver(
            #    output_dir=config["outputs_directory"],
            #    output_postfix=name_addition + "_resampling_mask",
            #    output_ext=".nii.gz",
            #    resample=False,
            #    scale=None,
            #    dtype=np.dtype(np.float32),
                #batch_transform= lambda batch: {'filename_or_obj':  batch["quantization_meta_dict"]["filename_or_obj"]},
            #    batch_transform=get_batch_transform(
            #    no_augmented_extractions=0,
            #    is_nii_based=True,
            #    filename_or_objs_only=True,
            #    mode="extracting"),
            #    output_transform=lambda output: output[CommonKeys.PRED]["resampling_mask"],
            #).attach(engine)
    
    
    
        ProgressBar().attach(engine, output_transform=lambda output: {"Loss": 0})
    
        engine.run()


def run(
        # File system parameters
        training_subjects: Union[
            str, Tuple[str, ...]
        ] = "/home/danieltudosiu/storage/datasets/neuro_morphology/healthy/nii_training/",
        validation_subjects: Union[
            str, Tuple[str, ...]
        ] = "/home/danieltudosiu/storage/datasets/neuro_morphology/healthy/nii_validation/",
        project_directory: str = "/home/danieltudosiu/storage/projects/nmcgm/",
        experiment_name: str = "nvidia",
        transformer_network_checkpoint: str = "/nfs/home/apatel/CT_PET_FDG/results/vqgan_suv_15_jp_do_005_wn_none_dropout_end_ne64_PET_6/enc_dec_performer/checkpoints/checkpoint_epoch=200.pt",
        vqvae_network_checkpoint: str = "/nfs/home/apatel/CT_PET_FDG/private_NSCLC_results/vqgan_suv_15_jp_do_005_dropout_end_ne64_PET/baseline_vqvae/checkpoints/checkpoint_epoch=1500.pt",
        infer_mode: str = "anomaly_detection",
        conditioning_path: str = None,
        conditionings: Tuple[str, ...] = None,
        encoding_conditioning_path: str = None,
        original_nii_path: str = None,
        token_masking_path: str = None,
        conditioning_type: str = TransformerConditioningType.NONE.value,
        use_continuous_conditioning: Union[bool, Tuple[bool, ...]] = False,
        outputs_directory: str = None,
        # Hardware and input parameters
        device: int = 0,
        deterministic: bool = False,
        cuda_benchmark: bool = True,
        seed: int = 2,
        starting_epoch: int = -1,

        # data processing parameters
        batch_size: int = 1,
        eval_batch_size: int = 1,
        num_workers: int = 8,
        prefetch_factor: int = 6,

        # transformer inputs
        network: str = "performer",
        vocab_size: int = 32,
        vocab_size_enc: int = 32,
        n_embd: int = 256,
        n_layers: int = 10,
        n_head: int = 8,
        local_attn_heads: int = 0,
        local_window_size: int = 256,
        feature_redraw_interval: int = 1000,
        generalized_attention: bool = False,
        emb_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        ordering_type: str = OrderingType.RASTER_SCAN.value,
        reflected_spatial_dims: Union[Tuple[bool, bool], Tuple[bool, bool, bool]] = (
            False,
            False,
            False,
        ),
        transpositions_axes: Union[
            Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int], ...]
        ] = tuple(),
        rot90_axes: Union[
            Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int], ...]
        ] = tuple(),
        transformation_order: Tuple[
            OrderingTransformations, OrderingTransformations, OrderingTransformations
        ] = (
            OrderingTransformations.TRANSPOSE.value,
            OrderingTransformations.ROTATE_90.value,
            OrderingTransformations.REFLECT.value,
        ),
        spatial_position_emb: str = None,


        # VQ-VAE inputs
        vqvae_network: str = "baseline_vqvae",
        use_subpixel_conv: bool = False,
        use_slim_residual: bool = True,
        no_levels: int = 3,
        no_res_layers: int = 3,
        no_channels: int = 256,
        codebook_type: str = "ema",
        num_embeddings: Tuple[int, ...] = (64,),
        embedding_dim: Tuple[int, ...] = (256,),
        embedding_init: Tuple[str, ...] = ("normal",),
        dropout_penultimate: bool = True,
        dropout_enc: float = 0.0,
        dropout_dec: float = 0.05,
        act: str = "LEAKYRELU",
        output_act: str = None,
        modality="PET",
        weight_norm: bool = False, 
        downsample_parameters: Tuple[Tuple[int, int, int, int], ...] = (
            (4, 2, 1, 1),
            (4, 2, 1, 1),
            (4, 2, 1, 1),
        ),
        upsample_parameters: Tuple[Tuple[int, int, int, int, int], ...] = (
            (4, 2, 1, 0, 1),
            (4, 2, 1, 0, 1),
            (4, 2, 1, 0, 1),
        ),
        commitment_cost: Tuple[float, ...] = (0.25,),
        decay: Tuple[float, ...] = (0.99,),
        decay_warmup: str = "none",
        max_decay_epochs: Union[str, int] = 50,
        norm: str = None,

        # inference parameters
        num_passes_dropout: int = 20,
        num_passes_sampling: int = 20,
        threshold: float = 0.005,
        epsilon_percentile: float = 0.97,
        use_prior_llmap: bool = False,
        llmap_path: str = None,
        stdev_smoothing: float = 1.0,
        use_resampling_mask: bool = False,
        model_num: int = 0,
        min_bandwidth: float = 0.0,

        evaluation_checkpoint: str = "recent",


        # Others for compatibility
        vqvae_checkpoint: str = None,
        vqvae_conditioning_checkpoint: str = None,
        use_vqvae_aug_conditionings: bool = False,
        vqvae_aug_load_nii_canonical: bool = False,
        vqvae_aug_augmentation_probability: float = 0.2,
        vqvae_aug_augmentation_strength: float = 0.0,
        vqvae_aug_normalize: bool = True,
        vqvae_aug_standardize: bool = False,
        vqvae_aug_roi: Union[
            Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]
        ] = None,

        vqvae_net_level: int = 0,
        vqvae_net_use_subpixel_conv: bool = False,
        vqvae_net_use_slim_residual: bool = True,
        vqvae_net_no_levels: int = 3,
        vqvae_net_downsample_parameters: Tuple[Tuple[int, int, int, int], ...] = (
            (4, 2, 1, 1),
            (4, 2, 1, 1),
            (4, 2, 1, 1),
        ),
        vqvae_net_upsample_parameters: Tuple[Tuple[int, int, int, int, int], ...] = (
            (4, 2, 1, 0, 1),
            (4, 2, 1, 0, 1),
            (4, 2, 1, 0, 1),
        ),
        vqvae_net_no_res_layers: int = 1,
        vqvae_net_no_channels: int = 128,
        vqvae_net_codebook_type: str = "ema",
        vqvae_net_num_embeddings: Tuple[int, ...] = (32,),
        vqvae_net_embedding_dim: Tuple[int, ...] = (64,),
        vqvae_net_embedding_init: Tuple[str, ...] = ("normal",),
        vqvae_net_commitment_cost: Tuple[float, ...] = (0.25,),
        vqvae_net_decay: Tuple[float, ...] = (0.99,),
        vqvae_net_norm: str = None,
        vqvae_net_dropout_penultimate: bool = False,
        vqvae_net_dropout_enc: float = 0.1,
        vqvae_net_dropout_dec: float = 0.1,
        vqvae_net_act: str = "RELU",
        vqvae_net_output_act: str = None,
        # VQVAE CONDITIONING Network Parameters
        vqvae_network_cond: str = "baseline_vqvae",
        vqvae_net_level_cond: int = 0,
        vqvae_net_use_subpixel_conv_cond: bool = False,
        vqvae_net_use_slim_residual_cond: bool = True,
        vqvae_net_no_levels_cond: int = 3,
        vqvae_net_downsample_parameters_cond: Tuple[Tuple[int, int, int, int], ...] = (
                (4, 2, 1, 1),
                (4, 2, 1, 1),
                (4, 2, 1, 1),
        ),
        vqvae_net_upsample_parameters_cond: Tuple[Tuple[int, int, int, int, int], ...] = (
                (4, 2, 1, 0, 1),
                (4, 2, 1, 0, 1),
                (4, 2, 1, 0, 1),
        ),
        vqvae_net_no_res_layers_cond: int = 1,
        vqvae_net_no_channels_cond: int = 128,
        vqvae_net_codebook_type_cond: str = "ema",
        vqvae_net_num_embeddings_cond: Tuple[int, ...] = (32,),
        vqvae_net_embedding_dim_cond: Tuple[int, ...] = (64,),
        vqvae_net_embedding_init_cond: Tuple[str, ...] = ("normal",),
        vqvae_net_commitment_cost_cond: Tuple[float, ...] = (0.25,),
        vqvae_net_decay_cond: Tuple[float, ...] = (0.99,),
        vqvae_net_norm_cond: str = None,
        vqvae_net_dropout_penultimate_cond: bool = False,
        vqvae_net_dropout_enc_cond: float = 0.1,
        vqvae_net_dropout_dec_cond: float = 0.1,
        vqvae_net_act_cond: str = "RELU",
        vqvae_net_output_act_cond: str = None,
        additional_samples_multiplier: int = 0,

        bucket_values: bool = False,
        spatial_bias_max_dist: int = 50,
        use_scalenorm: bool = False,
        use_rezero: bool = False,
        position_emb: str = "absolute",
        tie_embedding: bool = False,
        data_num_channels: int = 3


):
    """
    Entry point for the transformer handling. It follows this structure since it is the same one found in the
    Distributed Data Parallelism Ignite tutorial found at :

    https://github.com/pytorch/ignite/tree/master/examples/contrib/cifar10

    Args:
        training_subjects (Union[str, Tuple[str, ...]]): Path(s) towards either a folder with .nii.gz files or towards
            a csv/tsv which has a 'path' column that stores full paths towards .nii.gz or .npy files. The files must be
            .nii.gz for training and extracting mode, and for decoding they must be .npy. A tuple of paths must be
            passed when the selected model is a hierarchical VQVAE. Those will be used for training.

        validation_subjects (Union[str, Tuple[str, ...]]): Path(s) towards either a folder with .nii.gz files or towards
            a csv/tsv which has a 'path' column that stores full paths towards .nii.gz or .npy files. The files must be
            .nii.gz for training and extracting mode, and for decoding they must be .npy. A tuple of paths must be
            passed when the selected model is a hierarchical VQVAE. Those will be used for validation.

        encoding_conditioning_path (str): Path towards a csv/tsv file that has a 'subject' column in which the file names from
            both training and validation subjects are and has a second column 'encoding' that contains a path to .npy files
            that represent in the inputs to the encoding portion of the encoder/decoder performer. Only valid when using 
            and enc-dec architecture.

        project_directory (str): Path towards folder where the experiment folder will be created.

        experiment_name (str): Name of the experiment which will be used to name a folder in the project_directory.
            Defaults to 'nvidia'.

        transformer_network_checkpoint (str): Path to the checkpoint of the saved transformer model state to use

        transformer_network_checkpoint (str): Path to the checkpoint of the saved transformer model state to use

        infer_mode (str) : It can be one of the following: ['anomaly_detection', 'likelihood_map', 'zscore', 'kde', 'zscore_kde']
                    'anomaly_detection': Given the location of the .nii.gz images, the vqvae and transformer networks, an anomaly detection inference
                    routine is run by replacing low probability tokens from the encoded images using the transformer
                    'likelihood_map': Generate likelihood maps from sampling the tokens of the encoded image through the transformer network

        device (int): The index of the GPU in the PCI_BUS_ID order. Defaults to 0.

        deterministic (bool): Boolean that sets monai.utils.set_determinism. Defaults to True.

        cuda_benchmark (bool): Boolean that sets whether cuda_benchmark will be used. It is not exclusive with
            deterministic, but it supersedes it. Defaults to False.

        seed (int): The seed to be used for the experiment. Defaults to 2.

        starting_epoch (int): At which epoch we start the training. Defaults to -1 i.e. uses latest checkpoint.

        num_workers (int): The number of threads that will be used to load batches. Defaults to 8.

        prefetch_factor (int): How may batches each thread will try and buffer. Defaults to 6.

        vocab_size (int): The size of the vocabulary. It must be the same values as the "num_embeddings" argument used
            during the vqvae training. Defaults to 32.

        vocab_size_enc (int): The size of the vocabulary for encoder portion of network. It must be the same values as the "num_embeddings" argument used
            during the vqvae training. Defaults to 32.

        n_embd (int): The size of the latent representation that the transformer will use. Defaults to 256.

        n_layers (int): The number of layers the transformer will have. Defaults to 10.

        n_head (int): The number of heads that the self attention mechanism will use.

        emb_dropout (float): Drop probability for the Dropout layer just after the embedding layer.

        ff_dropout (float): Drop probability for the Dropout layer just after the linear layers.

        attn_dropout (float): Drop probability for the Dropout layer just after the attention mechanism.

        ordering_type (str): The ordering logic that will be applied to project from 2D/3D tensor to 1D tensor. It can
            be one of the following: {[e.value for e in OrderingType]}. Defaults to 'raster_scan'.

        reflected_spatial_dims (Union[Tuple[bool, bool], Tuple[bool, bool, bool]]): Weather or not to flip axes of the
            2D/3D tensor before being projected to a 1D tensor. Defaults to (False, False, False).

        transpositions_axes (Union[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int], ...]]): Around which axes to
            apply transpose. Defaults to ().

        spatial_position_emb (str): It can be either None for no spatial positioning or 'fixed' or 'absolute'.
            Defaults to None.

        num_passes_dropout (int): The number of forward passes of decoding for uncertainty estimation using dropout.
                                  Only relavant when infer_mode = 'anomaly_detection_dropout' or
                                  'anomaly_detection_full_uncertainty'. Defaults to 20.

        num_passes_sampling (int): The number of times of resampling during "healing" for uncertainty estimation. Only
                                  relavant when infer_mode = 'anomaly_detection_multisampling' or
                                  'anomaly_detection_full_uncertainty'. Defaults to 20.

        threshold (float): threshold used for minimum likilihood required in transformer sampling to warrant token
                            resampling.  If using prior likelihood map this threshold is the minimum ratio between the
                            likelihoods for resampling otherwise it is an absolute threshold. Defaults to 0.005.

        use_prior_llmap (bool): Whether to use a priors map for average likelihood of tokens to use the ratio of
                                likelihoods to the average map for resampling as opposed to an absolute value use to
                                decide whether to resample. Default to False.

        llmap_path (str): Path pointing to location of prior likelihood map used. Only required if use_prior_llmap is
                          True.

        stdev_smoothing (float): Smoothing applied to the standard deviation map for implementing zscore analysis.
                                 Only relevant when implementing zscore inference. Defaults to None i.e. no smoothing
                                applied.

        use_resampling_mask (bool) : Whether to use resmapling mask to mask residuals outside of regions of resampling
                                     from the transformer
        min_bandwidth (float) : minimum bandwidth when calculating KDE values for inference

    """
    config = locals()

    if config.get("encoding_conditioning_path", None):
        config["conditioned_encoding"] = True
    else:
        config["conditioned_encoding"] = False

    if config.get("token_masking_path", None):
        config["token_masking"] = True
    else:
        config["token_masking"] = False

    if config["device"] == "ddp":
        deepspeed.init_distributed(
            dist_backend="nccl",
            auto_mpi_discovery=True,
            verbose=False,
            init_method=None,
        )

        config["rank"] = int(os.environ["RANK"])
        config["local_rank"] = int(os.environ["LOCAL_RANK"])
        config["world_size"] = int(os.environ["WORLD_SIZE"])
    else:
        config["rank"] = 0
        config["local_rank"] = 0
        config["world_size"] = 1

    # Only included as this vqvae_checkpoint triggers augmentations in transformer
    config["vqvae_checkpoint"] = None

    if config["use_prior_llmap"] and config["llmap_path"] is None:
        raise ValueError(
            "Undeclared ll map path in inputs."
        )

    config["mode"] = "EXTRACTING"

    inference(config=config)


if __name__ == "__main__":
    Fire({"run": run})

