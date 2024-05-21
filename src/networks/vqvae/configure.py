from enum import Enum
from logging import Logger
from typing import List

import numpy as np

from src.handlers.general import ParamSchedulerHandler
from src.networks.vqvae.baseline import BaselineVQVAE
from src.networks.vqvae.single_vqvae import SingleVQVAE
from src.networks.vqvae.vqvae import VQVAEBase
from src.utils.general import get_max_decay_epochs
from src.utils.constants import DecayWarmups


class VQVAENetworks(Enum):
    SINGLE_VQVAE = "single_vqvae"
    BASELINE_VQVAE = "baseline_vqvae"
    SLIM_VQVAE = "slim_vqvae"


def get_vqvae_network(config: dict) -> VQVAEBase:
    if config["network"] == VQVAENetworks.SINGLE_VQVAE.value:
        network = SingleVQVAE(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            use_subpixel_conv=config["use_subpixel_conv"],
            use_slim_residual=config["use_slim_residual"],
            no_levels=config["no_levels"],
            downsample_parameters=config["downsample_parameters"],
            upsample_parameters=config["upsample_parameters"],
            no_res_layers=config["no_res_layers"],
            no_channels=config["no_channels"],
            codebook_type=config["codebook_type"],
            num_embeddings=config["num_embeddings"],
            embedding_dim=config["embedding_dim"],
            embedding_init=config["embedding_init"],
            commitment_cost=config["commitment_cost"],
            decay=config["decay"],
            norm=config["norm"],
            dropout_at_end=config["dropout_penultimate"],
            dropout_enc=config["dropout_enc"],
            dropout_dec=config["dropout_dec"],
            act=config["act"],
            output_act=config["output_act"],
        )
    elif config["network"] == VQVAENetworks.BASELINE_VQVAE.value:
        network = BaselineVQVAE(
            n_levels=config["no_levels"],
            downsample_parameters=config["downsample_parameters"],
            upsample_parameters=config["upsample_parameters"],
            n_embed=config["num_embeddings"][0],
            embed_dim=config["embedding_dim"][0],
            commitment_cost=config["commitment_cost"][0],
            n_channels=config["no_channels"],
            n_res_channels=config["no_channels"],
            n_res_layers=config["no_res_layers"],
            dropout_at_end=config["dropout_penultimate"],
            dropout_enc=config["dropout_enc"],
            dropout_dec=config["dropout_dec"],
            output_act=config["output_act"],
            vq_decay=config["decay"][0],
            use_subpixel_conv=config["use_subpixel_conv"],
            apply_coordConv=config["apply_coordConv"],
        )
    elif config["network"] == VQVAENetworks.SLIM_VQVAE.value:
        raise NotImplementedError(
            f"{VQVAENetworks.SLIM_VQVAE}'s parsing is not implemented yet."
        )
    else:
        raise ValueError(
            f"VQVAE unknown. Was given {config['network']} but choices are {[vqvae.value for vqvae in VQVAENetworks]}."
        )

    return network


def add_vqvae_network_handlers(
    train_handlers: List, vqvae: VQVAEBase, config: dict, logger: Logger
) -> List:

    if config["decay_warmup"] == DecayWarmups.STEP.value:
        delta_step = (0.99 - config["decay"][0]) / 4
        stair_steps = np.linspace(0, config["max_decay_epochs"], 5)[1:]

        def decay_anealing(current_step: int) -> float:
            if (current_step + 1) >= 80:
                return 0.99
            if (current_step + 1) >= 40:
                return 0.95
            if (current_step + 1) >= 20:
                return 0.80
            if (current_step + 1) >= 10:
                return 0.70
            return 0.5

        #def decay_anealing(current_step: int) -> float:
        #    if (current_step + 1) >= stair_steps[3]:
        #        return config["decay"][0] + 4 * delta_step
        #    if (current_step + 1) >= stair_steps[2]:
        #        return config["decay"][0] + 3 * delta_step
        #    if (current_step + 1) >= stair_steps[1]:
        #        return config["decay"][0] + 2 * delta_step
        #    if (current_step + 1) >= stair_steps[0]:
        #        return config["decay"][0] + delta_step
        #    return config["decay"][0]

        train_handlers += [
            ParamSchedulerHandler(
                parameter_setter=vqvae.set_ema_decay,
                value_calculator=decay_anealing,
                vc_kwargs={},
                epoch_level=True,
            )
        ]
    elif config["decay_warmup"] == DecayWarmups.LINEAR.value:
        train_handlers += [
            ParamSchedulerHandler(
                parameter_setter=vqvae.set_ema_decay,
                value_calculator="linear",
                vc_kwargs={
                    "initial_value": config["decay"],
                    "step_constant": 0,
                    "step_max_value": config["max_decay_epochs"]
                    if isinstance(config["max_decay_epochs"], int)
                    else get_max_decay_epochs(config=config, logger=logger),
                    "max_value": 0.99,
                },
                epoch_level=True,
            )
        ]

    return train_handlers

