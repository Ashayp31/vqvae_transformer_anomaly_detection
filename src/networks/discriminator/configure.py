import torch.nn as nn

from src.networks.discriminator.utils import DiscriminatorNetworks
from src.networks.discriminator.taming import TamingDiscriminator
from src.networks.discriminator.baseline import BaselineDiscriminator


def get_discriminator_network(config: dict) -> nn.Module:
    if config["discriminator_network"] == DiscriminatorNetworks.BASELINE_DISCRIMINATOR.value:
        network = BaselineDiscriminator(
            input_nc=1,
            ndf=64,
            n_layers=3
        )
    elif config["discriminator_network"] == DiscriminatorNetworks.TAMING_DISCRIMINATOR.value:
        network = TamingDiscriminator(
            dimensions=3,
            in_channels=1,
            no_channels=64,
            no_layers=3,
            act="LEAKYRELU",
            dropout=0.0,
            norm="BATCH",
        )
    else:
        raise ValueError(
            f"Discriminator unknown. Was given {config['discriminator_network']} but choices are"
            f" {[discriminator.value for discriminator in DiscriminatorNetworks]}."
        )

    return network


# If needs be one can implement an
#   def add_vqvae_network_handlers(
#        train_handlers: List, discriminator: nn.Module, config: dict, logger: Logger
#   ) -> List:
# to add discriminator specific handlers
