from typing import Any, Optional

import torch

from monai.inferers import Inferer

from src.networks.transformers.transformer import TransformerBase


class TransformerTrainingInferer(Inferer):
    def __init__(self) -> None:
        Inferer.__init__(self)

    def __call__(
        self, inputs: torch.Tensor, network: TransformerBase, *args: Any, **kwargs: Any
    ):
        """
        Training Inferer for the transformers models.

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``\
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """
        if len(inputs) == 2:
            sequence, conditioning = inputs
            return network(sequence, conditioning, *args, **kwargs).transpose(1, 2)
        elif len(inputs) == 3:
            sequence_out, sequence_in, conditioning = inputs
            return network(sequence_out, sequence_in, conditioning, *args, **kwargs).transpose(1, 2)
        elif len(inputs) == 4:
            sequence_out, sequence_in, conditioning, token_mask = inputs
            return network(sequence_out, sequence_in, conditioning, token_mask, *args, **kwargs).transpose(1, 2)
        else:
            input_pet, input_ct, conditioning, token_mask, original_pet = inputs 
            return network(sequence_out, sequence_in, conditioning, token_mask, *args, **kwargs).transpose(1, 2)


class TransformerInferenceInferer(Inferer):
    def __init__(
        self, temperature: float = 1.0, sample: bool = True, top_k: Optional[int] = None
    ) -> None:
        Inferer.__init__(self)
        self.temperature = temperature
        self.sample = sample
        self.top_k = top_k

    def __call__(
        self, inputs: torch.Tensor, network: TransformerBase, *args: Any, **kwargs: Any
    ):
        """
        Inference Inferer for the transformers models where we sample from the latent space.

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network.sample``.
            kwargs: optional keyword args to be passed to ``network.sample``.

        """
        sequence, conditioning = inputs

        n = (
            network.module
            if isinstance(network, torch.nn.parallel.DistributedDataParallel)
            else network
        )

        return n.sample(
            prefix=sequence,
            conditioning=conditioning,
            temperature=self.temperature,
            sample=self.sample,
            top_k=self.top_k,
            *args,
            **kwargs
        )
