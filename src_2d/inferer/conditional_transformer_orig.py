from typing import List, Any

import copy
import torch
from monai.inferers import Inferer
import numpy as np
from torch.nn import functional as F
import scipy as sp
from sklearn.neighbors import KernelDensity

from monai.transforms import (
GaussianSmooth
)

from src.networks.transformers.transformer import TransformerBase
from src.networks.vqvae.vqvae import VQVAEBase

class TransformerConditionalInferer(Inferer):
    def __init__(self, device: Any, threshold: float, embedding_shape: Any, vqvae_net:VQVAEBase, vqvae_checkpoint: Any, clip_encoding: bool,
                use_llmap: bool, llmap: str) -> None:
        Inferer.__init__(self)

        self.device = device
        self.threshold = threshold
        self.embedding_shape = embedding_shape
        self.vqvae_model = vqvae_net
        self.vqvae_checkpoint = vqvae_checkpoint
        self.clip_encoding = clip_encoding
        self.use_llmap = use_llmap
        self.llmap = llmap


    def __call__(self, inputs: torch.Tensor, network: TransformerBase, *args: Any, **kwargs: Any):
        """
        Inferer for the VQVAE models and Transformer that extract quantization indicies and sample their probabilities
        with the transformer model to replace low probability codes from the transformer.

        Args:
            inputs: model input data for inference.
            networks: trained VQVAE

            args: optional args to be passed to ``network``. It is ignored.
            kwargs: optional keyword args to be passed to ``network``. It is ignored.

        """
        if len(inputs) == 3:
            input_pet, input_ct, conditioning = inputs 
        else:
            input_pet, input_ct, conditioning, original_pet = inputs 
        print(self.device)
        if self.device == "ddp":
            revert_ordering = network.module.ordering.get_revert_sequence_ordering()
            index_sequence = network.module.ordering.get_sequence_ordering()
        else:
            revert_ordering = network.ordering.get_revert_sequence_ordering()
            index_sequence = network.ordering.get_sequence_ordering()

        resampled_latent, resample_mask = self.resample(network, input_pet, input_ct, conditioning, revert_ordering, index_sequence)

        #resample_mask = self.smooth_resampling_mask(resample_mask)

        del network
        self.vqvae_model.load_state_dict(self.vqvae_checkpoint)
        self.vqvae_model.to(self.device)
        recon = self.vqvae_model.decode_samples(embedding_indices=resampled_latent.to(self.device).long()).cpu().numpy()
        outputs = {"resampled": recon, "resampling_mask": resample_mask}

        return outputs

    @torch.no_grad()
    def resample(self, transformer_network, pet_in, ct_in, conditioning, rev_ordering, ind_sequencing):

        zs_in = pet_in[:, :-1]
        zs_out = pet_in[:, 1:]

        # Only for performer
        if self.clip_encoding and (ct_in is not None):
            ct_in = ct_in[:, :-1]

        logits = transformer_network(zs_in, ct_in, conditioning)
        probs = F.softmax(logits, dim=-1).cpu()
        selected_probs = torch.gather(probs, 2, zs_out.cpu().unsqueeze(2).long())
        selected_probs = selected_probs.squeeze(2)

        latent_shape = None
        if self.use_llmap:
            average_map = np.load(self.llmap)
            average_map = torch.from_numpy(average_map)
            average_map = average_map.unsqueeze(0)
            latent_shape = average_map.shape
            average_map = average_map.reshape(average_map.shape[0], -1)
            average_map = average_map[:, ind_sequencing]
            selected_probs = torch.div(selected_probs, average_map)

        mask = (selected_probs.float() < self.threshold).long().squeeze(1)

        sampled = zs_in.clone().to(self.device)
        number_resampled = 0
        for i in range(zs_in.shape[-1] - 1):
            if mask[:, i].max() == 0:
                continue
            else:
                number_resampled += 1
                logits = transformer_network(sampled[:, :i + 1], ct_in)[:, i, :]
                probs_ = F.softmax(logits, dim=1)
                #indexes = torch.multinomial(probs_[:, :-1], 1).squeeze(-1)
                indexes = torch.argmax(probs_[:,:-1]).unsqueeze(0)
                sampled[:, i + 1] = mask[:, i] * indexes.cpu() + (1 - mask[:, i]) * sampled[:, i + 1].cpu()

        if latent_shape is None:
            # need a workaround here,
            latent_map = np.load('/nfs/home/apatel/CT_PET_FDG/results/vqgan_suv_15_jp_do_005_wn_none_dropout_end_ne64_PET_6/enc_dec_performer/average_ll_map.npy')
            latent_map = torch.from_numpy(latent_map)
            latent_map = latent_map.unsqueeze(0)
            latent_shape = latent_map.shape
        upsampled_mask = copy.deepcopy(mask)
        upsampled_mask = upsampled_mask[:, rev_ordering]
        upsampled_mask = upsampled_mask.reshape(latent_shape)
        upsampled_mask = upsampled_mask.cpu().numpy()
        upsampled_mask = upsampled_mask.repeat(8, axis=-1).repeat(8, axis=-2).repeat(8, axis=-3)
        upsampled_mask = np.expand_dims(upsampled_mask, 1)

        logits = transformer_network(sampled[:, 1:], ct_in)[:, -1, :]
        probs_ = F.softmax(logits, dim=1)
        indexes = torch.argmax(probs_[:,:-1]).unsqueeze(0).unsqueeze(0)
        #indexes = torch.multinomial(probs_[:, :-1], 1)
        sampled = torch.cat((sampled.cpu(), indexes.cpu()), dim=1)

        sampled = sampled[:, 1:][:, rev_ordering]

        sampled = sampled.reshape(self.embedding_shape)
        sampled = sampled.unsqueeze(0)
        return sampled, upsampled_mask

    @torch.no_grad()
    def smooth_resampling_mask(self, resampling_mask):

        mask_smoothed = []
        for up in resampling_mask:
            y = sp.ndimage.filters.gaussian_filter(up[0].astype("float32"), sigma=6, mode='constant')
            if y.max() != 0:
                y = np.clip((y / (0.075 * y.max())), 0, 1)
            else:
                y = np.ones_like(y, dtype="float32")
            mask_smoothed.append(y)

        mask_smoothed = np.array(mask_smoothed)
        mask_smoothed = np.expand_dims(mask_smoothed, 1)
        #mask_smoothed[mask_smoothed>0.1]=1

        return mask_smoothed


class LikelihoodMapInferer(Inferer):
    def __init__(self, device: Any, embedding_shape: Any, clip_encoding: bool) -> None:
        Inferer.__init__(self)

        self.device = device
        self.embedding_shape = embedding_shape
        self.clip_encoding = clip_encoding

    def __call__(self, inputs: torch.Tensor, network: TransformerBase, *args: Any, **kwargs: Any):
        """
        Inferer for the VQVAE models and Transformer that extract quantization indicies and sample their probabilities
        to ouput the probability maps of the tokens upsampled to the same shape as the original input image.

        Args:
            inputs: model input data for inference.
            networks: trained VQVAE

            args: optional args to be passed to ``network``. It is ignored.
            kwargs: optional keyword args to be passed to ``network``. It is ignored.

        """

        input_pet, input_ct, conditioning, original_pet = inputs 
        revert_ordering = network.ordering.get_revert_sequence_ordering()

        likelihood_map = self.get_likelihood_maps(network, input_pet, input_ct, conditioning, revert_ordering)

        outputs = {"llmap": likelihood_map}
        return outputs

    @torch.no_grad()
    def get_likelihood_maps(self, network, pet_in, ct_in, conditioning, revert_ordering):

        zs_in = pet_in[:, :-1]
        zs_out = pet_in[:, 1:]

        if self.clip_encoding:
            ct_in = ct_in[:, :-1]

        logits = network(zs_in, ct_in, conditioning)
        probs = F.softmax(logits, dim=-1).cpu()
        selected_probs = torch.gather(probs, 2, zs_out.cpu().unsqueeze(2).long())

        selected_probs = selected_probs.squeeze(2)
        reordered_mask = copy.deepcopy(selected_probs)
        reordered_mask = reordered_mask[:, revert_ordering]
        reordered_mask = reordered_mask.reshape(self.embedding_shape)
        return reordered_mask



class VQVAETransformerUncertaintyCombinedInferer(TransformerConditionalInferer):
    def __init__(self, device: Any, threshold: float, embedding_shape: Any, vqvae_net:VQVAEBase, vqvae_checkpoint: Any,
                 num_passes_sampling: int, num_passes_dropout: int, use_llmap: bool,
                 llmap: str, smoothing: float, clip_encoding: bool) -> None:
        TransformerConditionalInferer.__init__(self, device=device, threshold=threshold, embedding_shape=embedding_shape, vqvae_net=vqvae_net,
                                         vqvae_checkpoint=vqvae_checkpoint, clip_encoding=clip_encoding, use_llmap=use_llmap,
                 llmap=llmap)


        self.device = device
        self.threshold = threshold
        self.embedding_shape = embedding_shape
        self.vqvae_model = vqvae_net
        self.vqvae_checkpoint = vqvae_checkpoint
        self.num_passes_sampling = num_passes_sampling
        self.num_passes_dropout = num_passes_dropout
        self.use_llmap = use_llmap
        self.llmap = llmap
        self.smoothing = smoothing
        self.clip_encoding = clip_encoding

    def __call__(self, inputs: torch.Tensor, network: VQVAEBase, *args: Any, **kwargs: Any):
        """
        Inferer for the VQVAE models and Transformer that extract quantization indicies and sample their probabilities
        to ouput the probability maps of the tokens upsampled to the same shape as the original input image.

        Args:
            inputs: model input data for inference.
            networks: trained VQVAE

            args: optional args to be passed to ``network``. It is ignored.
            kwargs: optional keyword args to be passed to ``network``. It is ignored.

        """
        input_pet, input_ct, conditioning, original_pet = inputs 

        revert_ordering = network.ordering.get_revert_sequence_ordering()
        index_sequence = network.ordering.get_sequence_ordering()

        for i in range(self.num_passes_sampling):
            if i == 0:
                resampled_latents, resample_mask = self.resample(network, input_pet, input_ct, revert_ordering, index_sequence)
            else:
                resampled_latent, _ = self.resample(network, input_pet, input_ct, revert_ordering, index_sequence)
                resampled_latents = torch.cat((resampled_latents, resampled_latent), 0)

        del network
        self.vqvae_model.load_state_dict(self.vqvae_checkpoint)
        self.vqvae_model.to(self.device)

        for k in range(self.num_passes_sampling):
            resampled_latent_sample = resampled_latents[k]
            resampled_latent_sample = resampled_latent_sample.unsqueeze(0)
            for j in range(self.num_passes_dropout):
                if j==0 and k==0:
                    recons = self.vqvae_model.decode_samples(embedding_indices=resampled_latent_sample.to(self.device).long()).cpu()
                else:
                    recon = self.vqvae_model.decode_samples(embedding_indices=resampled_latent_sample.to(self.device).long()).cpu()
                    recons = torch.cat((recons, recon), 1)

        #at this point reconds and original_pet have the same dimension i.e. [1,1, 208, 168, 216] or [1,num_passes_sampling*num_passes_dropout, 208,168,216]
        mean_rec = torch.mean(recons, 1, True)
        stdev_rec = torch.std(recons, dim=1, keepdim=True)

        if self.smoothing is not None:
            smooth = GaussianSmooth(self.smoothing)
            stdev_rec = smooth(stdev_rec[0])
            stdev_rec = torch.from_numpy(stdev_rec).unsqueeze(0)

        resample_mask = self.smooth_resampling_mask(resample_mask)
        
        #outputs = {"std": stdev_rec, "mean": mean_rec, "resampling_mask": resample_mask}
        outputs = {"std": stdev_rec, "mean": mean_rec}

        return outputs


class VQVAETransformerKDEInferer(TransformerConditionalInferer):
    def __init__(self, device: Any, threshold: float, embedding_shape: Any, vqvae_net:VQVAEBase, vqvae_checkpoint: Any,
                 num_passes_sampling: int, num_passes_dropout: int, use_llmap: bool,
                 llmap: str, smoothing: float, clip_encoding: bool, min_bandwidth: float) -> None:
        TransformerConditionalInferer.__init__(self, device=device, threshold=threshold, embedding_shape=embedding_shape, vqvae_net=vqvae_net,
                                         vqvae_checkpoint=vqvae_checkpoint, clip_encoding=clip_encoding, use_llmap=use_llmap,
                 llmap=llmap)


        self.device = device
        self.threshold = threshold
        self.embedding_shape = embedding_shape
        self.vqvae_model = vqvae_net
        self.vqvae_checkpoint = vqvae_checkpoint
        self.num_passes_sampling = num_passes_sampling
        self.num_passes_dropout = num_passes_dropout
        self.use_llmap = use_llmap
        self.llmap = llmap
        self.smoothing = smoothing
        self.clip_encoding = clip_encoding
        self.min_bandwidth = min_bandwidth

    def __call__(self, inputs: torch.Tensor, network: VQVAEBase, *args: Any, **kwargs: Any):
        """
        Inferer for the VQVAE models and Transformer that extract quantization indicies and sample their probabilities
        to ouput the probability maps of the tokens upsampled to the same shape as the original input image.

        Args:
            inputs: model input data for inference.
            networks: trained VQVAE

            args: optional args to be passed to ``network``. It is ignored.
            kwargs: optional keyword args to be passed to ``network``. It is ignored.

        """
        input_pet, input_ct, conditioning, original_pet = inputs 

        revert_ordering = network.ordering.get_revert_sequence_ordering()
        index_sequence = network.ordering.get_sequence_ordering()

        for i in range(self.num_passes_sampling):
            if i == 0:
                resampled_latents, resample_mask = self.resample(network, input_pet, input_ct, revert_ordering, index_sequence)
            else:
                resampled_latent, _ = self.resample(network, input_pet, input_ct, revert_ordering, index_sequence)
                resampled_latents = torch.cat((resampled_latents, resampled_latent), 0)

        del network
        self.vqvae_model.load_state_dict(self.vqvae_checkpoint)
        self.vqvae_model.to(self.device)

        for k in range(self.num_passes_sampling):
            resampled_latent_sample = resampled_latents[k]
            resampled_latent_sample = resampled_latent_sample.unsqueeze(0)
            for j in range(self.num_passes_dropout):
                if j==0 and k==0:
                    recons = self.vqvae_model.decode_samples(embedding_indices=resampled_latent_sample.to(self.device).long()).cpu()
                else:
                    recon = self.vqvae_model.decode_samples(embedding_indices=resampled_latent_sample.to(self.device).long()).cpu()
                    recons = torch.cat((recons, recon), 1)

        #at this point reconds and original_pet have the same dimension i.e. [1,1, 208, 168, 216] or [1,num_passes_sampling*num_passes_dropout, 208,168,216]
        kde_result = (2*torch.ones_like(original_pet)).numpy()

        resample_mask = self.smooth_resampling_mask(resample_mask)
        kde_resample_mask = copy.deepcopy(resample_mask)
        kde_resample_mask[kde_resample_mask>0.05] = 1
        kde_resample_mask[kde_resample_mask<=0.05] = 0
        
        for i in range(recons.shape[2]):
            for j in range(recons.shape[3]):
                for k in range(recons.shape[4]):
                    if kde_resample_mask[0,0,i,j,k] == 0:
                        continue
                    else:
                        voxel_vals = recons[0,:,i,j,k].numpy()
                        voxel_vals[voxel_vals > 1] = 1
                        bandwidth = 1.06 * np.std(voxel_vals) * (np.power(self.num_passes_sampling*self.num_passes_dropout, -0.2))
                        bandwidth = max(bandwidth, self.min_bandwidth)
                        a=np.array([original_pet[0,0,i,j,k]]).reshape(-1,1)
                        voxel_vals = voxel_vals.reshape(-1,1)

                        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(voxel_vals)
                        kde_result[0,0,i,j,k] = np.exp(kde.score_samples(np.array([original_pet[0,0,i,j,k]]).reshape(-1,1)))   # need to get the right setting for this

        outputs = {"kde_result": kde_result}

        return outputs
