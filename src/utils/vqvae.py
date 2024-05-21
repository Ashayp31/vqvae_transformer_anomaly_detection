import os
from copy import deepcopy
from logging import Logger
from math import floor
from typing import Tuple, Union, List, Dict, Callable

import pandas as pd
import torch
from ignite.engine import _prepare_batch
from ignite.utils import convert_tensor
from monai.data import Dataset, CacheDataset, DataLoader, DistributedSampler
from monai.data.utils import pad_list_data_collate
import torchvision.transforms as T

from monai.transforms import (
    Compose,
    AddChanneld,
    AsChannelFirstd,
    ScaleIntensityd,
    RandGaussianNoised,
    RandShiftIntensityd,
    RandAdjustContrastd,
    CenterSpatialCropd,
    SpatialCropd,
    SpatialPadd,
    RandAffined,
    ThresholdIntensityd,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    RandFlipd,
    RandRotate90d,
    ToTensord,
    RandGaussianSmoothd,
    Rand3DElasticd,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd,
    Resize,
    #SignalFillEmptyd,
)


import itertools
from monai.transforms.io.dictionary import LoadImaged
from monai.utils.enums import NumpyPadMode

from src.transforms.general.dictionary import TraceTransformsd
from src.utils.constants import VQVAEModes, AugmentationStrengthScalers

import random
from monai.transforms import MapTransform
from monai.config import KeysCollection
from typing import Optional
import numpy as np




class RandResizeImg(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        rand_val = random.uniform(0, 1)
        img = d["MRI"]
        
        if rand_val < 0.8:
            rand_resolution_change = 1+random.uniform(0,1)
            x_len = img.shape[1]
            y_len = img.shape[2]
            z_len = img.shape[3]
            new_x_len = int(16*((x_len/rand_resolution_change)//16))
            new_y_len = int(16*((y_len/rand_resolution_change)//16))
            new_z_len = int(16*((z_len/rand_resolution_change)//16))

            resize_transform = Resize(spatial_size=[new_x_len, new_y_len, new_z_len], mode="trilinear")
            img = resize_transform(img)
            np.nan_to_num(img, copy=False)
            d["MRI"] = img
        else:
            d["MRI"] = img
        return d
    

class RotateImages(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        img = d["MRI"]
        rand_val_x = random.uniform(0, 1)
        rand_val_y = random.uniform(0, 1)

        prev_shape = img.shape
        if rand_val_x > 0.5:
            kx = random.randint(1,3)
            img = torch.rot90(img, k=kx, dims=[2, 3])

        if rand_val_y > 0.5:
            ky = random.randint(1,3)
            img = torch.rot90(img, k=ky, dims=[1, 2])

        d["MRI"] = img
        return d
    
    
class AddCordConv(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        img = d["MRI"]
        crops = d["crop"]
        spatial_dims = (0,1,2)
        spatial_size = img.shape[1:]
        coord_channels = np.array(
            np.meshgrid(*tuple(np.linspace(crops[k*2], crops[k*2+1], spatial_size[k]) for k in range(len(spatial_size))),
                        indexing="ij"), dtype=np.float32)
        coord_channels = coord_channels[list(spatial_dims)]
        new_img = np.concatenate((img, coord_channels), axis=0)
        d["MRI"] = new_img
        return d
    


class CropWithPadding(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        use_crop_input: bool = False,
        allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.use_crop = use_crop_input

    def __call__(self, data):
        d = dict(data)


        if self.use_crop:
            image_data = d["MRI"]
            crop_vals = d["crop"]
            image_data[:,:crop_vals[0],:crop_vals[2],:crop_vals[4]] = 0
            image_data[:,crop_vals[1]:,crop_vals[3]:,crop_vals[5]:] = 0
        else:
            rand_val = random.uniform(0, 1)
            if rand_val > 0.6:
                image_data = d["MRI"]
                axial_len = image_data.shape[3]

                first_crop = random.randint(0, axial_len - 20)
                second_crop = random.randint(first_crop, axial_len)
                image_data[:,:,:,:first_crop] = 0
                image_data[:,:,:,second_crop:] = 0
                d["MRI"] = image_data

        return d


class CropWithoutPadding(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        use_crop_input: bool = False,
        allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.use_crop = use_crop_input

    def __call__(self, data):
        d = dict(data)

        if self.use_crop:
            image_data = d["MRI"]
            crop_vals = d["crop"]
            crop_vals_0 = int(crop_vals[0] * image_data.shape[1])
            crop_vals_1 = int(crop_vals[1] * image_data.shape[1])
            new_image_size_ax_1 = ((crop_vals_1 - crop_vals_0) // 8) * 8
            crop_vals_1 = crop_vals_0 + new_image_size_ax_1

            crop_vals_2 = int(crop_vals[2] * image_data.shape[2])
            crop_vals_3 = int(crop_vals[3] * image_data.shape[2])
            new_image_size_ax_2 = ((crop_vals_3 - crop_vals_2) // 8) * 8
            crop_vals_3 = crop_vals_2 + new_image_size_ax_2

            crop_vals_4 = int(crop_vals[4] * image_data.shape[3])
            crop_vals_5 = int(crop_vals[5] * image_data.shape[3])
            new_image_size_ax_3 = ((crop_vals_5 - crop_vals_4) // 8) * 8
            crop_vals_5 = crop_vals_4 + new_image_size_ax_3

            image_data = image_data[:,crop_vals_0:crop_vals_1,crop_vals_2:crop_vals_3,crop_vals_4:crop_vals_5]
            d["MRI"] = image_data

        else:
            rand_val_x = random.uniform(0, 1)
            rand_val_y = random.uniform(0, 1)
            rand_val_z = random.uniform(0, 1)
            image_data = d["MRI"]
            x_len = image_data.shape[1]
            y_len = image_data.shape[2]
            z_len = image_data.shape[3]

            if rand_val_x > 1.5:

                first_crop_x = random.randint(0, x_len - 96) if x_len > 96 else 0
                second_crop_x = random.randint(first_crop_x, x_len)
                image_size_x = second_crop_x - first_crop_x
                new_image_size_x = max(image_size_x, 96)
                second_crop_x = min(x_len,first_crop_x + new_image_size_x)
                new_image_size_x = min(192,((second_crop_x - first_crop_x)//16) * 16)
                second_crop_x = first_crop_x + new_image_size_x
                image_data = image_data[:,first_crop_x:second_crop_x,:,:]
            else:
                if x_len > 192:
                    first_crop_x = random.randint(0, x_len - 192)
                    second_crop_x = first_crop_x + 192
                    image_data = image_data[:,first_crop_x:second_crop_x,:,:]
                else:
                    first_crop_x = 0
                    second_crop_x = x_len

            if rand_val_y > 1.5:


                first_crop_y = random.randint(0, y_len - 96) if y_len > 96 else 0
                second_crop_y = random.randint(first_crop_y, y_len)
                image_size_y = second_crop_y - first_crop_y
                new_image_size_y = max(image_size_y, 96)
                second_crop_y = min(y_len,first_crop_y + new_image_size_y)
                new_image_size_y = min(192,((second_crop_y - first_crop_y)//16) * 16)
                second_crop_y = first_crop_y + new_image_size_y
                image_data = image_data[:,:,first_crop_y:second_crop_y,:]
            else:
                if y_len > 192:
                    first_crop_y = random.randint(0, y_len - 192)
                    second_crop_y = first_crop_y + 192
                    image_data = image_data[:,:,first_crop_y:second_crop_y,:]
                else:
                    first_crop_y = 0
                    second_crop_y = y_len

            if rand_val_z > 1.5:

                first_crop_z = random.randint(0, z_len - 96) if z_len > 96 else 0
                second_crop_z = random.randint(first_crop_z, z_len)
                image_size_z = second_crop_z - first_crop_z
                new_image_size_z = max(image_size_z, 96)
                second_crop_z = min(z_len,first_crop_z + new_image_size_z)
                new_image_size_z = min(192,((second_crop_z - first_crop_z)//16) * 16)
                second_crop_z = first_crop_z + new_image_size_z
                image_data = image_data[:,:,:,first_crop_z:second_crop_z]

            else:
                if z_len > 192:
                    first_crop_z = random.randint(0, z_len - 192)
                    second_crop_z = first_crop_z + 192
                    image_data = image_data[:,:,:,first_crop_z:second_crop_z]
                else:
                    first_crop_z = 0
                    second_crop_z = z_len


            d["MRI"] = image_data
            d["crop"] = [first_crop_x/x_len, second_crop_x/x_len,first_crop_y/y_len, second_crop_y/y_len,first_crop_z/z_len, second_crop_z/z_len]

        return d


def get_transformations(
    mode: str,
    load_nii_canonical: bool,
    augmentation_probability: float,
    augmentation_strength: float,
    no_augmented_extractions: int,
    num_embeddings: int,
    normalize: bool,
    standardize: bool,
    roi: Union[Tuple[int, ...], Tuple[Tuple[int, int], ...]],
    num_samples: int,
    patch_size: Tuple[int, ...],
    crop_path: str,
    crop_type: str,
    apply_coordConv: bool,
    apply_rotations: bool,
    input_has_coordConv: bool,
    key: str = "MRI",
):
    assert not (standardize and normalize), "standardize and normalize should not be turned on at the same time."
    
    augmentations = []
    is_augmented = mode == VQVAEModes.TRAINING.value or no_augmented_extractions != 0

    if mode == VQVAEModes.DECODING.value:
        keys = [f"quantization_{idx}" for idx in range(len(num_embeddings))]
        transform = [LoadImaged(keys=keys, reader="NumpyReader"), ToTensord(keys=keys)]
    else:
        transform = [
            LoadImaged(
                keys=[key],
                reader="NibabelReader",
                as_closest_canonical=load_nii_canonical,
            ),
            AddChanneld(keys=[key]) if not input_has_coordConv else AsChannelFirstd(keys=[key]),
        ]

        if normalize:
            transform += [ScaleIntensityd(keys=[key], minv=0.0, maxv=1.0)]

        if standardize:
            transform += [NormalizeIntensityd(keys=[key])]

        if roi:
            if type(roi[0]) is int:
                transform += [CenterSpatialCropd(keys=[key], roi_size=roi)]
            elif type(roi[0]) is tuple:
                transform += [
                    SpatialCropd(
                        keys=[key],
                        roi_start=[a[0] for a in roi],
                        roi_end=[a[1] for a in roi],
                    )
                ]
            else:
                raise ValueError(
                    f"roi should be either a Tuple with three ints like (0,1,2) or a Tuple with three Tuples that have "
                    f"two ints like ((0,1),(2,3),(4,5)). But received {roi}."
                )

            transform += [
                # This is here to guarantee no sample has lower spatial resolution than the ROI
                # YOU SHOULD NOT RELY ON THIS TO CATCH YOU SLACK, ALWAYS CHECK THE SPATIAL SIZES
                # OF YOU DATA PRIOR TO TRAINING ANY MODEL.
                SpatialPadd(
                    keys=[key],
                    spatial_size=roi
                    if type(roi[0]) is int
                    else [a[1] - a[0] for a in roi],
                    mode=NumpyPadMode.SYMMETRIC,
                )
            ]

        if patch_size:
            transform += [
                RandSpatialCropSamplesd(
                    keys=[key],
                    num_samples=num_samples,
                    roi_size=patch_size,
                    random_size=False,
                    random_center=True,
                )
            ]

        if is_augmented:
            if patch_size:
                # Patch based transformations
                augmentations += [
                    RandFlipd(
                        keys=[key], prob=augmentation_probability, spatial_axis=0
                    ),
                    RandFlipd(
                        keys=[key], prob=augmentation_probability, spatial_axis=1
                    ),
                    RandFlipd(
                        keys=[key], prob=augmentation_probability, spatial_axis=2
                    ),
                    RandRotate90d(
                        keys=[key], prob=augmentation_probability, spatial_axes=(0, 1)
                    ),
                    RandRotate90d(
                        keys=[key], prob=augmentation_probability, spatial_axes=(1, 2)
                    ),
                    RandRotate90d(
                        keys=[key], prob=augmentation_probability, spatial_axes=(0, 2)
                    ),
                ]
            else:
                augmentations += [Rand3DElasticd(
                    keys=[key],
                    prob=0.8,
                    sigma_range=[1.0, 2.0],
                    magnitude_range=[2.0, 5.0],
                    rotate_range=[0, 0, 0.0],
                    translate_range=[6, 6, 0],
                    scale_range=[0.05, 0.05, 0],
                    padding_mode="zeros"
                ), ]

                augmentations += [
                    RandAdjustContrastd(keys=[key], prob=0.3, gamma=(0.98, 1.02)),
                    RandShiftIntensityd(keys=[key], prob=0.5, offsets=(0.0, 0.025)),
                    RandGaussianNoised(keys=[key], prob=0.5, mean=0.0, std=0.01),
                    RandGaussianSmoothd(keys=[key], prob=0.3, sigma_x=(0.15, 0.5), sigma_y=(0.15, 0.5),
                                        sigma_z=(0.15, 0.5))]

            augmentations = Compose(augmentations)

            transform += [
                augmentations,
                TraceTransformsd(keys=[key], composed_transforms=[augmentations]),
            ]

        transform += [
            ThresholdIntensityd(keys=[key], threshold=1, above=False, cval=1.0),
            ThresholdIntensityd(keys=[key], threshold=0, above=True, cval=0),

        ]



        if mode == VQVAEModes.TRAINING.value:
            if crop_type == "with_padding":
                if crop_path:
                    transform += [CropWithPadding(keys=[key], use_crop_input=True)]
                else:
                    transform += [CropWithPadding(keys=[key])]
            elif crop_type == "without_padding":
                if crop_path:
                    transform += [CropWithoutPadding(keys=[key], use_crop_input=True)]
                else:
                    transform += [CropWithoutPadding(keys=[key])]
            
        if apply_coordConv and not input_has_coordConv:
            transform += [AddCordConv(keys=[key])]

        transform += [RandResizeImg(keys=[key])]


        if apply_coordConv:
            transform += [ToTensord(keys=[key, "crop"])]
        else: 
            transform += [ToTensord(keys=[key])]

        if apply_rotations:
           transform += [RotateImages(keys=[key])]

    return Compose(transform), augmentations


def get_subjects(
    paths: Union[str, Tuple[str, ...]], mode: str, crop_paths:str, no_augmented_extractions: int
) -> List[Dict[str, str]]:
    if isinstance(paths, str):
        paths = [paths]
    else:
        paths = list(paths)


    files_list = []
    for path in paths:
        if os.path.isdir(path):
            files_list.append([os.path.join(path, os.fsdecode(f)) for f in os.listdir(path)])
        elif os.path.isfile(path):
            if path.endswith(".csv"):
                files_list.append(
                    pd.read_csv(filepath_or_buffer=path, sep=",")["path"].to_list()
                )
            elif path.endswith(".tsv"):
                files_list.append(
                    pd.read_csv(filepath_or_buffer=path, sep="\t")["path"].to_list()
                )
        else:
            raise ValueError(
                "Path is neither a folder (to source all the files inside) or a csv/tsv with file paths inside."
            )

    files = [list(itertools.chain.from_iterable(files_list))]

    if crop_paths:
        if os.path.isfile(crop_paths):
            if crop_paths.endswith(".csv"):
                crop_paths_file = pd.read_csv(
                    filepath_or_buffer=crop_paths, sep=","
                )
            elif crop_paths.endswith(".tsv"):
                crop_paths_file = pd.read_csv(
                    filepath_or_buffer=crop_paths, sep="\t"
                )
        else:
            raise ValueError("Cropping Path is not a csv/tsv with file paths inside.")
        
    subjects = []
    if mode == VQVAEModes.DECODING.value:
        is_multi_level = len(files) > 1
        for file in zip(files) if is_multi_level else files[0]:
            subject = {}
            for idx, quantization in enumerate(file if is_multi_level else [file]):
                if quantization.endswith(".npy"):
                    subject[f"quantization_{idx}"] = quantization
                else:
                    raise ValueError(f"Path given is not a .npy file, but {file} ")

            if crop_paths:
                try:
                    crop_subject = [crop_paths_file.loc[crop_paths_file["subject"] == subject_name, "x0"].values[0],
                                        crop_paths_file.loc[crop_paths_file["subject"] == subject_name, "x1"].values[0],
                                        crop_paths_file.loc[crop_paths_file["subject"] == subject_name, "y0"].values[0],
                                        crop_paths_file.loc[crop_paths_file["subject"] == subject_name, "y1"].values[0],
                                        crop_paths_file.loc[crop_paths_file["subject"] == subject_name, "z0"].values[0],
                                        crop_paths_file.loc[crop_paths_file["subject"] == subject_name, "z1"].values[0]]
                    subject["crop"] = crop_subject

                except IndexError:
                    print("Cannot find Cropping Info for ", file)
                    mia_subjects += 1
                    valid_subject = False
                    break
            else:
                crop_subject = [0.0,1.0,0.0,1.0,0.0,1.0]
                subject["crop"] = crop_subject

            subjects.append(subject)
    else:
        for file in files[0]:
            subject_name = os.path.basename(file)
            if file.endswith(".nii.gz"):
                if crop_paths:
                    try:
                        crop_for_subject = [
                            crop_paths_file.loc[crop_paths_file["subject"] == subject_name, "x0"].values[0],
                            crop_paths_file.loc[crop_paths_file["subject"] == subject_name, "x1"].values[0],
                            crop_paths_file.loc[crop_paths_file["subject"] == subject_name, "y0"].values[0],
                            crop_paths_file.loc[crop_paths_file["subject"] == subject_name, "y1"].values[0],
                            crop_paths_file.loc[crop_paths_file["subject"] == subject_name, "z0"].values[0],
                            crop_paths_file.loc[crop_paths_file["subject"] == subject_name, "z1"].values[0]]
                    except IndexError:
                        print("Cannot find Cropping Info for ", file)
                        mia_subjects += 1
                        valid_subject = False
                        break
                else:
                    crop_for_subject = [0.0,1.0,0.0,1.0,0.0,1.0]

                if (
                    no_augmented_extractions != 0
                    and mode == VQVAEModes.EXTRACTING.value
                ):
                    for augmentation_id in range(no_augmented_extractions):
                        subjects.append(
                            {"MRI": file, "augmentation_id": int(augmentation_id), "crop": crop_for_subject}
                        )
                else:
                    subjects.append({"MRI": file, "crop": crop_for_subject})
            else:
                raise ValueError(f"Path given is not a .nii.gz file, but {file} ")

    return subjects


def get_data_flow(
    config: dict, logger: Logger = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs the data ingestion logic. There are different approaches for full-image and patch-based training due to
    gpu usage efficiency.

    The following fields are needed in config (in order of appearance):

        training_subjects (Union[str,Tuple[str, ...]]): Either one or multiple absolute paths to either a folder, csv or
            tsv. If it is a csv or tsv, it is expected that a "path" column is present and holds absolute paths to
            individual files. Those subjects will be used for the training dataset. For 'decoding' a Tuple of paths it
            is expected matching that the number of elements in num_embeddings each element should point to either
            folder or csv/tsv are pointing towards .npy files otherwise a single element is expected pointing towards
            .nii.gz files.

        mode (str): For which mode the data flow is being created. It should be one of the following: 'training',
            'extracting',  'decoding'.

        load_nii_canonical (bool): If true will reorder image array data when loading .nii images to be as closest to
         canonical.

        augmentation_probability (float): The probabilities of every augmentation.

        no_augmented_extractions (int):  The number of augmentations per sample for extracting codes. This is useful
            when the dataset is small and the transformer is overfitting. When it is 0 no augmentations are used during
            extraction.

        num_embeddings (Tuple[int,...]): How many atomic elements each quantization elements has. This is used to
            determine the number of quantizations to be loaded.

        normalize (bool): Whether or not the training and validation datasets are 0-1 normalised. Defaults to True.

        roi (Tuple[int,int,int]): The region of interest in the image that will be cropped out and forward modified. If
            None then no cropping will happen. Defaults to None.

        patch_size (Tuple[int,int,int]): How big the randomly cropped area will be for training data. If None no random
            crop will happen. Defaults to None.

        batch_size (int): The batch size that will be used to train the network. Defaults to 2.

        num_workers (int): How many threads will be spawn to load batches. Defaults to 8.

        device (Union[str,int]): The index of the GPU in the PCI_BUS_ID order or 'ddp' for Distributed Data Parallelism.

        prefetch_factor (int): How many batches each thread will try to keep as a buffer. Defaults to 6.

        validation_subjects (Union[str,Tuple[str, ...]]): Either one or multiple absolute paths to either a folder, csv
        or tsv. If it is a csv or tsv, it is expected that a "path" column is present and holds absolute paths to
        individual files. Those subjects will be used for the training dataset. For 'decoding' a Tuple of paths it is
        expected matching that the number of elements in num_embeddings each element should point to either folder or
        csv/tsv are pointing towards .npy files otherwise a single element is expected pointing towards .nii.gz files.

        eval_patch_size (Tuple[int,int,int]): How big the randomly cropped area will be for evaluation data.
        If None no random crop will happen. Defaults to None.

        eval_batch_size (int): The batch size that will be used to evaluate the network. Defaults to 1.

    Args:
        config (dict): Configuration dictionary that holds all the required parameters.

        logger (Logger): Logger that will be used to report DataLoaders parameters.

    Returns:
        DataLoader: Training DataLoader which has data augmentations

        DataLoader: Evaluation DataLoader for the validation data. No data augmentations.

        DataLoader: Evaluation DataLoader for the training data. No data augmentations.
    """

    training_subjects = get_subjects(
        paths=config["training_subjects"],
        mode=config["mode"],
        crop_paths=config["cropping_file"],
        no_augmented_extractions=0,
    )

    training_transform, _ = get_transformations(
        mode=config["mode"],
        load_nii_canonical=config["load_nii_canonical"],
        augmentation_probability=config["augmentation_probability"],
        augmentation_strength=config["augmentation_strength"],
        no_augmented_extractions=config.get("no_augmented_extractions", 0),
        num_embeddings=config["num_embeddings"],
        normalize=config.get("normalize", True),
        standardize=config.get("standardize",False),
        roi=config.get("roi", None),
        patch_size=config.get("patch_size", None),
        num_samples=config.get("num_samples", 1),
        crop_path=config.get("cropping_file", None),
        crop_type=config.get("cropping_type", None),
        apply_coordConv=config.get("apply_coordConv", False),
        apply_rotations=config.get("apply_rotations", False),
        input_has_coordConv=config.get("input_has_coordConv", False),

    )

    training_dataset = Dataset(data=training_subjects, transform=training_transform)

    training_loader = DataLoader(
        training_dataset,
        batch_size=config.get("batch_size", 2),
        num_workers=config.get("num_workers", 8),
        # This is false due to the DistributedSampling handling the shuffling
        shuffle=config["device"] != "ddp",
        drop_last=True,
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 6),
        # Forcefully setting it to false due to this pull request
        # not being in PyTorch 1.7.1
        # https://github.com/pytorch/pytorch/pull/48543
        persistent_workers=False,
        collate_fn=pad_list_data_collate,
        sampler=DistributedSampler(
            dataset=training_dataset, shuffle=True, even_divisible=True
        )
        if config["device"] == "ddp"
        else None,
    )

    evaluation_subjects = get_subjects(
        paths=config["validation_subjects"],
        mode=config["mode"],
        crop_paths=config["cropping_file"],
        no_augmented_extractions=config["no_augmented_extractions"],
    )


    evaluations_transform, _ = get_transformations(
        mode=config["mode"],
        load_nii_canonical=config["load_nii_canonical"],
        augmentation_probability=config["augmentation_probability"],
        augmentation_strength=config["augmentation_strength"],
        no_augmented_extractions=config.get("no_augmented_extractions", 0),
        num_embeddings=config["num_embeddings"],
        normalize=config.get("normalize", True),
        standardize=config.get("standardize", False),
        roi=config.get("roi", None),
        patch_size=config.get("eval_patch_size", None),
        num_samples=config.get("eval_num_samples", 1),
        crop_path=config.get("cropping_file", None),
        crop_type=config.get("cropping_type", None),
        apply_coordConv=config.get("apply_coordConv", False),
        apply_rotations = config.get("apply_rotations", False),
        input_has_coordConv=config.get("input_has_coordConv", False),
    )

    evaluation_dataset = Dataset(data=evaluation_subjects, transform=evaluations_transform)

    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=config.get("eval_batch_size", 1),
        num_workers=config.get("num_workers", 8),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 6),
        persistent_workers=False,
        collate_fn=pad_list_data_collate,
        sampler=DistributedSampler(
            dataset=evaluation_dataset, shuffle=False, even_divisible=False
        )
        if config["device"] == "ddp"
        else None,
    )


    training_evaluation_dataset = Dataset(data=training_subjects, transform=evaluations_transform)

    training_evaluation_loader = DataLoader(
        training_evaluation_dataset,
        batch_size=config.get("eval_batch_size", 1),
        num_workers=config.get("num_workers", 8),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 6),
        persistent_workers=False,
        collate_fn=pad_list_data_collate,
        sampler=DistributedSampler(
            dataset=evaluation_dataset, shuffle=False, even_divisible=False
        )
        if config["device"] == "ddp"
        else None,
    )

    if logger:
        logger.info("Dataflow setting:")
        logger.info("\tTraining:")
        if config.get("patch_size", None):
            logger.info(f"\t\tPatch Size: {config['patch_size']}")
        logger.info(f"\t\tLength: {len(training_loader)}")
        logger.info(f"\t\tBatch Size: {training_loader.batch_size}")
        logger.info(f"\t\tPin Memory: {training_loader.pin_memory}")
        logger.info(f"\t\tNumber of Workers: {training_loader.num_workers}")
        logger.info(f"\t\tPrefetch Factor: {training_loader.prefetch_factor}")
        logger.info("\tValidation:")
        if config.get("eval_patch_size", None):
            logger.info(f"\t\tPatch Size: {config['eval_patch_size']}")
        logger.info(f"\t\tLength: {len(evaluation_loader)}")
        logger.info(f"\t\tBatch Size: {evaluation_loader.batch_size}")
        logger.info(f"\t\tPin Memory: {evaluation_loader.pin_memory}")
        logger.info(f"\t\tNumber of Workers: {evaluation_loader.num_workers}")
        logger.info(f"\t\tPrefetch Factor: {evaluation_loader.prefetch_factor}")

    config["epoch_length"] = len(training_loader)

    if config["mode"] != VQVAEModes.DECODING.value:
        _, _, input_height, input_width, input_depth = next(iter(training_loader))[
            "MRI"
        ].shape
        config["input_shape"] = (input_height, input_width, input_depth)

    return training_loader, evaluation_loader, training_evaluation_loader


def get_ms_ssim_window(config: dict, logger: Logger = None) -> int:
    """
    Calculates the window size of the gaussian kernel for the MS-SSIM if the smallest dimension of the image is
    lower than 160 (requirement of the default parameters of MS-SSIM).

    It will first look for the 'eval_patch_size' since it has the last one applied, if not found it will look for 'roi'
    since all images are bing cropped or padded to that roi, and lastly it will look for 'input_shape'.

    Args:
        config (dict): Dictionary that holds the whole configuration
        logger (Logger): Logger for printing the logic

    Returns:
        int: Half of the maximum kernel size allowed or next odd int
    """
    if config["eval_patch_size"]:
        min_ps = min(config["eval_patch_size"])
    elif config["roi"]:
        if type(config["roi"][0]) is int:
            min_ps = min(config["roi"])
        elif type(config["roi"][0]) is tuple:
            min_ps = min([a[1] - a[0] for a in config["roi"]])
    else:
        min_ps = min(config["input_shape"])

    if min_ps > 160:
        win_size = 11
    else:
        win_size = floor(((min_ps / 2 ** 4) + 1) / 2)

        if win_size <= 1:
            raise ValueError(
                "Window size for MS-SSIM can't be calculated. Please increase patch_size's smallest dimension."
            )

        # Window size must be odd
        if win_size % 2 == 0:
            win_size += 1

    if logger:
        logger.info("MS-SSIM window calculation:")
        if config["eval_patch_size"]:
            logger.info(f"\tMinimum spatial dimension: {min_ps}")
        logger.info(f"\tWindow size {win_size}")

    return win_size


@torch.no_grad()
def prepare_batch(batch, device=None, non_blocking=False):
    """
    Prepare batch function that allows us to train an unsupervised mode by using the SupervisedTrainer
    """
    x_input = x_target = batch["MRI"]
    x_coords = batch["crop"]
    x_input = convert_tensor(x_input, device=device, non_blocking=non_blocking)
    x_target = convert_tensor(x_target, device=device, non_blocking=non_blocking)
    x_coords = convert_tensor(x_coords, device=device, non_blocking=non_blocking)
    if x_input.shape[1] > 1:
        x_target = x_target[:,0,:,:,:]
        x_target = torch.unsqueeze(x_target, 1)

    return [x_input, x_coords], x_target


@torch.no_grad()
def prepare_decoding_batch(
    batch, num_quantization_levels, device=None, non_blocking=False
):
    """
    Prepare batch function that allows us to train an unsupervised mode by using the SupervisedTrainer
    """
    x_input = x_target = [
        convert_tensor(
            batch[f"quantization_{i}"].long(), device=device, non_blocking=non_blocking
        )
        for i in range(num_quantization_levels)
    ]

    x_coords = batch["crop"]

    x_coords = convert_tensor(x_coords, device=device, non_blocking=non_blocking)
    return [x_input, x_coords], x_target


def get_batch_transform(
    mode: str,
    no_augmented_extractions: int,
    is_nii_based: bool,
    filename_or_objs_only: bool,
) -> Callable:
    """
    Batch transform generation, it handles the generation of the function for all modes. It also takes care of
    prepending the augmentation index to the filename.

    Args:
        mode (str): The running mode of the entry point. It can be either 'extracting' or 'decoding'.
        no_augmented_extractions (int): The number of augmentations per sample for extracting codes.
        is_nii_based (bool): Whether or not the batch data is based on nii input.
        filename_or_objs_only (bool): Whether or not we pass only the filename from the metadata.
    Returns:
        Batch transformations function
    """

    def batch_transform(batch: Dict) -> Dict:
        key = "quantization_0" if mode == VQVAEModes.DECODING.value else "MRI"

        if filename_or_objs_only:
            output_dict = {
                "filename_or_obj": deepcopy(
                    batch[f"{key}_meta_dict"]["filename_or_obj"]
                )
            }
        else:
            output_dict = deepcopy(batch[f"{key}_meta_dict"])

        if no_augmented_extractions > 0:
            file_extension = ".nii.gz" if is_nii_based else ".npy"
            output_dict["filename_or_obj"] = [
                f.replace(f"{file_extension}", f"_{i}{file_extension}")
                for f, i in zip(
                    output_dict["filename_or_obj"], batch["augmentation_id"]
                )
            ]

        return output_dict

    return batch_transform
