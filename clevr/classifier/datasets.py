# Adapted from https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch/blob/3b9492b0b8fc690f3ecc63445087aed1a7c68cf2/classifier/datasets.py


import csv
import os
import random
import math
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import namedtuple

from typing import Optional
from functools import partial
from collections import namedtuple
from torchvision.datasets.utils import verify_str_arg
import torchvision.transforms as transforms

CSV = namedtuple("CSV", ["header", "index", "data"])


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class Clevr2DPosDataset(Dataset):
    def __init__(
        self,
        data_path,
        resolution,
        random_crop=False,
        random_flip=False,
        split=None
    ):
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip

        data = np.load(data_path)

        # split == None --> use the whole dataset
        self.ims, self.labels = data['ims'], data['coords_labels']

        N = self.ims.shape[0]
        if split == 'train':
            self.ims = self.ims[:int(N * 0.8)]
            self.labels = self.labels[:int(N * 0.8)]
        elif split == 'val':
            self.ims = self.ims[int(N * 0.8):]
            self.labels = self.labels[int(N * 0.8):]
        else:
            raise ValueError('Split needs to be specified.')

    def __len__(self):
        return self.ims.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.ims[index]).convert('RGB')
        pos = self.labels[index]
        label = 1

        if random.uniform(0, 1) < 0.5:
            if random.uniform(0, 1) < 0.5:
                # sample negative relation
                x = np.random.uniform(0, 1)
                y = np.random.uniform(0, 1)
                pos = np.array([x, y])
                label = 0
            else:
                # sample negative image
                neg_idx = random.randint(0, len(self.ims) - 1)
                while neg_idx == index or np.abs(np.sum(self.labels[neg_idx] - pos)) < 1e-5:
                    neg_idx = random.randint(0, len(self.ims) - 1)
                image = Image.fromarray(self.ims[neg_idx]).convert('RGB')
                label = 0

        if self.random_crop:
            arr = random_crop_arr(image, self.resolution)
        else:
            arr = center_crop_arr(image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        # range 0 to 1
        arr = arr.astype(np.float32) / 255.
        return np.transpose(arr, [2, 0, 1]), pos, label


if __name__ == '__main__':
    dataset = Clevr2DPosDataset(resolution=128, split='train')
