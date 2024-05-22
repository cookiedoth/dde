# Adapted from https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch/blob/1e396f5fc9cdedeed5e889b0b6fd53520f380839/classifier/model.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from denoising_diffusion_pytorch import Unet


class Classifier(nn.Module):
    def __init__(self, dim, dim_mults):
        super().__init__()
        self.unet = Unet(dim=dim, dim_mults=dim_mults)
        self.final_conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)

    def forward(self, x):
        t = torch.ones((x.shape[0],), device=x.device)
        x = self.unet(x, t)
        x = self.final_conv(x)
        x = torch.squeeze(x, 1)
        return x
