import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

from maps.uncond.dit import get_2d_sincos_pos_embed, TimestepEmbedder, DiTBlock, FinalLayer


# Some parts of this code are adapted from https://github.com/facebookresearch/DiT
class DiTi(nn.Module):
    def __init__(
            self,
            base_size,
            patch_size,
            in_channels,
            hidden_size,
            depth,
            num_heads,
            mlp_ratio,
            out_channels,
            stride
    ):
        super().__init__()
        self.base_size = base_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.stride = stride

        self.x_embedder = PatchEmbed(base_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, M, T, patch_size**2 * C)
        imgs: (N, M, C, H, W)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[2] ** 0.5)
        assert h * w == x.shape[2]

        x = x.reshape(shape=(x.shape[0], x.shape[1], h, w, p, p, c))
        x = torch.einsum('nmhwpqc->nmchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], x.shape[1], c, h * p, h * p))
        return imgs

    def forward(self, x, t):
        """
        x: (N, MH, MW, C, H, W)
        t: (N,)
        """
        N, MH, MW, C, H, W = x.shape
        M = MH * MW
        embeds = []
        for i in range(MH):
            for j in range(MW):
                embeds.append(get_2d_sincos_pos_embed(self.hidden_size, int(self.x_embedder.num_patches ** 0.5),
                                                      shift=(i * self.base_size, j * self.base_size)))
        pos_embed = np.stack(embeds, axis=0)
        pos_embed = torch.from_numpy(pos_embed).float().to(x.device)

        x = self.x_embedder(x.view((N * M, C, H, W)))  # (N * M, T, D)
        T = x.shape[1]
        D = x.shape[2]
        x = x.view((N, M, T, D))
        x = x + pos_embed[None, :, :, :]  # (N, M, T, D)
        x = x.reshape((N, M * T, D))  # (N, M * T, D)
        t = self.t_embedder(t)  # (N, D)
        for block in self.blocks:
            x = block(x, t)
        x = self.final_layer(x, t)  # (N, M * T, patch_size ** 2 * out_channels)
        x = x.reshape((N, M, T, x.shape[-1]))  # (N, M, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, M, out_channels, H, W)
        H_out = self.stride * (MH - 1) + self.base_size
        W_out = self.stride * (MW - 1) + self.base_size
        res = torch.zeros((N, self.out_channels, H_out, W_out), device=x.device)
        w = torch.zeros((N, self.out_channels, H_out, W_out), device=x.device)
        patch_ptr = 0
        for i in range(MH):
            for j in range(MW):
                L1 = i * self.stride
                R1 = i * self.stride + self.base_size
                L2 = j * self.stride
                R2 = j * self.stride + self.base_size
                res[:, :, L1:R1, L2:R2] += x[:, patch_ptr, :, :, :]
                w[:, :, L1:R1, L2:R2] += 1.0
                patch_ptr += 1
        return res / w
