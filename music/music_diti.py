import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from maps.uncond.dit import modulate, TimestepEmbedder, get_2d_sincos_pos_embed_from_grid
from utils.rotary import RotaryAttention, get_idx_h_w


# Some of the code imported from https://github.com/facebookresearch/DiT
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, use_rotary_attn, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        attn_class = RotaryAttention if use_rotary_attn else Attention
        self.attn = attn_class(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.use_rotary_attn = use_rotary_attn

    def forward(self, x, c, idx_h, idx_w):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        if self.use_rotary_attn:
            attn = self.attn(attn_input, idx_h, idx_w)
        else:
            attn = self.attn(attn_input)
        x = x + gate_msa.unsqueeze(1) * attn
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PatchEmbed(nn.Module):
    def __init__(
            self,
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            bias,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, shift=(0, 0)):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(shift[0], shift[0] + grid_size[0], dtype=np.float32)
    grid_w = np.arange(shift[1], shift[1] + grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


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
            stride,
            add_pos_emb,
            use_rotary_attn
    ):
        super().__init__()
        self.base_size = base_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.stride = stride
        self.add_pos_emb = add_pos_emb

        self.x_embedder = PatchEmbed(base_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, use_rotary_attn=use_rotary_attn, mlp_ratio=mlp_ratio) for _ in range(depth)
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
        x: (N, M, T, patch_size[0] * patch_size[1] * C)
        imgs: (N, M, C, H, W)
        """
        c = self.out_channels
        p1 = self.x_embedder.patch_size[0]
        p2 = self.x_embedder.patch_size[1]
        h = self.base_size[0] // p1
        w = self.base_size[1] // p2
        x = x.reshape(shape=(x.shape[0], x.shape[1], h, w, p1, p2, c))
        x = torch.einsum('nmhwpqc->nmchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], x.shape[1], c, h * p1, w * p2))
        return imgs

    def forward(self, x, t, pos=(0, 0)):
        """
        x: (N, MH, MW, C, H, W)
        t: (N,)
        """
        N, MH, MW, C, H, W = x.shape
        M = MH * MW

        x = self.x_embedder(x.view((N * M, C, H, W)))  # (N * M, T, D)
        T = x.shape[1]
        D = x.shape[2]
        x = x.view((N, M, T, D))
        if self.add_pos_emb:
            embeds = []
            for i in range(MH):
                for j in range(MW):
                    p1 = self.x_embedder.patch_size[0]
                    p2 = self.x_embedder.patch_size[1]
                    h = self.base_size[0] // p1
                    w = self.base_size[1] // p2
                    embeds.append(get_2d_sincos_pos_embed(self.hidden_size, (h, w),
                                                          shift=(i * self.base_size[0] + pos[0],
                                                                 j * self.base_size[1] + pos[1])))
            pos_embed = np.stack(embeds, axis=0)
            pos_embed = torch.from_numpy(pos_embed).float().to(x.device)
            x = x + pos_embed[None, :, :, :]  # (N, M, T, D)

        idx_h, idx_w = get_idx_h_w(MH, MW, H // self.patch_size[0], W // self.patch_size[1])
        idx_h = idx_h.to(x.device)
        idx_w = idx_w.to(x.device)

        x = x.reshape((N, M * T, D))  # (N, M * T, D)
        t = self.t_embedder(t)  # (N, D)
        for block in self.blocks:
            x = block(x, t, idx_h, idx_w)
        x = self.final_layer(x, t)  # (N, M * T, patch_size ** 2 * out_channels)
        x = x.reshape((N, M, T, x.shape[-1]))  # (N, M, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, M, out_channels, H, W)
        H_out = self.stride[0] * (MH - 1) + self.base_size[0]
        W_out = self.stride[1] * (MW - 1) + self.base_size[1]
        res = torch.zeros((N, self.out_channels, H_out, W_out), device=x.device)
        w = torch.zeros((N, self.out_channels, H_out, W_out), device=x.device)
        patch_ptr = 0
        for i in range(MH):
            for j in range(MW):
                L1 = i * self.stride[0]
                R1 = i * self.stride[0] + self.base_size[0]
                L2 = j * self.stride[1]
                R2 = j * self.stride[1] + self.base_size[1]
                res[:, :, L1:R1, L2:R2] += x[:, patch_ptr, :, :, :]
                w[:, :, L1:R1, L2:R2] += 1.0
                patch_ptr += 1
        return res / w
