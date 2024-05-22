import numpy as np
import torch
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch import nn

from gutils.rotary import RotaryAttention, get_idx_h_w
from maps.uncond.dit import get_2d_sincos_pos_embed, modulate, TimestepEmbedder, FinalLayer


# Adapted from https://github.com/facebookresearch/DiT
class CondDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, use_rotary_attn, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        attn_class = RotaryAttention if use_rotary_attn else Attention
        self.attn = attn_class(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_c = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # isn't norm functional?
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.mlp_c = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
        self.use_rotary_attn = use_rotary_attn

    def forward(self, x, c, x_c, idx_h, idx_w):
        shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp, \
            shift_mlp_c, scale_mlp_c, gate_mlp_c = self.adaLN_modulation(c).chunk(9, dim=1)
        join = torch.cat((x, x_c), dim=1)
        attn_input = modulate(self.norm1(join), shift_msa, scale_msa)
        if self.use_rotary_attn:
            attn = self.attn(attn_input, idx_h, idx_w)
        else:
            attn = self.attn(attn_input)
        join = join + gate_msa.unsqueeze(1) * attn
        x, x_c = torch.split(join, (x.shape[1], x_c.shape[1]), dim=1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x_c = x_c + gate_mlp_c.unsqueeze(1) + self.mlp_c(modulate(self.norm2_c(x_c), shift_mlp_c, scale_mlp_c))
        return x, x_c


class CondDiTi(nn.Module):
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
        self.c_embedder = PatchEmbed(img_size=None, patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size,
                                     bias=True, strict_img_size=False)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            CondDiTBlock(hidden_size, num_heads, use_rotary_attn=use_rotary_attn, mlp_ratio=mlp_ratio) for _ in
            range(depth)
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

        w = self.c_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.c_embedder.proj.bias, 0)

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

    def forward(self, x, t, x_c, pos=(0, 0)):
        """
        x: (N, MH, MW, C, H, W)
        t: (N,)
        x_c: (N, C, H_out, W_out)
        """
        N, MH, MW, C, H, W = x.shape
        M = MH * MW

        _, _, H_c, W_c = x_c.shape
        assert H_c == W_c
        x_c = self.c_embedder(x_c)  # (N, T_c, D)
        if self.add_pos_emb:
            pos_embed_c = get_2d_sincos_pos_embed(self.hidden_size, H_c // self.patch_size)
            pos_embed_c = torch.from_numpy(pos_embed_c).float().to(x.device)  # (T_c, D)
            x_c = x_c + pos_embed_c[None, :, :]

        x = self.x_embedder(x.view((N * M, C, H, W)))  # (N * M, T, D)
        T = x.shape[1]
        D = x.shape[2]
        x = x.view((N, M, T, D))

        if self.add_pos_emb:
            embeds = []
            for i in range(MH):
                for j in range(MW):
                    embeds.append(get_2d_sincos_pos_embed(self.hidden_size, int(self.x_embedder.num_patches ** 0.5),
                                                          shift=(
                                                          i * self.base_size + pos[0], j * self.base_size + pos[1])))
            pos_embed = np.stack(embeds, axis=0)
            pos_embed = torch.from_numpy(pos_embed).float().to(x.device)
            x = x + pos_embed[None, :, :, :]  # (N, M, T, D)

        idx_h, idx_w = get_idx_h_w(MH, MW, H // self.patch_size, W // self.patch_size)
        idx_h_c, idx_w_c = get_idx_h_w(1, 1, H_c // self.patch_size, W_c // self.patch_size)
        # We first have original tokens, then conditioning tokens
        idx_h = torch.cat((idx_h, idx_h_c), dim=0).to(x.device)
        idx_w = torch.cat((idx_w, idx_w_c), dim=0).to(x.device)

        x = x.reshape((N, M * T, D))  # (N, M * T, D)
        t = self.t_embedder(t)  # (N, D)
        for block in self.blocks:
            x, x_c = block(x, t, x_c, idx_h, idx_w)
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
