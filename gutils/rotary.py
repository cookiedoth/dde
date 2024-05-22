from timm.layers.pos_embed_sincos import rot
import torch
from torch import nn
from torch.jit import Final
from timm.layers import use_fused_attn
import torch.nn.functional as F
from ml_logger import logger


def get_theta(D, device):
    i = torch.arange(D // 2, device=device)
    pw = -2 * i / D
    return 10000 ** pw


def rotate(M, idx):
    # M.shape = (B, num_heads, T, D)
    D = M.shape[3]
    assert D % 2 == 0
    theta = get_theta(D, M.device)
    angle = torch.outer(idx, theta)[None, None, :, :]
    M1, M2 = torch.chunk(M, 2, dim=3)
    return torch.cat((torch.cos(angle) * M1 - torch.sin(angle) * M2,
                      torch.sin(angle) * M1 + torch.cos(angle) * M2), dim=3)


def rotate_2d(M, idx_h, idx_w):
    M1, M2 = torch.chunk(M, 2, dim=3)
    M1 = rotate(M1, idx_h)
    M2 = rotate(M2, idx_w)
    return torch.cat((M1, M2), dim=3)


class RotaryAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, idx_h, idx_w) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = rotate_2d(q, idx_h, idx_w)
        k = rotate_2d(k, idx_h, idx_w)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_idx_h_w(MH, MW, H, W):
    idx_h = []
    idx_w = []
    for i in range(MH):
        for j in range(MW):
            h_list = torch.arange(i * H, (i + 1) * H)
            w_list = torch.arange(j * W, (j + 1) * W)
            ch, cw = torch.meshgrid(h_list, w_list)
            idx_h.append(ch.flatten())
            idx_w.append(cw.flatten())
    return torch.cat(idx_h, dim=0), torch.cat(idx_w, dim=0)
