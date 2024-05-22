import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from enum import Enum


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        res = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return res


def create_mlp(in_features, hidden_counts, out_features):
    layers = []
    cur_dim = in_features
    for layer_dim in hidden_counts:
        layers.append(
            nn.Sequential(nn.Linear(in_features=cur_dim, out_features=layer_dim), nn.ReLU()))
        cur_dim = layer_dim
    layers.append(nn.Linear(in_features=cur_dim, out_features=out_features))
    return nn.Sequential(*layers)


class MLPScoreNet(nn.Module):
    def __init__(self, dim, hidden_counts=[100, 100], embed_dim=16):
        super().__init__()
        assert (embed_dim % 2 == 0)
        self.embed = GaussianFourierProjection(embed_dim=embed_dim)
        self.mlp = create_mlp(in_features=dim + embed_dim, hidden_counts=hidden_counts, out_features=dim)

    def forward(self, x, t):
        embed = self.embed(t)
        h = torch.cat([x, embed], dim=1)
        h = self.mlp(h)
        return h


class InferenceOutput(Enum):
    EPS = 0
    SCORE = 1
    SCORE_ENERGY = 2


class UNETEnergyNet(nn.Module):
    def __init__(self, x_dim, score_scale, h_dim=512, widen=2, n_layers=4, emb_dim=512, no_energy=False):
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.embed = GaussianFourierProjection(embed_dim=emb_dim)
        self.widen = widen
        self.score_scale = score_scale
        self.no_energy = no_energy
        self.initial_layer = nn.Linear(self.x_dim, self.h_dim)
        self.layers_h = nn.ModuleList([nn.Linear(self.h_dim, self.h_dim * self.widen) for _ in range(self.n_layers)])
        self.layers_emb = nn.ModuleList(
            [nn.Linear(self.emb_dim, self.h_dim * self.widen) for _ in range(self.n_layers)])
        self.layers_int = nn.ModuleList(
            [nn.Linear(self.h_dim * self.widen, self.h_dim * self.widen) for _ in range(self.n_layers)])
        self.layers_out = nn.ModuleList([nn.Linear(self.h_dim * self.widen, self.h_dim) for _ in range(self.n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.h_dim) for _ in range(self.n_layers)])
        self.final_layer = nn.Linear(self.h_dim, 2)

    def forward(self, x, t, mode, eval_mode=False):
        x.requires_grad_(True)

        emb = self.embed(t)

        cur = self.initial_layer(x)

        for i in range(self.n_layers):
            h = self.norms[i](cur)
            h = F.silu(h)
            h = self.layers_h[i](h)
            h += self.layers_emb[i](emb)
            h = F.silu(h)
            h = self.layers_int[i](h)
            h = F.silu(h)
            h = self.layers_out[i](h)
            cur = cur + h
        out = self.final_layer(cur)

        if self.no_energy:
            if mode == InferenceOutput.EPS:
                return out
            elif mode == InferenceOutput.SCORE:
                return out * self.score_scale(t)[:, None]
            else:
                raise ValueError('Incorrect forward pass mode')

        energy_score = x - out
        energy_norm = 0.5 * (energy_score ** 2)
        grad = torch.autograd.grad([energy_norm.sum()], [x], create_graph=not eval_mode)[0]
        energy = torch.sum(energy_norm, dim=1)
        if mode == InferenceOutput.EPS:
            return grad
        elif mode == InferenceOutput.SCORE:
            return grad * self.score_scale(t)[:, None]
        elif mode == InferenceOutput.SCORE_ENERGY:
            score_scale_t = self.score_scale(t)
            return grad * score_scale_t[:, None], energy * score_scale_t
        else:
            raise ValueError('Incorrect forward pass mode')


if __name__ == '__main__':
    score_net = MLPScoreNet(dim=2, hidden_counts=[30, 50, 30], embed_dim=10)
    print(score_net)
    x = torch.randn(1, 2)
    print(x)
    print(score_net(x, torch.tensor([0.4])))
