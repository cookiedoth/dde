import math

import torch
import torch.nn as nn
from audio_diffusion_pytorch import LogNormalDistribution
from denoising_diffusion_pytorch import Unet

from maps.uncond.dit import DiT
from maps.uncond.diti import DiTi
from maps.uncond.utils import pad_dimensions


# Adapted from https://github.com/NVlabs/edm
class Model(nn.Module):
    def __init__(self,
                 net,
                 device,
                 diffusion_sigma_distribution=LogNormalDistribution(mean=-1.0, std=1.6),
                 diffusion_sigma_data=0.2):
        super().__init__()
        self.unet = net
        self.unet.to(device)
        self.sigma_distribution = diffusion_sigma_distribution
        self.sigma_data = diffusion_sigma_data
        self.device = device

    def get_scale_weights(self, sigmas, x_noisy):
        sigma_data = self.sigma_data
        sigmas_padded = pad_dimensions(sigmas, x_noisy)
        c_skip = (sigma_data ** 2) / (sigmas_padded ** 2 + sigma_data ** 2)
        c_out = (sigmas_padded * sigma_data * (sigma_data ** 2 + sigmas_padded ** 2) ** -0.5)
        c_in = (sigmas_padded ** 2 + sigma_data ** 2) ** -0.5
        c_noise = torch.log(sigmas) * 0.25
        return c_skip, c_out, c_in, c_noise

    def denoise_fn(self, x_noisy, sigmas):
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas, x_noisy)
        x_pred = self.unet(c_in * x_noisy, c_noise)
        x_denoised = c_skip * x_noisy + c_out * x_pred
        return x_denoised.clamp(-1.0, 1.0)

    def loss_weight(self, sigmas):
        return (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2

    def forward(self, x_noisy, sigmas):
        x_denoised = self.denoise_fn(x_noisy, sigmas)
        sigmas_padded = pad_dimensions(sigmas, x_noisy)
        return (x_noisy - x_denoised) / sigmas_padded


def to_tuple(x):
    return (x, x) if isinstance(x, int) else x


def step_cnt(small, stride, large):
    return None if large < small or (large - small) % stride != 0 else (large - small) // stride + 1


class MultiDiffusion(nn.Module):
    def __init__(self, base_model, base_size, target_size, stride, kernel):
        super().__init__()
        self.base_size = to_tuple(base_size)
        self.target_size = to_tuple(target_size)
        self.stride = to_tuple(stride)
        steps_w = step_cnt(self.base_size[0], self.stride[0], self.target_size[0])
        steps_h = step_cnt(self.base_size[1], self.stride[1], self.target_size[1])
        if steps_w is not None and steps_h is not None:
            self.steps = (steps_w, steps_h)
        else:
            raise ValueError('Invalid size/stride combination')
        self.base_model = base_model
        self.kernel = nn.Parameter(kernel[None, None, :, :], requires_grad=False)

    def forward(self, x_noisy, sigmas):
        weights = torch.zeros_like(x_noisy)
        outputs = torch.zeros_like(x_noisy)
        # Averaging over x_noised is the same as averaging of scores
        # since it's a linear function of x_noised
        for i in range(self.steps[0]):
            for j in range(self.steps[1]):
                w0 = self.stride[0] * i
                h0 = self.stride[1] * j
                outputs[:, :, w0:w0 + self.base_size[0], h0:h0 + self.base_size[1]] += \
                    self.base_model(x_noisy[:, :, w0:w0 + self.base_size[0], h0:h0 + self.base_size[0]], sigmas) * \
                    self.kernel
                weights[:, :, w0:w0 + self.base_size[0], h0:h0 + self.base_size[1]] += self.kernel
        return outputs / weights


def get_kernel(size, corner_coef):
    row = (torch.arange(size) - (size - 1) / 2) / ((size - 1) / 2)
    matrix = (row ** 2)[None, :] + (row ** 2)[:, None]
    coef = math.log(corner_coef) / 2
    matrix *= coef
    return torch.exp(matrix)


class RNN2D(nn.Module):
    def __init__(self, base_model, base_size, target_size, stride, hidden_channels, unet_kwargs, kernel):
        super().__init__()
        self.base_size = to_tuple(base_size)
        self.target_size = to_tuple(target_size)
        self.stride = to_tuple(stride)
        self.base_model = base_model
        self.base_model.requires_grad_(False)
        unet_kwargs['channels'] = 3 + 2 * hidden_channels
        self.unet = Unet(**unet_kwargs)
        steps_w = step_cnt(self.base_size[0], self.stride[0], self.target_size[0])
        steps_h = step_cnt(self.base_size[1], self.stride[1], self.target_size[1])
        if steps_w is not None and steps_h is not None:
            self.steps = (steps_w, steps_h)
        else:
            raise ValueError('Invalid size/stride combination')
        self.kernel = nn.Parameter(kernel[None, None, :, :], requires_grad=False)
        self.hidden_channels = hidden_channels

    def forward(self, x_noisy, sigmas):
        weights = torch.zeros_like(x_noisy)
        outputs = torch.zeros_like(x_noisy)
        hiddens1 = torch.zeros(
            (x_noisy.shape[0], self.hidden_channels, self.steps[1], self.base_size[0], self.base_size[1]),
            device=x_noisy.device)  # (B, channels, steps, W, H)
        for i in range(self.steps[0]):
            hidden = torch.zeros((x_noisy.shape[0], self.hidden_channels, self.base_size[0], self.base_size[1]),
                                 device=x_noisy.device)
            for j in range(self.steps[1]):
                w0 = self.stride[0] * i
                h0 = self.stride[1] * j
                diff_output = self.base_model.unet(x_noisy[:, :, w0:w0 + self.base_size[0], h0:h0 + self.base_size[0]],
                                                   sigmas)
                rnn_input = torch.cat((diff_output, hidden, hiddens1[:, :, j, :, :]), dim=1)
                rnn_output = self.unet(rnn_input, sigmas)
                outputs[:, :, w0:w0 + self.base_size[0], h0:h0 + self.base_size[1]] += rnn_output[:, :3, :,
                                                                                       :] * self.kernel
                hidden = rnn_output[:, 3:3 + self.hidden_channels, :, :]
                hiddens1[:, :, j, :, :] = rnn_output[:, 3 + self.hidden_channels:, :, :]
                weights[:, :, w0:w0 + self.base_size[0], h0:h0 + self.base_size[1]] += self.kernel
        return outputs / weights


class TransformerDevil(nn.Module):
    def __init__(self, base_model, base_size, mult_shape, stride, dit_kwargs):
        super().__init__()
        self.base_model = base_model
        self.base_size = base_size
        self.mult_shape = mult_shape
        self.stride = stride
        self.base_model.requires_grad_(False)
        self.dit = DiT(**dit_kwargs)

    def forward(self, x_noisy, sigmas):
        scores = torch.zeros_like(x_noisy)
        w = torch.zeros_like(x_noisy)
        for i in range(self.mult_shape[0]):
            for j in range(self.mult_shape[1]):
                L1 = i * self.stride
                R1 = L1 + self.base_size
                L2 = j * self.stride
                R2 = L2 + self.base_size
                scores[:, :, L1:R1, L2:R2] = self.base_model.unet(x_noisy[:, :, L1:R1, L2:R2], sigmas)
                w[:, :, L1:R1, L2:R2] += 1
        scores = scores / w
        out = self.dit(scores, sigmas)
        return out


class TransformerAlpha(nn.Module):
    def __init__(self, base_model, base_size, mult_shape, stride, dit_kwargs):
        super().__init__()
        self.base_model = base_model
        self.base_size = base_size
        self.mult_shape = mult_shape
        self.stride = stride
        self.base_model.requires_grad_(False)
        assert dit_kwargs['in_channels'] == 4 * mult_shape[0] * mult_shape[1]
        assert dit_kwargs['out_channels'] == 3
        self.dit = DiT(**dit_kwargs)

    def forward(self, x_noisy, sigmas):
        w_ch = self.mult_shape[0] * self.mult_shape[1] * 4
        dit_input = torch.zeros((x_noisy.shape[0], w_ch, x_noisy.shape[2], x_noisy.shape[3]),
                                device=x_noisy.device)
        ch_ptr = 0
        for i in range(self.mult_shape[0]):
            for j in range(self.mult_shape[1]):
                L1 = i * self.stride
                R1 = L1 + self.base_size
                L2 = j * self.stride
                R2 = L2 + self.base_size
                dit_input[:, ch_ptr:ch_ptr + 3, L1:R1, L2:R2] = self.base_model.unet(x_noisy[:, :, L1:R1, L2:R2],
                                                                                     sigmas)
                dit_input[:, ch_ptr + 3, L1:R1, L2:R2] = 1.0
                ch_ptr += 4
        output = self.dit(dit_input, sigmas)
        return output


class TransformerDevilWithW(nn.Module):
    def __init__(self, base_model, base_size, mult_shape, stride, dit_kwargs):
        super().__init__()
        self.base_model = base_model
        self.base_size = base_size
        self.mult_shape = mult_shape
        self.stride = stride
        self.base_model.requires_grad_(False)
        self.dit = DiT(**dit_kwargs)
        dit_kwargs['in_channels'] = mult_shape[0] * mult_shape[1] * 3
        self.dit_w = DiT(**dit_kwargs)

    def forward(self, x_noisy, sigmas):
        scores = torch.zeros_like(x_noisy)

        w_ch = self.mult_shape[0] * self.mult_shape[1] * 3
        dit_w_input = torch.zeros((x_noisy.shape[0], w_ch, x_noisy.shape[2], x_noisy.shape[3]), device=x_noisy.device)
        ch_ptr = 0
        for i in range(self.mult_shape[0]):
            for j in range(self.mult_shape[1]):
                L1 = i * self.stride
                R1 = L1 + self.base_size
                L2 = j * self.stride
                R2 = L2 + self.base_size
                dit_w_input[:, ch_ptr:ch_ptr + 3, L1:R1, L2:R2] = x_noisy[:, :, L1:R1, L2:R2]
                ch_ptr += 3
        dit_w_output = self.dit_w(dit_w_input, sigmas)

        w = torch.zeros_like(x_noisy)
        ch_ptr = 0
        for i in range(self.mult_shape[0]):
            for j in range(self.mult_shape[1]):
                L1 = i * self.stride
                R1 = L1 + self.base_size
                L2 = j * self.stride
                R2 = L2 + self.base_size
                kernel = torch.exp(dit_w_output[:, ch_ptr:ch_ptr + 3, L1:R1, L2:R2])
                scores[:, :, L1:R1, L2:R2] = self.base_model.unet(x_noisy[:, :, L1:R1, L2:R2], sigmas) * kernel
                w[:, :, L1:R1, L2:R2] += kernel
                ch_ptr += 3
        scores = scores / w
        out = self.dit(scores, sigmas)
        return out


class DiTiDevil(nn.Module):
    def __init__(self, base_model, base_size, mult_shape, stride, diti_kwargs):
        super().__init__()
        self.base_model = base_model
        self.base_model.requires_grad_(False)
        self.diti = DiTi(
            base_size=base_size,
            patch_size=diti_kwargs.patch_size,
            in_channels=3,
            hidden_size=diti_kwargs.hidden_size,
            depth=diti_kwargs.depth,
            num_heads=diti_kwargs.num_heads,
            mlp_ratio=diti_kwargs.mlp_ratio,
            out_channels=3,
            stride=stride)
        self.base_size = base_size
        self.mult_shape = mult_shape
        self.stride = stride

    def forward(self, x, t):
        scores = []
        for i in range(self.mult_shape[0]):
            for j in range(self.mult_shape[1]):
                L1 = i * self.stride
                R1 = L1 + self.base_size
                L2 = j * self.stride
                R2 = L2 + self.base_size
                scores.append(self.base_model.unet(x[:, :, L1:R1, L2:R2], t))
        diti_input = torch.stack(scores, dim=1)
        N, _, C, H, W = diti_input.shape
        MH, MW = self.mult_shape
        diti_input = diti_input.reshape(
            (N, MH, MW, C, H, W))
        return self.diti(diti_input, t)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, sigma):
        return torch.zeros_like(x)


if __name__ == '__main__':
    print(get_kernel(5, 0.1))
