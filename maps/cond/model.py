import torch
import torch.nn as nn
from maps.uncond.utils import pad_dimensions
from audio_diffusion_pytorch import LogNormalDistribution
from gutils.basic import to_tuple2, torch_randint
from maps.uncond.model import step_cnt
from maps.cond.diti import CondDiTi
from ml_logger import logger


class CondEDM(nn.Module):
    def __init__(self,
                 net,
                 device,
                 diffusion_sigma_distribution=LogNormalDistribution(mean=-3.0, std=1.0),
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

    def denoise_fn(self, x_noisy, sigmas, x_cond):
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas, x_noisy)
        x_pred = self.unet(c_in * x_noisy, c_noise, x_cond)
        x_denoised = c_skip * x_noisy + c_out * x_pred
        return x_denoised.clamp(-1.0, 1.0)

    def loss_weight(self, sigmas):
        return (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2

    def forward(self, x_noisy, sigmas, x_cond):
        x_denoised = self.denoise_fn(x_noisy, sigmas, x_cond)
        sigmas_padded = pad_dimensions(sigmas, x_noisy)
        return (x_noisy - x_denoised) / sigmas_padded


class ControlCondModel(nn.Module):
    def __init__(self, diff_model, unet):
        super().__init__()
        self.diff_model = diff_model
        self.diff_model.requires_grad_(False)
        self.unet = unet

    def forward(self, x, t, x_c):
        out_uncond = self.diff_model.unet(x, t)
        return self.unet(torch.cat((out_uncond, x_c), dim=1), t)


class CondModel(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t, x_c):
        return self.unet(torch.cat((x, x_c), dim=1), t)


class FakeCondModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, t, x_c):
        return torch.zeros_like(x)


class MultiDiffusion(nn.Module):
    def __init__(self, base_model, base_size, target_size, stride, kernel):
        super().__init__()
        self.base_size = to_tuple2(base_size)
        self.target_size = to_tuple2(target_size)
        self.stride = to_tuple2(stride)
        steps_w = step_cnt(self.base_size[0], self.stride[0], self.target_size[0])
        steps_h = step_cnt(self.base_size[1], self.stride[1], self.target_size[1])
        if steps_w is not None and steps_h is not None:
            self.steps = (steps_w, steps_h)
        else:
            raise ValueError('Invalid size/stride combination')
        self.base_model = base_model
        self.kernel = nn.Parameter(kernel[None, None, :, :], requires_grad=False)

    def forward(self, x_noisy, sigmas, x_c):
        weights = torch.zeros_like(x_noisy)
        outputs = torch.zeros_like(x_noisy)
        # Averaging over x_noised is the same as averaging of scores
        # since it's a linear function of x_noised
        for i in range(self.steps[0]):
            for j in range(self.steps[1]):
                h0 = self.stride[0] * i
                w0 = self.stride[1] * j
                H, W = self.base_size
                outputs[:, :, h0:h0+H, w0:w0+W] += \
                        self.base_model(x_noisy[:, :, h0:h0+H, w0:w0+W],
                                        sigmas,
                                        x_c[:, :, h0:h0+H, w0:w0+W]) * \
                        self.kernel
                weights[:, :, h0:h0+H, w0:w0+W] += self.kernel
        return outputs / weights


class CondDiTiDevil(nn.Module):
    def __init__(self, base_model, base_size, stride, diti_kwargs, max_mult_shape=None):
        super().__init__()
        self.base_model = base_model
        self.base_model.requires_grad_(False)
        self.diti = CondDiTi(
                 base_size=base_size,
                 patch_size=diti_kwargs.patch_size,
                 in_channels=3,
                 hidden_size=diti_kwargs.hidden_size,
                 depth=diti_kwargs.depth,
                 num_heads=diti_kwargs.num_heads,
                 mlp_ratio=diti_kwargs.mlp_ratio,
                 out_channels=3,
                 stride=stride,
                 add_pos_emb=diti_kwargs.add_pos_emb,
                 use_rotary_attn=diti_kwargs.use_rotary_attn)
        self.base_size = base_size
        self.max_mult_shape = max_mult_shape
        self.stride = stride

    def forward(self, x, t, x_c):
        MH = step_cnt(self.base_size, self.stride, x.shape[2])
        MW = step_cnt(self.base_size, self.stride, x.shape[3])
        scores = []
        for i in range(MH):
            for j in range(MW):
                L1 = i * self.stride
                R1 = L1 + self.base_size
                L2 = j * self.stride
                R2 = L2 + self.base_size
                scores.append(self.base_model.unet(x[:, :, L1:R1, L2:R2], t, x_c[:, :, L1:R1, L2:R2]))
        diti_input = torch.stack(scores, dim=1)
        N, _, C, H, W = diti_input.shape
        diti_input = diti_input.reshape(
                (N, MH, MW, C, H, W))
        pos = (0, 0)
        # if self.max_mult_shape is not None:
        #    pos = (torch_randint(0, self.max_mult_shape[0] - MH + 1) * self.stride,
        #           torch_randint(0, self.max_mult_shape[1] - MW + 1) * self.stride)
        #else:
        #    pos = (0, 0)
        return self.diti(diti_input, t, x_c, pos=pos)
