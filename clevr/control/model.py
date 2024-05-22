import torch
import torch.nn as nn

from clevr.model import UNetModel
from maps.uncond.diti import DiTi


class RNNDevil(nn.Module):
    def __init__(self, cube_model, image_size, device, hidden_channels=64):
        super().__init__()
        self.diff_model = cube_model
        self.diff_model.requires_grad_(False)
        self.hidden_channels = hidden_channels
        self.model = UNetModel(
            in_channels=6 + hidden_channels,
            model_channels=image_size * 2,
            out_channels=3 + hidden_channels,
            num_res_blocks=2,
            attention_resolutions=(2, 4, 8),
            dropout=0.0,
            channel_mult=(1, 2, 3, 3),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            num_heads=8,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=True,
            encoder_channels=None,
        ).to(device)
        self.image_size = image_size
        self.device = device

    def forward(self, x, t, y, mask):
        x.requires_grad_(True)
        batch_size = y.shape[0]
        c = y.shape[1]
        ret = torch.zeros((batch_size, 3, self.image_size, self.image_size), device=self.device)
        hidden = torch.zeros((batch_size, self.hidden_channels, self.image_size, self.image_size), device=self.device)
        for i in range(0, c):
            output = self.diff_model.unet(x, t, y[:, i, :], mask[:, i])
            concat = self.model(torch.cat([output, ret, hidden], dim=1), t, y[:, i, :], mask[:, i])
            ret = concat[:, :3, :, :]
            hidden = concat[:, 3:, :, :]
        return ret


class TransformerDevil(nn.Module):
    def __init__(self, cube_model, image_size, device):
        super().__init__()
        self.diff_model = cube_model
        self.diff_model.requires_grad_(False)
        self.dit = DiTi(base_size=image_size,
                        patch_size=4,
                        in_channels=4,
                        hidden_size=384,
                        depth=12,
                        num_heads=6,
                        mlp_ratio=4.0,
                        out_channels=3,
                        stride=0).to(device)
        self.image_size = image_size
        self.device = device

    def forward(self, x, t, y, mask):
        x.requires_grad_(True)
        outputs = []
        for i in range(y.shape[1]):
            w, h = x.shape[2], x.shape[3]
            y1 = (y[:, i, 0] * w).int()
            y2 = (y[:, i, 1] * h).int()
            extra_channel = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
            extra_channel[torch.arange(x.shape[0], device=x.device), 0, y1, w - 1 - y2] = 1.0
            extra_channel[~mask[:, i], :, :, :] = 0.0
            out = self.diff_model.unet(x, t, y[:, i, :], mask[:, i])
            outputs.append(torch.cat([out, extra_channel], dim=1))
        dit_input = torch.stack(outputs, dim=1)
        dit_input = torch.unsqueeze(dit_input, dim=1)
        return self.dit(dit_input, t)


class ScoreSum(nn.Module):
    def __init__(self, cube_model):
        super().__init__()
        self.diff_model = cube_model

    def forward(self, x, t, y, guidance_scale=4):
        b = x.shape[0]
        results = []
        uncond = self.diff_model(x, t, torch.zeros((b, 2), device=x.device), torch.zeros((b,), dtype=torch.bool, device=x.device))
        for i in range(y.shape[1]):
            results.append(self.diff_model(x, t, y[:, i, :], torch.ones((b, ), dtype=torch.bool, device=x.device)) - uncond)
        return uncond + sum(results) * guidance_scale
