import torch
import torch.nn as nn

from diffusion_2d.model import InferenceOutput, GaussianFourierProjection
from diffusion_2d_control.control_model import AttentionBlock, CrossAttentionBlock
from diffusion_cubes.model import UNetModel
from maps.uncond.diti import DiTi


class ControlDevil(nn.Module):
    def __init__(self, cube_model, classifier_model, encoder, decoder, image_size, score_scale, emb_dim=256,
                 num_layers=3):
        super().__init__()
        self.cube_model = cube_model
        self.classifier_model = classifier_model
        self.image_size = image_size
        self.emb_dim = emb_dim
        self.num_layers = num_layers

        self.null_emb = nn.Linear(1, emb_dim)
        self.time_embed = GaussianFourierProjection(embed_dim=emb_dim)

        self.label_embed = nn.Linear(2, emb_dim)
        self.null_y_emb = nn.Linear(1, emb_dim)
        self.label_attention = AttentionBlock(dim=emb_dim, num_heads=4)

        self.encoder = encoder
        self.encoder_linear = nn.Linear(128 * (image_size // 8) * (image_size // 8), emb_dim)
        self.out_models_linear = nn.Linear(emb_dim * 2, emb_dim)

        self.model_attentions = torch.nn.ModuleList(
            [AttentionBlock(dim=emb_dim, num_heads=4) for _ in range(num_layers)])
        self.cross_attentions = torch.nn.ModuleList(
            [CrossAttentionBlock(dim=emb_dim, num_heads=4) for _ in range(num_layers)])

        self.score_scale = score_scale
        self.final_linear = nn.Linear(1, 64)
        self.end_attention = AttentionBlock(dim=64, num_heads=4)

        self.decoder = decoder

    def forward(self, x, t, y, mask, mode):
        B = x.shape[0]
        num_labels = y.shape[1] + 1

        y = y.permute((1, 0, 2))
        # y of size (num_labels x B x 2)
        y = torch.cat((y, torch.zeros((1, B, 2), device=x.device)), dim=0)

        mask = mask.permute((1, 0))
        # mask of size (num_labels x B)

        mask = torch.cat((mask, torch.zeros((1, B), dtype=torch.bool, device=x.device)), dim=0)

        x.requires_grad_(True)

        emb_t = self.time_embed(t)
        # emb_t of size (B x emb_dim)

        embeds_list = []

        add = torch.zeros_like(x)
        py_x = torch.sigmoid(self.classifier_model((x + 1) / 2))
        norm = torch.zeros((x.shape[0],), device=x.device)
        for i in range(num_labels):
            output = self.cube_model(x, t, y[i], mask[i], mode=InferenceOutput.EPS, eval_mode=True)
            # output of size (B x 3 x image_size x image_size)

            if i < num_labels - 1:
                ry = torch.zeros_like(y[i])
                ry[:, 0] = 1 - y[i, :, 1]
                ry[:, 1] = y[i, :, 0]
                ry = torch.round(ry * self.image_size).int()
                ry = torch.clip(ry, 0, self.image_size - 1)

                selected_py_x = py_x[torch.arange(py_x.shape[0]), ry[:, 0], ry[:, 1]]
                selected_py_x = selected_py_x * mask[i].float()
                norm += selected_py_x
                add = add + output * selected_py_x.view([-1] + [1] * (len(x.shape) - 1))

            output = self.encoder(x - output)
            # output of size (B x 128 * image_size // 8 * image_size // 8)

            output = self.encoder_linear(output)
            # output of size (B x emb_dim)

            if i < num_labels - 1:
                output[~mask[i]] = self.null_emb.weight[0][None].repeat(output[~mask[i]].shape[0], 1)

            output_with_emb_t_m = torch.cat((output, emb_t), dim=1)
            # output_with_emb_t_m of size (B x emb_dim * 2)

            output = self.out_models_linear(output_with_emb_t_m)
            embeds_list.append(output)

        add = add / torch.max(norm.view([-1] + [1] * (len(x.shape) - 1)), torch.tensor([1e-6], device=x.device))

        model_embeds = torch.stack(embeds_list, dim=1)
        # model_embeds of size (B x num_labels x emb_dim)

        y = y.permute((1, 0, 2))
        # y of size (B x num_labels x 2)

        mask = mask.permute((1, 0))
        # mask of size (B x num_labels)

        emb_y = torch.zeros((B, num_labels, self.emb_dim), device=x.device)

        emb_y[mask] = self.label_embed(y[mask])
        emb_y[~mask] = self.null_y_emb.weight[0][None].repeat(y[~mask].shape[0], 1)
        # emb_y of size (B x num_labels x emb_dim)

        emb_y = self.label_attention(emb_y)

        for i in range(self.num_layers):
            model_embeds = self.model_attentions[i](model_embeds)
            model_embeds = self.cross_attentions[i](model_embeds, emb_y)

        embeds = torch.mean(model_embeds, dim=1)
        # embeds of size (B x emb_dim)

        embeds = torch.unsqueeze(embeds, dim=2)
        # embeds of size (B x emb_dim x 1)

        out = self.final_linear(embeds)
        # out of size (B x emb_dim x 16)

        out = self.end_attention(out)
        # out of size (B x emb_dim x 16)

        out = torch.squeeze(out.mean(dim=2))
        # out of size (B x emb_dim)

        image_out = x - self.decoder(out)

        # image_out of size (B x 3 x image_size x image_size)

        final_out = add + image_out
        if mode == InferenceOutput.EPS:
            return final_out
        elif mode == InferenceOutput.SCORE:
            return final_out * self.score_scale(t).view([-1] + [1] * (len(x.shape) - 1))
        else:
            raise ValueError('Incorrect forward pass mode')


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
