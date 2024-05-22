import time

import matplotlib.pyplot as plt
from multi_source.main.my_utils import to_dict
import torch
import torch.nn.functional as F
import torchinfo
from einops import reduce
from ml_logger import logger
from params_proto import ParamsProto
from params_proto import Proto
from torch.optim import Adam
from torch.utils.data import DataLoader
from diffusion_cubes.model2 import CubeUnet
from diffusion_2d.utils import sample_image
from diffusion_cubes.loader import Clevr2DPosDataset
from diffusion_cubes.model import Model, UNetModel
from single_instrument.utils import pad_dimensions


class Args(ParamsProto):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_path = Proto(env='$DATASET_ROOT/clevr/clevr_pos_data_128_30000.npz')
    image_size = 64
    seed = 25
    n_epochs = 2000
    batch_size = 32 if torch.cuda.is_available() else 1
    lr = 1e-4
    checkpoint_interval = 50
    render_interval = 5
    sampler_batch_size = 4
    num_timesteps = 50
    random_crop = False
    random_flip = False

    class unet_kwargs(Proto):
        dim = 128 # Channel count
        dim_mults = (1, 2, 4, 8, 8)
        channels = 3
        resnet_block_groups = 8
        learned_sinusoidal_dim = 32
        attn_dim_head = 64
        attn_heads = 8


def loss_fn(model, x_0, y, mask):
    batch, device = x_0.shape[0], x_0.device
    sigmas = model.sigma_distribution(num_samples=batch, device=device)
    sigmas_padded = pad_dimensions(sigmas, x_0)

    noise = torch.randn_like(x_0)
    x_noisy = x_0 + sigmas_padded * noise
    x_denoised = model.denoise_fn(x_noisy, sigmas, y, mask)

    losses = F.mse_loss(x_denoised, x_0, reduction="none")
    losses = reduce(losses, "b ... -> b", "mean")
    losses = losses * model.loss_weight(sigmas)
    loss = losses.mean()
    return loss


def main(**deps):
    Args._update(deps)

    print(logger)
    logger.log_text("""
       charts:
       - yKey: loss/mean
         xKey: epoch
       - yKey: steps_per_sec/mean
         xKey: epoch
       - type: image
         glob: samples/0.png
       - type: image
         glob: samples/5.png
       - type: image
         glob: samples/10.png
       - type: image
         glob: samples/15.png
       - type: image
         glob: samples/20.png
       - type: image
         glob: samples/25.png
       - type: image
         glob: samples/50.png
       - type: image
         glob: samples/75.png
       - type: image
         glob: samples/100.png
       - type: image
         glob: samples/150.png
       - type: image
         glob: samples/200.png
       - type: image
         glob: samples/300.png
       - type: image
         glob: samples/500.png
       - type: image
         glob: samples/700.png
       - type: image
         glob: samples/1000.png
       """, ".charts.yml", dedent=True, overwrite=True)

    torch.manual_seed(Args.seed)
    logger.log_params(Args=vars(Args))

    attention_resolutions = "32,16,8"
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(Args.image_size // int(res))

    model = Model(net=UNetModel(in_channels=3,
                                model_channels=128,
                                out_channels=3,
                                num_res_blocks=2,
                                attention_resolutions=tuple(attention_ds),
                                dropout=0.0,
                                channel_mult=(1, 2, 2, 2),
                                conv_resample=True,
                                dims=2,
                                use_checkpoint=False,
                                num_heads=4,
                                num_head_channels=64,
                                num_heads_upsample=-1,
                                use_scale_shift_norm=False,
                                resblock_updown=True,
                                encoder_channels=None,
                                )).to(Args.device)

    model_stats = torchinfo.summary(model, verbose=0)
    logger.print(str(model_stats))

    dataset = Clevr2DPosDataset(
        data_path=Args.dataset_path,
        resolution=Args.image_size,
        random_crop=Args.random_crop,
        random_flip=Args.random_flip
    )

    logger.print("Dataset size is: " + str(len(dataset)))

    data_loader = DataLoader(dataset, batch_size=Args.batch_size, shuffle=True, num_workers=4)
    optimizer = Adam(model.parameters(), lr=Args.lr)
    logger.start('log_timer')
    for epoch in range(0, Args.n_epochs + 1):
        if epoch % Args.render_interval == 0:
            fx = torch.rand((Args.sampler_batch_size,)) * 0.4 + 0.3
            fy = torch.rand((Args.sampler_batch_size,)) * 0.4 + 0.3
            coords = torch.stack([fx, fy], dim=1).to(Args.device)
            logger.print('rendering diffusion steps...', color="yellow")
            images = sample_image(model, coords, Args, batch_size=Args.sampler_batch_size)
            logger.print('saving the samples...', color="green")
            fx = fx.cpu().detach().numpy()
            fy = fy.cpu().detach().numpy()
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
            for i, image in enumerate(images):
                np_image = image.cpu().detach().numpy().transpose(1, 2, 0)
                np_image = (np_image + 1) / 2.0
                row = i // 2
                col = i % 2
                axs[row, col].imshow(np_image, interpolation='nearest')
                axs[row, col].scatter(fx[i] * Args.image_size,
                                      (1 - fy[i]) * Args.image_size,
                                      color='red',
                                      marker='x',
                                      s=80)
                axs[row, col].axis('off')
            plt.tight_layout()
            plt.show()
            logger.savefig(f"samples/{epoch}.png")
        if epoch == Args.n_epochs:
            break

        timer_steps = 0
        for x, y, mask in data_loader:
            x = x.to(Args.device).float()
            y = y.to(Args.device).float()
            mask = mask.to(Args.device)
            loss = loss_fn(model, x, y, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.store_metrics(loss=loss.cpu().item())
            logger.print(loss.cpu().item())
            timer_steps += 1

        if Args.checkpoint_interval and epoch % Args.checkpoint_interval == 0:
            logger.print(f"Saving the checkpoint at epoch {epoch}", color="yellow")
            logger.torch_save(model, f"checkpoints/model.pkl")

        logger.store_metrics({
            'steps_per_sec': timer_steps / logger.split('log_timer')
        })

        logger.log_metrics_summary(key_values={"epoch": epoch})
        logger.print(f'Completed epoch:{epoch}, time = {time.asctime(time.localtime())}')

if __name__ == '__main__':
    main()
