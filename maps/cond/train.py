import torch
import torchinfo
from maps.uncond.utils import pad_dimensions
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from params_proto import ParamsProto, Proto
from einops import reduce
from maps.uncond.utils import TimedAction, tensor_to_image
from maps.cond.model import CondEDM, CondModel, ControlCondModel, CondDiTiDevil, FakeCondModel
from denoising_diffusion_pytorch import Unet
from maps.cond.dataset import SatMapDataset
from torch.utils.data import DataLoader
from ml_logger import logger
from utils.basic import to_dict
from maps.uncond.dit import DiT
from maps.uncond.utils import sample_images
from audio_diffusion_pytorch import LogNormalDistribution
from ema_pytorch import EMA


test_run = False


class Args(ParamsProto):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_path = ''
    unet_control = False
    dit_control = False
    diti_devil = True
    use_ema = True
    base_model_path = ''
    image_size = 96
    seed = 42
    n_epochs = 3000
    display_samples = [5, 15, 30, 45, 60, 80, 100, 120, 150, 180, 220, 260, 300, 350, 400, 450, 500]
    display_size = 3
    display_figsize = (8, 8)
    batch_size = 8 if torch.cuda.is_available() else 1
    num_timesteps = 150
    lr = 3e-5
    checkpoint_interval = 5
    render_interval = 5

    sigma_min = 1e-4
    sigma_max = 20.0
    s_churn = 20.0
    rho = 7

    log_sigma_mean = -1.0
    log_sigma_std = 1.6
    sigma_data = 0.2

    flip_cond = True

    class unet_kwargs_control(Proto):
        dim = 64 # Channel count
        dim_mults = (1, 2, 4, 8)
        channels = 6
        out_dim = 3
        resnet_block_groups = 8
        learned_sinusoidal_dim = 32
        attn_dim_head = 64
        attn_heads = 8


    class unet_kwargs_full(Proto):
        dim = 128 # Channel count
        dim_mults = (1, 2, 4, 8, 8)
        channels = 6
        out_dim = 3
        resnet_block_groups = 8
        learned_sinusoidal_dim = 32
        attn_dim_head = 64
        attn_heads = 8


    class dit_kwargs(Proto):
        input_size = 64
        patch_size = 4
        in_channels = 6
        hidden_size = 384
        depth = 12
        num_heads = 6
        mlp_ratio = 4.0
        learn_sigma = False
        out_channels = 3

    base_size = 64
    max_mult_shape = (3, 3)
    stride = 32

    class diti_kwargs(Proto):
        patch_size = 2
        hidden_size = 384
        depth = 6
        num_heads = 6
        mlp_ratio = 4.0
        add_pos_emb = False
        use_rotary_attn = True


def loss_fn(model, x_0, x_c):
    batch, device = x_0.shape[0], x_0.device
    sigmas = model.sigma_distribution(num_samples=batch, device=device)
    sigmas_padded = pad_dimensions(sigmas, x_0)

    noise = torch.randn_like(x_0)
    x_noisy = x_0 + sigmas_padded * noise
    x_denoised = model.denoise_fn(x_noisy, sigmas, x_c)

    losses = F.mse_loss(x_denoised, x_0, reduction='none')
    losses = reduce(losses, 'b ... -> b', 'mean')
    losses = losses * model.loss_weight(sigmas)
    loss = losses.mean()
    return loss


def setup_logger():
    chart_string = """
       charts:
       - yKey: loss/mean
         xKey: epoch
       - yKey: steps_per_sec/mean
         xKey: epoch"""
    for epoch in Args.display_samples:
        chart_string = chart_string + f"""
       - type: image
         glob: samples/{epoch:04}.png
         """
    logger.log_text(chart_string, '.charts.yml', dedent=True, overwrite=True)


def sample(dataset, model, epoch):
    sample_cnt = Args.display_size
    sample_ids = torch.randint(0, len(dataset), (sample_cnt,))
    samples_sat = torch.stack([dataset[i][0] for i in sample_ids]).to(Args.device)
    samples_map = torch.stack([dataset[i][1] for i in sample_ids]).to(Args.device)
    samples = sample_images(lambda x, t: model(x, t, samples_sat),
                            sample_cnt,
                            Args)

    _, ax = plt.subplots(Args.display_size, 3, figsize=Args.display_figsize)
    for row in range(Args.display_size):
        ax[row, 0].imshow(tensor_to_image(samples_sat[row]))
        ax[row, 1].imshow(tensor_to_image(samples_map[row]))
        ax[row, 2].imshow(tensor_to_image(samples[row]))
        for i in range(3):
            ax[row, i].axis('off')

    plt.tight_layout()
    logger.savefig(f'samples/{epoch:04}.png')


def cond_edm(model):
    return CondEDM(model,
                   device=Args.device,
                   diffusion_sigma_distribution=LogNormalDistribution(mean=Args.log_sigma_mean, std=Args.log_sigma_std),
                   diffusion_sigma_data=Args.sigma_data)


def load_base_model():
    if test_run:
        return CondEDM(FakeCondModel(), device=Args.device)
    else:
        base_model = torch.load(Args.base_model_path, map_location=Args.device)
        if isinstance(base_model, EMA):
            return base_model.ema_model


def main(**deps):
    Args._update(deps)
    setup_logger()
    logger.log_params(Args=vars(Args))

    with TimedAction('initialize_model'):
        if Args.unet_control:
            map_model = load_base_model()
            unet = Unet(**to_dict(Args.unet_kwargs_control))
            model = cond_edm(ControlCondModel(map_model, unet))
        elif Args.dit_control:
            map_model = load_base_model()
            dit = DiT(**to_dict(Args.dit_kwargs))
            model = cond_edm(ControlCondModel(map_model, dit))
        elif Args.diti_devil:
            map_model = load_base_model()
            model = CondDiTiDevil(map_model,
                                  Args.base_size,
                                  Args.stride,
                                  Args.diti_kwargs,
                                  max_mult_shape=Args.max_mult_shape)
            model = cond_edm(model)
        else:
            unet = Unet(**to_dict(Args.unet_kwargs_full))
            model = cond_edm(CondModel(unet))

    if Args.use_ema:
        ema = EMA(model,
                  beta = 0.9999,
                  update_after_step = 100,
                  update_every = 10)

    model_stats = torchinfo.summary(model, verbose=0)
    logger.print(str(model_stats))

    with TimedAction('load_dataset'):
        full_dataset = SatMapDataset(Args.dataset_path, Args.image_size, flip_cond=Args.flip_cond)
        train_size = int(len(full_dataset) * 0.8)
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=Args.batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=Args.lr)

    logger.start('log_timer')
    for epoch in range(1, Args.n_epochs + 1):
        logger.split('log_timer')

        timer_steps = 0
        for x_c, x_0 in train_dataloader:
            x_c = x_c.to(Args.device)
            x_0 = x_0.to(Args.device)
            loss = loss_fn(model, x_0, x_c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.store_metrics(loss=loss.item())
            timer_steps += 1
            if Args.device == 'cpu':
                logger.print('batch complete')
            if Args.use_ema:
                ema.update()

        logger.store_metrics({
            'steps_per_sec': timer_steps / logger.split('log_timer')
        })
        logger.log_metrics_summary(key_values={'epoch' : epoch})

        if epoch % Args.checkpoint_interval == 0:
            logger.print(f'Saving the checkpoint at epoch {epoch}', color='yellow')
            logger.torch_save(ema if Args.use_ema else model, f'checkpoints/model_{epoch:04}.pkl')
            if Args.use_ema:
                logger.torch_save(model, f'checkpoints/model_no_ema_{epoch:04}.pkl')

        if epoch % Args.render_interval == 0 or epoch == 1:
            with TimedAction('sampling'):
                sample(val_dataset, ema if Args.use_ema else model, epoch)


if __name__ == '__main__':
    main()
