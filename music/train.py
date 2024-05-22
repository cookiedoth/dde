import os

import torch
import torch.nn.functional as F
import torchinfo
from einops import rearrange, reduce
from ml_logger import logger
from params_proto import ParamsProto
from torch.utils.data import DataLoader
from multi_source.main.data import MultiSourceDataset
from single_instrument.dataset import InstrumentDataset
from single_instrument.model import Model, UNet1d
from single_instrument.utils import TimedAction, pad_dimensions


class Args(ParamsProto):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42
    n_epochs = 400
    batch_size = 8
    lr = 1e-4
    beta1 = 0.9
    beta2 = 0.99
    checkpoint_interval = 20
    patch_size = 262144


def loss_fn(model, x_0):
    batch, device = x_0.shape[0], x_0.device
    sigmas = model.sigma_distribution(num_samples=batch, device=device)
    sigmas_padded = pad_dimensions(sigmas, x_0)

    noise = torch.randn_like(x_0)
    x_noisy = x_0 + sigmas_padded * noise
    x_denoised = model.denoise_fn(x_noisy, sigmas=sigmas)

    losses = F.mse_loss(x_denoised, x_0, reduction="none")
    losses = reduce(losses, "b ... -> b", "mean")
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
    logger.log_text(chart_string, '.charts.yml', dedent=True, overwrite=True)


def main(**deps):
    Args._update(deps)
    setup_logger()
    logger.log_params(Args=vars(Args))

    with TimedAction('initialize_model'):
        model = Model(net=UNet1d(
            in_channels=4,
            channels=256,
            patch_factor=16,
            patch_blocks=1,
            resnet_groups=8,
            kernel_multiplier_downsample=2,
            kernel_sizes_init=[1, 3, 7],
            multipliers=[1, 2, 4, 4, 4, 4, 4],
            factors=[4, 4, 4, 2, 2, 2],
            num_blocks=[2, 2, 2, 2, 2, 2],
            attentions=[False, False, False, True, True, True],
            attention_heads=8,
            attention_features=128,
            attention_multiplier=2,
            use_nearest_upsample=False,
            use_skip_scale=True,
            use_attention_bottleneck=True,
            use_context_time=True,
            out_channels=4
        )).to(Args.device)
    model_stats = torchinfo.summary(model, verbose=0)
    logger.print(str(model_stats))

    with TimedAction('load_dataset'):
        dataset_root = os.getenv('DATASET_ROOT')
        dataset_path = os.path.join(dataset_root, 'audio/slakh2100/train')
        dataset = MultiSourceDataset(
            sr=22050,
            channels=1,
            min_duration=Args.patch_size / 22050.0 + 0.5,
            max_duration=640.0,
            aug_shift=True,
            sample_length=Args.patch_size,
            audio_files_dir=dataset_path,
            stems=['bass', 'drums', 'guitar', 'piano'])

    logger.print("Dataset size is: " + str(len(dataset)))

    data_loader = DataLoader(dataset, batch_size=Args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=Args.lr,
        betas=(Args.beta1, Args.beta2),
    )
    logger.start('log_timer')
    for epoch in range(1, Args.n_epochs + 1):
        logger.split('log_timer')

        timer_steps = 0
        for x in data_loader:
            x = x.to(Args.device)
            loss = loss_fn(model, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.store_metrics(loss=loss.item())
            logger.print(loss.item())
            timer_steps += 1

        logger.store_metrics({
            'steps_per_sec': timer_steps / logger.split('log_timer')
        })
        logger.log_metrics_summary(key_values={'epoch': epoch})

        if epoch % Args.checkpoint_interval == 0:
            logger.print(f'Saving the checkpoint at epoch {epoch}', color='yellow')
            logger.torch_save(model, 'checkpoints/model' + str(epoch) + '.pkl')


if __name__ == '__main__':
    main()
