import os

import torch
import torch.nn.functional as F
import torchinfo
from einops import reduce
from ema_pytorch import EMA
from ml_logger import logger
from params_proto import ParamsProto
from torch.utils.data import DataLoader

from diffusion_2d.loader import load_model
from music.data import MultiSourceDataset
from music.model import Model, RNNDevil, RNNOverlapDevil, TransformerDevil
from music.music_utils import TimedAction, pad_dimensions, patch_model


class Args(ParamsProto):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diff_model_path = '-'
    seed = 25
    n_epochs = 30
    batch_size = 16 if torch.cuda.is_available() else 1
    lr = 5e-5
    beta1 = 0.9
    beta2 = 0.99
    checkpoint_interval = 1
    min_clips = 4
    max_clips = 4
    clip_length = 12
    sample_rate = 22050
    model_type = "RNNDevil"
    patch_size = 262144
    n = 4
    max_train_patches = 5
    max_test_patches = 13
    use_ema = True


def loss_fn(model, x_0):
    batch, device = x_0.shape[0], x_0.device
    sigmas = model.sigma_distribution(num_samples=batch, device=device)
    sigmas_padded = pad_dimensions(sigmas, x_0)

    noise = torch.randn_like(x_0)
    x_noisy = x_0 + sigmas_padded * noise
    x_denoised = model.denoise_fn(x_noisy, sigmas)

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

    with (TimedAction('initialize_model')):
        diff_model = load_model(Args.diff_model_path, "checkpoints/model180.pkl").to(Args.device)
        patch_model(diff_model)
        diff_model.eval()
        model = Model(net=RNNDevil(Args.patch_size, Args.n, diff_model, Args.device)) if Args.model_type == "RNNDevil" \
            else Model(net=RNNOverlapDevil(diff_model, Args.device, n=Args.n, patch_size=Args.patch_size,
                                           overlap_size=Args.patch_size // 4)) if Args.model_type == "RNNOverlapDevil" \
            else Model(net=TransformerDevil(diff_model, Args.patch_size, Args.patch_size // 4 * 3,
                                            Args.device)) if Args.model_type == "TransformerDevil" \
            else None
        if model is None:
            raise ValueError('Unknown model type')
        model = model.to(Args.device)
    if Args.use_ema:
        ema = EMA(model,
                  beta=0.9999,
                  update_after_step=100,
                  update_every=10)

    model_stats = torchinfo.summary(model, verbose=0)
    logger.print(str(model_stats))

    with (TimedAction('load_dataset')):
        dataset_root = os.getenv('DATASET_ROOT')
        dataset_path = os.path.join(dataset_root, 'audio/slakh2100/train')
        dataset = MultiSourceDataset(
            sr=22050,
            channels=1,
            min_duration=48.0,
            max_duration=640.0,
            aug_shift=True,
            sample_length=Args.patch_size * Args.max_clips,
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
            if Args.model_type == "RNNDevil":
                num_patches = torch.randint(1, 5, (1,)).item()
                x_len = Args.patch_size * num_patches
                x = x[:, :, :x_len]
            elif Args.model_type == "RNNOverlapDevil" or Args.model_type == "TransformerDevil":
                num_patches = torch.randint(1, (Args.max_train_patches + 1), (1,)).item()
                stride = Args.patch_size // 4 * 3
                x_len = Args.patch_size + stride * (num_patches - 1)
                x = x[:, :, :x_len]
            loss = loss_fn(model, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.store_metrics(loss=loss.item())
            logger.print(loss.item())
            timer_steps += 1
            if Args.use_ema:
                ema.update()

        logger.store_metrics({
            'steps_per_sec': timer_steps / logger.split('log_timer')
        })
        logger.log_metrics_summary(key_values={'epoch': epoch})

        if epoch % Args.checkpoint_interval == 0:
            logger.print(f'Saving the checkpoint at epoch {epoch}', color='yellow')
            logger.torch_save(ema if Args.use_ema else model, f"checkpoints/model{epoch}.pkl")


if __name__ == '__main__':
    main()
