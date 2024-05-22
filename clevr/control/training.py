import random

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchinfo
from einops import reduce
from ml_logger import logger
from params_proto import ParamsProto
from params_proto import Proto
from torch.optim import Adam
from torch.utils.data import DataLoader

from diffusion_2d.loader import load_model
from diffusion_2d.utils import sample_control_image
from diffusion_cubes.control.model import RNNDevil, TransformerDevil
from diffusion_cubes.loader import Clevr2DPosDataset
from diffusion_cubes.model import Model
from single_instrument.utils import pad_dimensions
from ema_pytorch import EMA


class Args(ParamsProto):
    cube_model_path = "/diffusion-comp/2024/04-27/diffusion_cubes/sweep/21.36.30/lr:0.0001"
    classifier_path = "/diffusion-comp/2023/11-23/diffusion_cubes/classifier/sweep/17.57.43/"
    dataset_path = Proto(env='$DATASET_ROOT/clevr/clevr_pos_data_128_30000.npz')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = 64
    seed = 25
    n_epochs = 25
    lr = 1e-4
    batch_size = 16
    checkpoint_interval = 5
    render_interval = 5
    sampler_batch_size = 9
    sampler_num_labels = 5
    num_timesteps = 50
    random_crop = False
    random_flip = False
    max_train_labels = 2
    use_ema = True
    w = 20.0


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
         glob: samples/5.png
       - type: image
         glob: samples/10.png
       - type: image
         glob: samples/15.png
       - type: image
         glob: samples/20.png
       """, ".charts.yml", dedent=True, overwrite=True)

    torch.manual_seed(Args.seed)
    logger.log_params(Args=vars(Args))
    cube_model = load_model(Args.cube_model_path, "checkpoints/model.pkl").to(Args.device)
    for param in cube_model.parameters():
        param.requires_grad = False
    cube_model.eval()
    classifier_model = load_model(Args.classifier_path, "checkpoints/model_best.pkl").to(Args.device)
    for param in classifier_model.parameters():
        param.requires_grad = False
    classifier_model.eval()
    control_model = Model(net=TransformerDevil(cube_model=cube_model,
                                               image_size=Args.image_size,
                                               device=Args.device)).to(Args.device)

    if Args.use_ema:
        ema = EMA(control_model,
                  beta=0.9999,
                  update_after_step=100,
                  update_every=10)

    model_stats = torchinfo.summary(control_model, verbose=0)
    logger.print(str(model_stats))

    optimizer = Adam(control_model.parameters(), lr=Args.lr)

    logger.start('log_timer')
    dataset = Clevr2DPosDataset(
        data_path=Args.dataset_path,
        resolution=Args.image_size,
        random_crop=Args.random_crop,
        random_flip=Args.random_flip
    )

    logger.print("Dataset size is: " + str(len(dataset)))

    data_loader = DataLoader(dataset, batch_size=Args.batch_size, shuffle=True, num_workers=4)

    for epoch in range(0, Args.n_epochs + 1):
        if epoch > 0 and epoch % Args.render_interval == 0:
            num_labels = torch.randint(1, Args.sampler_num_labels + 1, (1,)).item()
            coords = torch.zeros((Args.sampler_batch_size, num_labels, 2), device=Args.device)
            for i in range(Args.sampler_batch_size):
                while True:
                    coords[i] = torch.rand((num_labels, 2)).to(Args.device) * 0.4 + 0.3
                    bad = False
                    for j in range(num_labels):
                        for k in range(j):
                            if ((coords[i][j][0] - coords[i][k][0]) ** 2 + (
                                    coords[i][j][1] - coords[i][k][1]) ** 2) ** 0.5 < 0.15:
                                bad = True
                                break
                        if bad:
                            break
                    if not bad:
                        break

            logger.print('rendering diffusion steps...', color="yellow")
            images = sample_control_image(control_model, coords,
                                          args=Args,
                                          batch_size=Args.sampler_batch_size,
                                          num_labels=num_labels)
            logger.print('saving the samples...', color="green")
            coords = coords.cpu().detach().numpy()
            fig, axs = plt.subplots(3, 3, figsize=(8, 8))
            for i, image in enumerate(images):
                np_image = image.cpu().detach().numpy().transpose(1, 2, 0)
                np_image = (np_image + 1) / 2.0
                row = i // 3
                col = i % 3
                axs[row, col].imshow(np_image, interpolation='nearest')
                for j in range(num_labels):
                    axs[row, col].scatter(coords[i][j][0] * Args.image_size,
                                          (1 - coords[i][j][1]) * Args.image_size,
                                          color='red',
                                          marker='x',
                                          s=80)
                axs[row, col].axis('off')
            plt.tight_layout()
            plt.show()
            logger.savefig(f"samples/{epoch}.png")

        if Args.checkpoint_interval and epoch > 0 and epoch % Args.checkpoint_interval == 0:
            logger.print(f"Saving the checkpoint at epoch {epoch}", color="yellow")
            logger.torch_save(ema if Args.use_ema else control_model, f"checkpoints/model{epoch}.pkl")

        if epoch == Args.n_epochs:
            break

        timer_steps = 0
        for x_0, _, _ in data_loader:
            x_0 = x_0.to(Args.device).float()
            probs = torch.sigmoid(classifier_model((x_0 + 1) / 2))
            p_mask = probs > 0.5
            batch_coords = []
            batch_len = x_0.shape[0]
            for i in range(batch_len):
                (numLabels, _, _, centroids) = cv2.connectedComponentsWithStats(
                    p_mask[i].cpu().detach().numpy().astype('uint8'))
                numLabels -= 1
                centroids = centroids[1:]
                x = torch.from_numpy(centroids[:, 0]).to(Args.device) / Args.image_size
                y = torch.from_numpy(centroids[:, 1]).to(Args.device) / Args.image_size
                coords = torch.stack([y, 1 - x], dim=1).to(Args.device)
                p = torch.randperm(coords.shape[0]).to(Args.device)
                coords = coords[p]
                batch_coords.append(coords)

            nx_0 = torch.zeros_like(x_0).to(Args.device)
            train_len = torch.randint(1, Args.max_train_labels + 1, (1,)).item()
            y = torch.zeros((batch_len, train_len, 2), device=Args.device)
            mask = torch.zeros((batch_len, train_len), device=Args.device, dtype=torch.bool)
            ptr = 0
            for i in range(batch_len):
                cnt = min(batch_coords[i].shape[0], train_len)
                nx_0[ptr] = x_0[i]
                y[ptr, :cnt] = batch_coords[i][:cnt].float()
                mask[ptr, :cnt] = torch.rand((cnt,)).to(Args.device) > 0.1
                ptr += 1
            if ptr == 0:
                continue
            nx_0 = nx_0[:ptr]
            y = y[:ptr]
            mask = mask[:ptr]
            loss = loss_fn(control_model, nx_0, y, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.print(loss.cpu().item())
            logger.store_metrics({'loss': loss.cpu().item()})
            timer_steps += 1
            if Args.use_ema:
                ema.update()

        logger.store_metrics({
            'steps_per_sec': timer_steps / logger.split('log_timer')
        })
        logger.log_metrics_summary(key_values={"epoch": epoch})


if __name__ == '__main__':
    main()
