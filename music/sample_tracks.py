import argparse
import os

import torch
import torchaudio
from audio_diffusion_pytorch import KarrasSchedule

from diffusion_2d.loader import load_model
from diffusion_2d.samplers import heun_sampler
from music.music_utils import patch_model

SAMPLE_RATE = 22050
ROOT_PATH = ""
STEMS = ['bass', 'drums', 'guitar', 'piano']


def concat_model(PATCH_SIZE, num_patches, model, x, t):
    outs = []
    for i in range(num_patches):
        start = i * PATCH_SIZE
        out = model(x[:, :, start: start + PATCH_SIZE], t)
        outs.append(out)
    return torch.cat(outs, dim=2)


def multi_model(PATCH_SIZE, PATCH_OVERLAP, num_patches, w, model, x, t):
    outs = []
    for i in range(num_patches):
        start = i * (PATCH_SIZE - PATCH_OVERLAP)
        out = model(x[:, :, start: start + PATCH_SIZE], t)
        if i > 0:
            outs[-1][:, :, -PATCH_OVERLAP:] = outs[-1][:, :, -PATCH_OVERLAP:] * w + out[:, :, :PATCH_OVERLAP] * (1 - w)
            out = out[:, :, PATCH_OVERLAP:]
        outs.append(out)
    return torch.cat(outs, dim=2)


def sample_concat(PATCH_SIZE, num_patches, num_channels, batch_size, model, num_timesteps, device):
    return heun_sampler(lambda x, t: concat_model(PATCH_SIZE, num_patches, model, x, t),
                        sigmas=KarrasSchedule(sigma_min=1e-4, sigma_max=20.0, rho=7)(num_timesteps,
                                                                                     device),
                        noises=torch.randn(batch_size, num_channels, num_patches * PATCH_SIZE).to(device),
                        s_churn=20,
                        num_resamples=1,
                        device=device)


def sample_multi(PATCH_SIZE, PATCH_OVERLAP, num_patches, num_channels, w, batch_size, model, num_timesteps, device):
    return heun_sampler(lambda x, t: multi_model(PATCH_SIZE, PATCH_OVERLAP, num_patches, w, model, x, t),
                        sigmas=KarrasSchedule(sigma_min=1e-4, sigma_max=20.0, rho=7)(num_timesteps,
                                                                                     device),
                        noises=torch.randn(batch_size, num_channels,
                                           PATCH_SIZE * num_patches - PATCH_OVERLAP * (num_patches - 1)).to(device),
                        s_churn=20,
                        num_resamples=1,
                        device=device)


def sample_rnn(PATCH_SIZE, num_patches, num_channels, batch_size, model, num_timesteps, device):
    return heun_sampler(model,
                        sigmas=KarrasSchedule(sigma_min=1e-4, sigma_max=20.0, rho=7)(num_timesteps,
                                                                                     device),
                        noises=torch.randn(batch_size, num_channels, num_patches * PATCH_SIZE).to(device),
                        s_churn=20,
                        num_resamples=1,
                        device=device)


def sample_fn(PATCH_SIZE, num_channels, batch_size, model, num_timesteps, device):
    return heun_sampler(model,
                        sigmas=KarrasSchedule(sigma_min=1e-4, sigma_max=20.0, rho=7)(num_timesteps,
                                                                                     device),
                        noises=torch.randn(batch_size, num_channels, PATCH_SIZE).to(device),
                        s_churn=20,
                        num_resamples=1,
                        device=device)


def main(Args):
    patch_size = args.patch_size
    patch_overlap = patch_size // 4
    target_dir = ROOT_PATH / Args.save_path
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        for file in os.listdir(target_dir):
            file_path = os.path.join(target_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
    device = torch.device(Args.device)
    model = load_model(Args.ckpt_path, "checkpoints/" + Args.checkpoint).to(device)
    patch_model(model)
    model.eval()
    for batch_start in range(0, Args.total_samples, Args.batch_size):
        generated_tracks = torch.randn((Args.batch_size, 1, patch_size * Args.num_patches), device=device) if Args.noise \
            else sample_concat(patch_size, Args.num_patches, Args.num_channels, Args.batch_size, model,
                               Args.num_timesteps, device) if Args.concat \
            else sample_multi(patch_size, patch_overlap, Args.num_patches, Args.num_channels, Args.w, Args.batch_size,
                              model, Args.num_timesteps, device) if Args.multi \
            else sample_rnn(patch_size, Args.num_patches, Args.num_channels, Args.batch_size, model, Args.num_timesteps,
                            device) if Args.rnn \
            else sample_fn(patch_size, Args.num_channels, Args.batch_size, model, Args.num_timesteps, device)

        if Args.separate:
            for i, track in enumerate(generated_tracks):
                track_index = batch_start + i
                track_save_dir = target_dir / f"track_{track_index}"
                if not os.path.exists(track_save_dir):
                    os.makedirs(track_save_dir)
                for j in range(Args.num_channels):
                    track_save_path = track_save_dir / f"{STEMS[j]}.wav"
                    track_save_path.parent.mkdir(exist_ok=True, parents=True)
                    torchaudio.save(str(track_save_path), torch.tensor(track.cpu().numpy()), SAMPLE_RATE)
                    print(f"Saved {track_save_path}")
        else:
            if Args.num_channels > 1:
                generated_tracks = torch.sum(generated_tracks, dim=1, keepdim=True)

            for i, track in enumerate(generated_tracks):
                track_index = batch_start + i
                track_save_path = target_dir / f"track_{track_index}.wav"
                track_save_path.parent.mkdir(exist_ok=True, parents=True)
                torchaudio.save(str(track_save_path), torch.tensor(track.cpu().numpy()), SAMPLE_RATE)
                print(f"Saved {track_save_path}")

    print("All tracks have been generated and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process save path and number of instruments.')
    parser.add_argument('ckpt_path', type=str, help='Path to the checkpoint from jaynes exp')
    parser.add_argument('save_path', type=str, help='Directory in - to save model')

    parser.add_argument('--total_samples', type=int, default=128,
                        help='Total number of samples to generate (default: 128)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for sampling (default: 16)')
    parser.add_argument('--step_type', type=str, default="sde",
                        help='Type of step to use for sampling (default: "sde")')
    parser.add_argument('--num_timesteps', type=int, default=150, help='Number of timesteps to sample (default: 400)')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to run sampling on (default: "cuda:0")')
    parser.add_argument('--concat', action='store_true', help='Perform concatenation (default: False)')
    parser.add_argument('--multi', action='store_true', help='Perform multi-diffusion concat (default: False)')
    parser.add_argument('--noise', action='store_true', help='Sample songs out of noise (default: False)')
    parser.add_argument('--rnn', action='store_true', help='Sample from the RNN devil model (default: False)')
    parser.add_argument('--separate', action='store_true', help='Store stems separately (default: False)')
    parser.add_argument('--num_patches', type=int, default=2, help='Number of patches to merge (default: 2)')
    parser.add_argument('--num_channels', type=int, default=1, help='Number of channels generated (default: 1)')
    parser.add_argument('--patch_size', type=int, default=262144, help='Patch size for sampling (default: 262144)')
    parser.add_argument('--w', type=float, default=0.5, help='Weight for merging patches (default: 0.5)')
    parser.add_argument('--checkpoint', type=str, default="model.pkl", help='Checkpoint to use')
    args = parser.parse_args()
    main(args)
