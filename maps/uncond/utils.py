import shutil
from ml_logger import logger
import torch
from audio_diffusion_pytorch import KarrasSchedule
from diffusion_2d.samplers import heun_sampler
import random
import string
import os
from PIL import Image
from cleanfid import fid
import numpy as np
from tqdm import tqdm


def tensor_to_image(tensor):
    """tensor is of shape [3, H, W], in [-1, 1] range
    returns numpy suitable for plt.imshow
    """
    np_image = tensor.cpu().detach().numpy().transpose(1, 2, 0)
    np_image = (np_image + 1) / 2.0
    return np_image


def random_str(k):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=k))


def get_path(loc):
    return os.path.join(os.environ['DATASET_ROOT'], 'img_tmp', loc)


def get_path_stem(loc, stem):
    return os.path.join(os.environ['DATASET_ROOT'], 'img_tmp', loc, stem)


def save_images(batch, loc, stem, start_from=0):
    path = get_path_stem(loc, stem)
    os.makedirs(path, exist_ok=True)
    for i in tqdm(range(batch.shape[0])):
        img8 = (tensor_to_image(batch[i]) * 255).astype(np.uint8)
        Image.fromarray(img8, 'RGB').save(f'{path}/{i+start_from}.png')


def find_fid(real, samples, cleanup=False):
    loc = random_str(10)
    logger.print('Calc FID, samples are saved at', get_path(loc))
    save_images(real, loc, 'real')
    save_images(samples, loc, 'samples')
    result = fid.compute_fid(get_path_stem(loc, 'real'), get_path_stem(loc, 'samples'))
    if cleanup:
        shutil.rmtree(get_path(loc))
    return result


def find_fid_loc(loc, cleanup=False):
    result = fid.compute_fid(get_path_stem(loc, 'real'), get_path_stem(loc, 'samples'))
    if cleanup:
        shutil.rmtree(get_path(loc))
    return result


def sample_images(model, sample_cnt, args):
    return heun_sampler(model,
                        sigmas=KarrasSchedule(sigma_min=args.sigma_min, sigma_max=args.sigma_max, rho=args.rho)(args.num_timesteps, args.device),
                        noises=torch.randn((sample_cnt, 3, args.image_size, args.image_size), device=args.device),
                        s_churn=args.s_churn,
                        num_resamples=1,
                        device=args.device)


def sample_images_batched(model, sample_cnt, args):
    samples = []
    while sample_cnt > 0:
        logger.print(f'Sampling, {sample_cnt} remaining')
        batch_size = min(sample_cnt, args.batch_size)
        samples.append(sample_images(model, batch_size, args))
        sample_cnt -= batch_size
    return torch.cat(samples, dim=0)


class TimedAction:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        logger.start(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        t = logger.split(self.name)
        logger.print(f'Action {self.name} took {t:.3}s')


def pad_dimensions(A, B):
    """given 1-dim tensor A, we append dimensions of size 1 
    so the number of dimensions matches B
    """
    return A.view([-1] + [1] * (len(B.shape) - 1))


def reduce_dim_except_0(A, f):
    return f(A, dim=tuple(range(1, len(A.shape))))


def fid_on_prefixes(real, sample_list, sizes):
    filenames = []
    for path in sample_list:
        for img in os.listdir(path):
            filenames.append(os.path.join(path, img))
    H = random_str(10)
    sample_path = get_path(H)
    for L in sizes:
        logger.print(f'Batch size {L}')
        start = 0
        fid_list = []
        while start + L <= len(filenames):
            logger.print(f'FID between {start} and {start+L}')
            os.makedirs(sample_path, exist_ok=True)
            for i in range(start, start + L):
                shutil.copyfile(filenames[i], os.path.join(sample_path, f'{i}.png'))
            res = fid.compute_fid(real, dataset_name='sat128', mode='clean', dataset_split='custom')
            fid_list.append(res)
            shutil.rmtree(sample_path)
            start += L
        logger.print('fid_list', fid_list)
        logger.print('avg fid', sum(fid_list) / len(fid_list))
        logger.print('min fid', min(fid_list))
        logger.print('max fid', max(fid_list))
