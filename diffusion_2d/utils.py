import matplotlib.pyplot as plt
import numpy as np
from diffusion_2d.samplers import edm_ode_step, pc_sampler, hmc_sampler, get_step_by_name, diffusion_sampler, heun_sampler, \
        edm_diffusion_sampler, edm_langevin_sampler, edm_sde_step
from ml_logger import logger
from audio_diffusion_pytorch import KarrasSchedule
import torch


def unlift(tensor):
    return tensor.detach().cpu().numpy()


def plot_samples(history, name):
    for t, frame in enumerate(history):
        plt.scatter(*(frame.cpu().detach().numpy()).T, s=2)

    centers = [(-0.75, 0.5), (0.75, 0.5), (0, -0.799)]
    radius = 1
    for center in centers:
        circle = patches.Circle(center, radius, fc='none', ec='blue')
        plt.gca().add_patch(circle)

    plt.gca().set_aspect('equal')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    logger.savefig(f"{name}.png")


class VariancePreservingParams:
    def __init__(self, beta_min, beta_max, device):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.device = device

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def alpha_bar(self, t):
        return torch.exp(-self.beta_min * t - (self.beta_max - self.beta_min) * t * t / 2)

    def sample_xt_x0(self, x_0, eps, t):
        alpha_bar = self.alpha_bar(t).view([-1] + [1] * (len(x_0.shape) - 1))
        return x_0 * torch.sqrt(alpha_bar) + eps * torch.sqrt(1.0 - alpha_bar)

    def f(self, x, t):
        return -0.5 * self.beta(t).view([-1] + [1] * (len(x.shape) - 1)) * x

    def g(self, t):
        return torch.sqrt(self.beta(t))

    def score_scale(self, t):
        return -1.0 / torch.sqrt(1 - self.alpha_bar(t))

    def gamma(self, t):
        return torch.ones_like(t, device=self.device)


def sample(model, params, args, sampling_method='hmc', dims=(2,)):
    if sampling_method == 'hmc':
        return hmc_sampler(
            model,
            params,
            dims,
            step=get_step_by_name(args.step_type),
            num_timesteps=args.num_timesteps,
            batch_size=args.sampler_batch_size,
            ha_steps=args.ha_steps,
            damping_coef=args.damping_coeff,
            num_leapfrog_steps=args.num_leapfrog_steps,
            return_trajectory=False,
            device=args.device)
    elif sampling_method == 'pc':
        return pc_sampler(
            model,
            params,
            dims,
            step=get_step_by_name(args.step_type),
            num_steps_per_time_step=0,
            device=args.device,
            return_trajectory=False,
            batch_size=args.sampler_batch_size,
            num_timesteps=args.num_timesteps,
            do_metropolis_hastings=False)
    else:
        raise ValueError('No sampling method', sampling_method)


def sample_control(model, ys, params, args):
    ys_tensors = [torch.tensor([y] * args.sampler_batch_size, device=args.device) for y in ys]
    return sample(lambda x, t, mode: model(x, t, ys_tensors, mode), params, args, sampling_method='pc')


def sample_embed(model, coords, params, args, batch_size=32):
    mask = torch.tensor([True] * batch_size, device=args.device)
    return diffusion_sampler(
        lambda x, t, mode: model(x, t, coords, mask, mode, True),
        params,
        dims=(args.emb_dim,),
        step=get_step_by_name(args.step_type),
        device=args.device,
        batch_size=batch_size,
        num_timesteps=args.num_timesteps)


def get_step(args):
    if args.step == 'sde':
        return edm_sde_step
    elif args.step == 'ode':
        return edm_ode_step
    else:
        raise ValueError('Invalid step')


def sample_image_base(cfg_model, args, batch_size):
    sigmas = KarrasSchedule(sigma_min=1e-4, sigma_max=80.0, rho=7)(args.num_timesteps, args.device)
    noises = torch.randn(batch_size, 3, args.image_size, args.image_size).to(args.device)
    if 'sampler' not in dir(args):
        logger.log('Sampler is not specified, using heun by default!')
    if 'sampler' not in dir(args) or args.sampler == 'heun':
        return heun_sampler(cfg_model,
                    sigmas=sigmas,
                    noises=noises,
                    s_churn=0,
                    num_resamples=1,
                    device=args.device)
    elif args.sampler == 'diffusion':
        return edm_diffusion_sampler(cfg_model, sigmas=sigmas, noises=noises, step=get_step(args))
    elif args.sampler == 'langevin':
        return edm_langevin_sampler(cfg_model, sigmas=sigmas, noises=noises, step=get_step(args), lsteps=args.lsteps)
    else:
        raise ValueError('Unknown sampler')


def sample_image(model, coords, args, batch_size=32):
    w = args.w
    mask = torch.tensor([True] * batch_size, device=args.device)
    mask0 = torch.tensor([False] * batch_size, device=args.device)
    cfg_model = lambda x, t: (1 + w) * model(x, t, coords, mask) - w * model(x, t, coords, mask0)
    return sample_image_base(cfg_model, args, batch_size)


def sample_image_unconditioned(model, args, batch_size=32):
    y = torch.zeros((batch_size, 2), device=args.device)
    mask = torch.tensor([False] * batch_size, device=args.device)
    uncond_model = lambda x, t: model(x, t, y, mask)
    return sample_image_base(uncond_model, args, batch_size)


def sample_control_image(model, y, args, batch_size=4, num_labels=2):
    w = args.w
    mask = torch.ones((batch_size, y.shape[1]), device=args.device, dtype=torch.bool)
    mask[:, :num_labels] = True
    cfg_model = lambda x, t: (1 + w) * model(x, t, y, mask) - w * model(x, t, y, ~mask)
    return sample_image_base(cfg_model, args, batch_size)


def create_batch(box, gaps, inclusive, device):
    x_lin = np.linspace(box[0], box[2], gaps + 1)
    y_lin = np.linspace(box[1], box[3], gaps + 1)
    if not inclusive:
        x_lin = x_lin[1:gaps]
        y_lin = y_lin[1:gaps]
    x, y = np.meshgrid(x_lin, y_lin)
    result = np.column_stack((x.ravel(), y.ravel()))
    return torch.tensor(result).to(device).float()
