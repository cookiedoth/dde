import torch
from audio_diffusion_pytorch import KarrasSchedule
from ml_logger import logger

from diffusion_utils.samplers import edm_ode_step, heun_sampler, edm_diffusion_sampler, edm_sde_step


def unlift(tensor):
    return tensor.detach().cpu().numpy()


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

