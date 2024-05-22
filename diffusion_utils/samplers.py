import torch
from tqdm import tqdm


# Adapted code from https://github.com/NVlabs/edm
def heun_sampler(model, sigmas, noises, num_resamples, s_churn, device):
    sigmas = sigmas.to(device)
    x = sigmas[0] * noises
    gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1)
    with torch.no_grad():
        for i in tqdm(range(len(sigmas) - 1)):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            for r in range(num_resamples):
                sigma_hat = sigma * (gamma + 1) if 0.01 <= sigmas[i] <= 1.0 else sigma
                x_hat = x + torch.randn_like(x) * (sigma_hat ** 2 - sigma ** 2) ** 0.5

                # Euler step.
                d = model(x_hat, torch.ones(x.shape[0], device=device) * sigma_hat)
                x = x_hat + d * (sigma_next - sigma_hat)

                # Apply 2nd order correction.
                if i < len(sigmas) - 2:
                    d_prime = model(x, torch.ones(x.shape[0], device=device) * sigma_next)
                    x = x_hat + (sigma_next - sigma_hat) * (0.5 * d + 0.5 * d_prime)

                if r < num_resamples - 1:
                    x = x + torch.randn_like(x) * (sigma ** 2 - sigma_next ** 2) ** 0.5

    return x


def edm_ode_step(x, sigma, dt, eps):
    return x - dt * eps


def edm_sde_step(x, sigma, dt, eps):
    return x - 2 * dt * eps + torch.sqrt(2 * sigma * dt) * torch.randn_like(x)


# model predicts eps
@torch.no_grad()
def edm_diffusion_sampler(model, sigmas, noises, step):
    x = sigmas[0] * noises
    for i in tqdm(range(len(sigmas) - 1)):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        eps = model(x, torch.ones(x.shape[0], device=x.device) * sigma)
        dt = sigma - sigma_next
        x = step(x, sigma, dt, eps)
    return x
