import torch
import numpy 
from typing import Callable, Optional
from tqdm import tqdm
import json
import numpy as np



@torch.no_grad()
def karras(
    mixture: torch.Tensor,
    noises: torch.Tensor,
    denoise_fn: Callable,
    sigmas: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    cond_index: int = 0,
    s_churn: float = 40.0,  # > 0 to add randomness
    num_resamples: int = 2,
    use_tqdm: bool = False,
    gaussian: bool = False,
):
    # Set initial noise
    x = sigmas[0] * noises  # [num-sources, sample-length]

    for i in tqdm(range(len(sigmas) - 1), disable=not use_tqdm):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]

        for r in range(num_resamples):
            # Inject randomness
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            sigma_hat = sigma * (gamma + 1)
            x = x + torch.randn_like(x) * (sigma_hat**2 - sigma**2) ** 0.5

            # Compute conditioned derivative
            
            x[:1] = mixture - x[1:].sum(dim=0, keepdim=True)
            score = (x - denoise_fn(x, sigma=sigma)) / sigma
            
            ds = score[1:] - score[:1]

            # Update integral
            x[1:] += ds * (sigma_next - sigma_hat)

            # Renoise if not last resample step
            if r < num_resamples - 1:
                x = x + torch.sqrt(sigma**2 - sigma_next**2) * torch.randn_like(x)

    return x

                
@torch.no_grad()
def restart_sampler(
    denoise_fn, randn_like=torch.randn_like,
    num_steps=100, sigma_min=0.001, sigma_max=5, rho=9,
    S_churn=0, S_min=0, S_max=5, S_noise=0,
    restart_info="", restart_gamma=0, noises = None,
    mixture = None,
    restart = False,
):

    def get_steps(min_t, max_t, num_steps, rho):

         step_indices = torch.arange(num_steps, dtype=torch.float, device="cuda")
         t_steps = (max_t ** (1 / rho) + step_indices / (num_steps - 1) * (min_t ** (1 / rho) - max_t ** (1 / rho))) ** rho
         return t_steps


    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device="cuda")
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
    total_step = len(t_steps)
    
    #noises 
    x_next = noises * t_steps[0]
        # Main sampling loop.

    # {[num_steps, number of restart iteration (K), t_min, t_max], ... }
    import json
    restart_list = json.loads(restart_info) if restart_info != '' else {}
    # cast t_min to the index of nearest value in t_steps
    restart_list = {int(torch.argmin(abs(t_steps - v[2]), dim=0)): v for k, v in restart_list.items()}
    # dist.print0(f"restart configuration: {restart_list}")

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N_main -1
        x_cur = x_next
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        # Euler step.
        x_next[:1] = mixture -  x_next[1:].sum(dim=0, keepdim=True)
        denoised = denoise_fn(x_next,  sigma=t_cur)
        d_cur = (x_hat - denoised) / t_hat
        ds = d_cur[1:] - d_cur[:1]
        x_next[1:] = x_hat[1:] + ds * (t_next - t_hat)

        #x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            x_next[:1] = mixture -  x_next[1:].sum(dim=0, keepdim=True)
            denoised = denoise_fn(x_next,  sigma=t_cur)
            d_prime = (x_hat - denoised) / t_next
            ds_prime = d_prime[1:] - d_prime[:1]
            x_next[1:] = x_hat[1:] + (t_next - t_hat) * (0.5 * ds + 0.5 * ds_prime)

        # ================= restart ================== #
        if restart == True:
            if i + 1 in restart_list.keys():
                restart_idx = i + 1

                for restart_iter in range(restart_list[restart_idx][1]):

                    new_t_steps = get_steps(min_t=t_steps[restart_idx], max_t=restart_list[restart_idx][3],
                                            num_steps=restart_list[restart_idx][0], rho=rho)
                    
                    new_total_step = len(new_t_steps)
                    
                    x_next = x_next + randn_like(x_next) * (new_t_steps[0] ** 2 - new_t_steps[-1] ** 2).sqrt() * S_noise


                    for j, (t_cur, t_next) in enumerate(zip(new_t_steps[:-1], new_t_steps[1:])):  # 0, ..., N_restart -1

                        x_cur = x_next
                        gamma = restart_gamma if S_min <= t_cur <= S_max else 0
                        t_hat = t_cur + gamma * t_cur

                        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

                        x_next[:1] = mixture -  x_next[1:].sum(dim=0, keepdim=True)
                        denoised = denoise_fn(x_next,  sigma=t_cur)
                        d_cur = (x_hat - denoised) / t_hat
                        ds = d_cur[1:] - d_cur[:1]
                        x_next[1:] = x_hat[1:] + ds * (t_next - t_hat)

                

                        # Apply 2nd order correction.
                        if j < new_total_step - 2 or new_t_steps[-1] != 0:
                            x_next[:1] = mixture -  x_next[1:].sum(dim=0, keepdim=True)
                            denoised = denoise_fn(x_next,  sigma=t_cur)
                            d_prime = (x_hat - denoised) / t_next
                            ds_prime = d_prime[1:] - d_prime[:1]
                            x_next[1:] = x_hat[1:] + (t_next - t_hat) * (0.5 * ds + 0.5 * ds_prime)
                        

    return x_next
        
