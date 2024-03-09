import torch 
import numpy as np

class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps: int=1000, beta_start: float=0.00085, beta_end: float=0.0120):
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32)**2
        self.alphas = 1 - self.betas
        self.alphas_cumprod  = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator

        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.array(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // num_inference_steps
        time_steps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(time_steps)

    def _get_previous_step(self, current_timestep):
        return current_timestep - (self.num_training_steps // self.num_inference_steps)

    def _get_variance(self, timestep: int):
        t = timestep
        prev_t = self._get_previous_step(t)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        variance = (1-alpha_prod_t_prev) / (1-alpha_prod_t) * current_beta_t

        variance = torch.clamp(variance, min=1e-20)

        return variance
        

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_step(t)

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev

        # 2. compute predicted original sample from predicted noise also called
        pred_original_sample = (latents - alpha_prod_t**0.5 * model_output) / alpha_prod_t**0.5

        # Compute coefficients for original sample in current sample
        pred_original_sample_coef = (1-alpha_prod_t)**0.5 / (alpha_prod_t-1)
        current_sample_coef = (current_alpha_t**0.5*(1-alpha_prod_t_prev)) / (1-alpha_prod_t)

        # Compute previous sample mean
        pred_prev_sample = pred_original_sample_coef * pred_original_sample + current_sample_coef * latents

        variance = 0
        if t > 0:
            device = model_output.device 
            noise = torch.rand(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t)**0.5)

        pred_prev_sample = pred_prev_sample + variance * noise 

        return pred_prev_sample

    def add_noise(self, original_sample: torch.Tensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alphas_cumprod.to(device=original_sample.device, dtype=original_sample.dtype)
        timesteps = timesteps.to(original_sample.device)

        sqrt_alpha_prod = alpha_cumprod[timesteps]**0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1-alpha_cumprod[timesteps])**0.5
        sqrt_one_minus_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = torch.rand(original_sample.shape, generator=self.generator, device=original_sample.device, dtype=original_sample.dtype)
        noisy_sample = (sqrt_alpha_prod*original_sample) + (sqrt_one_minus_alpha_prod*noise)

        return noisy_sample


        