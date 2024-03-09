import torch 
import numpy as np
import tqdm 
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8 
LATENTS_HEIGHT = HEIGHT // 8

def generate(
        prompt, 
        uncond_prompt,
        input_image,
        strength = 0.8,
        do_cfg = True,
        cfg_scale = 7.5,
        sampler_name="ddpm",
        n_inference_steps = 50,
        models = {},
        seed = None, 
        device = None,
        idle_device = None,
        tokenizer = None
):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be between 0 and 1")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator()
        if seed:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models['clip']
        clip.to(device)

        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding='max_length', max_length=77
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_tokens = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding='max_length', max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_tokens = clip(uncond_tokens)

            context = torch.concat([cond_tokens, uncond_tokens])
        else:
            con_tokens = tokenizer.batch_encode_plus(
                [prompt], padding='max_length', max_length=77
            ).input_ids
            con_tokens = torch.tensor(con_tokens, dtype=torch.long, device=device)

            context = clip(con_tokens)

        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models['encoder']
            encoder.to(device)

            # (Height, Width, Channel)
            input_image = input_image.resize(HEIGHT, WIDTH)
            # (Height, Width, Channel)
            input_image = np.array(input_image)
            # (Height, Width, Channel)
            input_image = torch.Tensor(input_image, dtype=torch.float32, device=device)
            # (Height, Width, Channel)
            input_image = rescale(input_image, (0, 255), (0, 1))
            # (Batch, Height, Width, Channel)
            input_image = input_image.unsqueeze(0)
            # (Batch, Channel, Height, Width)
            input_image = input_image.permute(0, 3, 1, 2)

            # (Batch, 4, Lantent_Height, Lantent_Width)
            encoder_noise = torch.rand(latents_shape, generator=generator, device=device)

            # (Batch, 4, Lantent_Height, Lantent_Width)
            latents = encoder(input_image, encoder_noise)

            # (Batch, 4, Lantent_Height, Lantent_Width)
            sampler.set_strenth(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)

        else:
            latents  = torch.rand(latents_shape, generator=generator, device=device)

        diffusion = models['diffusion']
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_embedding(timestep)

            # model_input: (Batch, 4, Lantent_Height, Latent_Width)
            model_input = latents

            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
    
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)
        image = decoder(latents)
        to_idle(decoder)

        image = rescale(image, (-1, 1), (0, 255), clamp=True)
        # (Batch, Channel, Height, Width) -> (Batch, Height, Width, Channel)
        image = image.permute(0, 2, 4, 1)
        image = image.to("cpu", torch.uint8).numpy()
        return image[0]
    

def rescale(x, old_scale, new_scale, clamp=False):
    min_new_scale, max_new_scale = new_scale
    min_old_scale, max_old_scale = old_scale

    x -= min_old_scale
    x *= (max_new_scale - min_new_scale)/(max_old_scale - min_old_scale)
    x += min_new_scale

    if clamp:
        x = x.clamp(min_new_scale, max_new_scale)

    return x

def get_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

