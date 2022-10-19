from sched import scheduler
import torch

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL, 
    UNet2DConditionModel, 
    LMSDiscreteScheduler,
    DDIMScheduler)
from tqdm import tqdm
from torch import autocast, inference_mode
import PIL
from PIL import Image
from matplotlib import pyplot as plt
import numpy


from IPython.display import HTML
from base64 import b64encode

import torchvision.transforms as T

from transformers import logging

def jupyter_display_video(imgs, tmp_folder = "vid_tmp/", framerate = 4):

    import os 
    from IPython.display import HTML
    from base64 import b64encode
    from glob import glob
    import shutil
    os.makedirs(tmp_folder,exist_ok=True)
    for i, img in enumerate(imgs):
        img.save(f'{tmp_folder}{i:04}.jpeg')
    cmd_mk_vid = f"ffmpeg -v 1 -y -f image2 -framerate {framerate} -i vid_tmp/%04d.jpeg -c:v libx264 -preset slow -qp 18 -pix_fmt yuv420p out.mp4"
    os.system(cmd_mk_vid)

    shutil.rmtree(tmp_folder)
    mp4 = open('out.mp4','rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
    <video width=512 controls>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)

logging.set_verbosity_error()


def tensor_to_pil(tensor_imgs):
    tensor_imgs = (tensor_imgs / 2 + 0.5).clamp(0, 1)
    to_pil = T.ToPILImage()
    pil_imgs = [to_pil(img) for img in tensor_imgs]
    return pil_imgs


def pil_to_tensor(pil_imgs):
    to_torch = T.ToTensor()
    if type(pil_imgs) == PIL.Image.Image:
        tensor_imgs = to_torch(pil_imgs).unsqueeze(0) * 2 - 1
    elif type(pil_imgs) == list:
        tensor_imgs = torch.cat(
            [to_torch(pil_imgs).unsqueeze(0) * 2 - 1 for img in pil_imgs]
        )
    else:
        raise Exception("Input need to be PIL.Image or list of PIL.Image")
    return tensor_imgs


def image_grid(imgs, rows=1, cols=1):
# TODO implement this
# n = 10
# num_rows = 4
# num_col = n // num_rows
# num_col  = num_col + 1 if n % num_rows else num_col
    if type(imgs) == torch.Tensor:
        imgs = tensor_to_pil(imgs)
    assert len(imgs) >= rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


class StableDifussionWrapper:
    def __init__(self, 
                img_size = (512,512),
                scheduler_type = "lms",
                ):
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.height, self.width = img_size

        assert scheduler_type in ["lms","ddim"]
        self.scheduler_type = scheduler_type

        # Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=True
        ).to(self.device)

        # Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device)
        self.token_length = self.tokenizer.model_max_length

        # The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=True
        ).to(self.device)

        self.set_scheduler()
        # Prompt engeneering
        self.pe_words = [
            "digital art",
            "intricate",
            "highly detailed",
            "sharp focus",
            "photograph",
            "4k",
            "uhd",
            "Ultra High Definition",
            "8k",
        ]  # " "nikon d3300:25"]

    def set_scheduler(self, scheduler_type = None):
        if not scheduler_type is None:
            self.scheduler_type = scheduler_type
        
        if self.scheduler_type == "lms":
            # The noise scheduler
            self.scheduler = LMSDiscreteScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )
        elif self.scheduler_type == "ddim":
            self.scheduler = DDIMScheduler(
                beta_start=0.00085, beta_end=0.012,
                beta_schedule='scaled_linear', num_train_timesteps=1000)
        else:
            raise NotImplementedError
    def encode(self, imgs):
        # Single image -> single latent in a batch (so size 1, 4, 64, 64)
        with torch.no_grad():
            latents = self.vae.encode(imgs)
        return 0.18215 * latents.mode()  # or .mean or .sample

    def decode(self, latents):
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        return image

    def diffusion_loop(
        self,
        prompt="",
        latents=None,
        start_step=0,
        seed=None,
        num_inference_steps=30,
        guidance_scale=7.5,  # Scale for classifier-free guidance
        return_latents_t0_preds = True,
        ):
        
        # TODO Implement this 
        # import easydict
        # history = easydict.EasyDict()
        # history.z0_preds = []
        # history.zts = []

        if seed is None:
            seed = torch.randint(int(1e6), (1,))
        torch.manual_seed(seed)

        # Prep Scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Get Text Embedding
        text_embeddings = self.prep_text(prompt)

        if latents is None:
            latents = self.random_latents(seed=seed)
        
        if start_step > 0:        
            start_timestep = self.scheduler.timesteps[start_step].long()
            start_timesteps = start_timestep.repeat(latents.shape[0]).long()
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, start_timesteps)
        
        if self.scheduler_type == "lms":
            latents = latents * self.scheduler.sigmas[start_step]
    
        latents_t0_preds = []

        with autocast("cuda"), inference_mode():
            for i, t in tqdm(enumerate(self.scheduler.timesteps),leave = False):
                if i >= start_step:
                
                    #print(i,t)
                    # expand the latents if we are doing classifier-free
                    # guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents] * 2)
                    
                    if self.scheduler_type == "lms":
                        sigma = self.scheduler.sigmas[i]
                        latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

                    # predict the noise residual
                    with torch.no_grad():
                        noise_pred = self.unet(
                            latent_model_input,t,encoder_hidden_states=text_embeddings,
                        )["sample"]

                    # perform classifier-free guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    # compute the previous noisy sample x_t -> x_t-1
                    # Append latent history if return_x0_preds
                    if self.scheduler_type == "lms":
                        latents = self.scheduler.step(noise_pred, i, latents)["prev_sample"]
                        if return_latents_t0_preds:
                            latents_t0_pred = latents.detach().clone() - sigma * noise_pred
                            latents_t0_preds.append(latents_t0_pred.detach().clone())
                                                
                    if self.scheduler_type == "ddim":
                        latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
                        if return_latents_t0_preds:
                            alpha_prod_t = self.alphas_cumprod[t]
                            latents_t0_pred = (latents.detach().clone() - (1 - alpha_prod_t) ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                            latents_t0_preds.append(latents_t0_pred.detach().clone())
        if return_latents_t0_preds:
            return latents, torch.cat(latents_t0_preds)
        else:
            return latents

 
        
    def z_to_pil(self, z0, size = (512,512)):
        n,_,_,_ = z0.shape
        if n == 1:
            x0 = self.decode(z0)
            return tensor_to_pil(x0)[0].resize(size)
        else:
            return [tensor_to_pil(self.decode(z.unsqueeze(0)))[0].resize(size) for z in z0]



    def add_pe_words(self, prompt):
        return prompt + ", " + ", ".join(self.pe_words)

    def encode_text(self, prompt):
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.token_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_encoding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding

    def prep_text(self, prompt):

        # add unconditional embedding
        return torch.cat([self.encode_text(""), self.encode_text(prompt)])

    def random_latents(self, batch_size=1, seed=None):
        if seed is None:
            seed = torch.randint(int(1e6), (1,))
        # Prep latents  (Need to sigma scale to match k)
        return torch.randn(
            (batch_size, self.unet.in_channels, self.height // 8, self.width // 8),
            generator=torch.manual_seed(seed),
        ).to(self.device)

    def add_noise(self, latents, timestep=10):
        noise = torch.randn_like(latents)
        return self.scheduler.add_noise(latents, noise, timestep)

    def test(self):
        ## Tests
        prompt = "prompt"
        print(self.encode_text(prompt).shape)
        print(self.prep_text(prompt).shape)
        print(self.random_latents().shape)
        print(self.decode(self.random_latents()).shape)
        print(self.add_noise(self.random_latents()).shape)

    
    def lerp(self, z1,z2,t):
        return z1*(1-t)**0.5 + z2*t**0.5
        # return z1*(1-t) + z2*t

    
    def interpolation(self, z1, z2, numsteps = 5,t_min = 0, t_max = 1):
        return torch.cat([self.lerp(z1,z2,t) for t in torch.linspace(t_min,t_max,numsteps)])
    
    def generate_images(self, prompt,
                        n = 4,
                        seed=None, 
                        num_inference_steps=40,  # More steps better quality but slower, 30-40 is usually a good zone
                        guidance_scale=3,  # Classifier-free guidance. Experiement with this value to affect image quality
                        ):
        torch.manual_seed(seed)
        all_imgs = []
        all_z0_preds = []
        
        for i in tqdm(range(n)):
            latents, z0_preds = self.diffusion_loop(
                prompt=prompt,
                seed=None,  # change seed for different images or use `None` for random ones
                num_inference_steps=num_inference_steps,  # More steps better quality but slower, 30-40 is usually a good zone
                guidance_scale=guidance_scale,  # Classifier-free guidance. Experiement with this value to affect image quality
            )
            imgs = self.decode(latents)
            all_imgs.append(imgs)
            all_z0_preds.append(z0_preds)
        all_imgs = torch.cat(all_imgs)
        return all_imgs, all_z0_preds
