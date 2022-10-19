import torch
from tqdm import tqdm

from wrapper import StableDifussionWrapper, image_grid




self = StableDifussionWrapper()


# Warning: This wrapper does not have a safety filter nor adds watermark to images. Be ethical!

prompt = "The Baha'i Gardens in Haifa at daytime, trending on artstation, vibrant, digital art, 8K"

n = 9
all_imgs = []
for i in tqdm(range(n)):
    latents = self.diffusion_loop(
        prompt=prompt,
        seed=None,  # change seed for different images or use `None` for random ones
        num_inference_steps=50,  # More steps better quality but slower, 30-40 is usually a good zone
        guidance_scale=5,  # Classifier-free guidance. Experiement with this value to affect image quality
    )

    imgs = self.decode(latents)
    all_imgs.append(imgs)
all_imgs = torch.cat(all_imgs)
image_grid(all_imgs, rows=3, cols=3).save("bahai-day.png")
