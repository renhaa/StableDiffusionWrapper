import torch
from tqdm import tqdm 
from torch import autocast, inference_mode
from wrapper import StableDifussionWrapper

from wrapper import tensor_to_pil, image_grid, make_vid_from_pil_imgs
#from wrapper import jupyter_display_video

self = StableDifussionWrapper(scheduler_type = "lms")

z1 = self.random_latents(seed = 41+3)
z2 = self.random_latents(seed = 42+3)
z3 = self.random_latents(seed = 43+3)

zss1 = self.interpolation(z1,z2,numsteps = 64)
zss2 = self.interpolation(z2,z3,numsteps = 64)
zss3 = self.interpolation(z3,z1,numsteps = 62)

zss = torch.cat([zss1,zss2,zss3],axis = 0)

prompt = "A beautiful human face, portrait photo, high res"

z0s = []
for z in tqdm(zss):
    z0, z0_preds = self.diffusion_loop(
                            prompt=prompt,
                            latents=z.unsqueeze(0),
                            start_step=0,
                            seed=None,
                            show_progbar = False,
                            num_inference_steps=25,
                            guidance_scale=7.5,
                            return_latents_t0_preds = True)
    z0s.append(z0)
    
pil_imgs = self.z_to_pil(torch.cat(z0s), size = (512,512))

out_name = "face_lerp2.mp4"
make_vid_from_pil_imgs(pil_imgs, 
                        framerate = 8,
                        out_name = out_name)
print("done saved to", out_name)