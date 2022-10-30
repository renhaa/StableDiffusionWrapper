import torch
from tqdm import tqdm 
from torch import autocast, inference_mode
from wrapper import StableDifussionWrapper

from wrapper import tensor_to_pil, image_grid, make_vid_from_pil_imgs
#from wrapper import jupyter_display_video

self = StableDifussionWrapper(scheduler_type = "lms")

# z1 = self.random_latents(seed = 41+3)
# z2 = self.random_latents(seed = 42+3)
# z3 = self.random_latents(seed = 43+3)
# z4 = self.random_latents(seed = 43+4)

# zss1 = self.interpolation(z1,z2,numsteps = 32)
# zss2 = self.interpolation(z2,z3,numsteps = 32)
# zss3 = self.interpolation(z3,z4,numsteps = 32)
# zss4 = self.interpolation(z4,z1,numsteps = 32)


xs = [
("A photo of an elephant standing on a skateboard, during Yum Kippur in Haifa, realistic, HD",(1,2,3,4,5,6,7,8)),
("A painting of a fork and a knive, digital art, middle eastern style, art, colorfull, HD",(432,454,4522,4455,54,76,35,88)),
("A painting of Cleaning articles, dish soap, kitchen cloth, art, colorfull, digital art, HD",(432,454,4522,4455,54,76,35,88))
]



# zss = torch.cat([zss1,zss2,zss3, zss4],axis = 0)
out_folder ="generations/"

for prompt,seeds in xs:
    z0s = []
    for seed in seeds:
        z0, z0_preds = self.diffusion_loop(
                                prompt=prompt,
                                latents=None,
                                start_step=0,
                                seed=seed,
                                show_progbar = True,
                                num_inference_steps=75,
                                guidance_scale=7.5,
                                return_latents_t0_preds = True )
        z0s.append(z0)
    imgs = torch.cat([self.decode(z0) for z0 in z0s])
    out_path = f"{out_folder}{prompt}.jpg"
    image_grid(imgs, cols=2,rows=4).save()  
    print("saved to", out_path)  
# pil_imgs = self.z_to_pil(torch.cat(z0s), size = (512,512))

# out_name = "haifa_cosmos_lerp1.mp4"
# make_vid_from_pil_imgs(pil_imgs, 
#                         framerate = 12,
#                         out_name = out_name)
# print("done saved to", out_name)
print("[ALL DONE!]")