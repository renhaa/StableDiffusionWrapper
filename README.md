# StableDiffusionWrapper

## Todo 

- [] Implement option for custom loss 
```python 
    def blue_loss(images):
    # How far are the blue channel values to 0.9:
    error = torch.abs(images[:,-1, :, :] - 0.9).mean() 
    return error    

    ## Input after clasifier free guidace
    #### ADDITIONAL GUIDANCE ###
    # Requires grad on the latents
    latents = latents.detach().requires_grad_()

    # Decode to image space
    denoised_images = vae.decode((1 / 0.18215) * latents_x0) / 2 + 0.5 # (0, 1)

    # Calculate loss
    loss = blue_loss(denoised_images) * blue_loss_scale
    if i%10==0:
      print(i, 'loss:', loss.item())

    # Get gradient
    cond_grad = -torch.autograd.grad(loss, latents)[0]

    # Modify the latents based on this gradient
    latents = latents.detach() + cond_grad * sigma**2
```
- [] Integrate dreambooth textual invertion
 


