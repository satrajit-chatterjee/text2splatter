import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline,  UNet2DConditionModel

pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
pipeline.unet.load_attn_procs("/data/satrajic/saved_models/text-3d-diffusion/lora/unet/checkpoint-16000/")
pipeline.to("cuda:2")
conditioning = " high quality, 3D rendering, no shadow in the style of ttsplat"
# prompt = "A sports shoe with a red sole and a white upper, high quality, 3D rendering, no shadow in the style of ttsplat"
prompt = "Illustrate a top-down view of a bee pull toy, highlighting the colorful stripes and wheel designs."
prompt += conditioning
negative_prompt = "disfigured, ugly, unrealistic, cartoon, anime, shadow, not in the style of ttsplat"
image = pipeline(prompt, 
                 num_inference_steps=30, 
                 guidance_scale=7.5, 
                 seed=42, 
                 negative_prompt=negative_prompt
                ).images[0]
image.save("results/output.png")
