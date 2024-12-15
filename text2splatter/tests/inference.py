import torch
import torch.nn as nn
from copy import deepcopy
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline, AutoencoderKL
from text2splatter.models import Text2SplatDecoder, ExtendedDecoder


device = "cuda:2"
pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
pipeline.unet.load_attn_procs("/data/satrajic/saved_models/text-3d-diffusion/lora/unet/checkpoint-32000/")
pipeline.to(device)
conditioning = " high quality, 3D rendering, no shadow in the style of ttsplat"
# prompt = "A sports shoe with a red sole and a white upper, high quality, 3D rendering, no shadow in the style of ttsplat"
prompt = "A jenga tower"
prompt += conditioning
negative_prompt = "disfigured, ugly, unrealistic, cartoon, anime, shadow, not in the style of ttsplat"
# image = pipeline(prompt, 
#                  num_inference_steps=30, 
#                  guidance_scale=7.5, 
#                 #  seed=42, 
#                 #  negative_prompt=negative_prompt,
#                 #  height=128,
#                 #  width=128,
#                 ).images[0]
# image.save("results/inference.png")

img_latent = pipeline(prompt, 
                 num_inference_steps=30, 
                 guidance_scale=7.5, 
                #  seed=42, 
                #  negative_prompt=negative_prompt,
                 height=512,
                 width=512,
                 output_type="latent"
                ).images[0]
del pipeline
print(img_latent.shape)

vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2", subfolder="vae"
        )
vae.requires_grad_(False)

gaussian_splat_decoder = ExtendedDecoder(deepcopy(vae.decoder))
del vae
gaussian_splat_decoder.load_state_dict(
    torch.load("/data/satrajic/saved_models/text-3d-diffusion/decoder/checkpoint-34500/gaussian_splat_decoder.pth", map_location=device)
    )
gaussian_splat_decoder.requires_grad_(False)
gaussian_splat_decoder.to(device)

output = gaussian_splat_decoder(img_latent.unsqueeze(0))
print(output.shape)

cfg_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                                 filename="config_{}.yaml".format("objaverse"))
cfg = OmegaConf.load(cfg_path)
cfg.data.category = "gso"
text2splat_dec = Text2SplatDecoder(cfg, do_decode=False).to(device)
split_dimensions, scale_inits, bias_inits = text2splat_dec.get_splits_and_inits(True, cfg)

output = text2splat_dec(output)

