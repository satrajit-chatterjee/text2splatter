import os
import torch
import numpy as np
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from tqdm import tqdm
from diffusers import AutoencoderKL
from copy import deepcopy

from text2splatter.models import (
    SongUNetEncoder, 
    SongUNetDecoder, 
    Text2SplatDecoder,
    load_encoder_weights, 
    load_decoder_weights
)
from text2splatter.splatter_image.datasets.dataset_factory import get_dataset

device = "cuda:1"

dataset_name = "gso"
cfg_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                                 filename="config_{}.yaml".format("objaverse"))
cfg = OmegaConf.load(cfg_path)
cfg.data.category = "gso"
split = "test"
dataset = get_dataset(cfg, split)

cfg_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1",
                                filename="config_{}.yaml".format("objaverse"))
cfg = OmegaConf.load(cfg_path)
gt_encoder = SongUNetEncoder(cfg)
gt_encoder = load_encoder_weights(gt_encoder, cfg, "gso")
gt_encoder = gt_encoder.to(device)
gt_encoder.requires_grad_(False)

text2splat_dec = Text2SplatDecoder(cfg)
split_dimensions, scale_inits, bias_inits = text2splat_dec.get_splits_and_inits(True, cfg)
gt_decoder = SongUNetDecoder(
    cfg,
    split_dimensions,
    bias_inits,
    scale_inits
)
gt_decoder = load_decoder_weights(gt_decoder, cfg, dataset_name)
gt_decoder = gt_decoder.to(device)
gt_decoder.requires_grad_(False)

vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2", subfolder="vae"
    )
vae.requires_grad_(False)
gaussian_splat_decoder = deepcopy(vae.decoder)
gaussian_splat_decoder.conv_in = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
gaussian_splat_decoder.conv_out = torch.nn.Conv2d(128, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

finetuned_state_dict = torch.load("/data/satrajic/saved_models/text-3d-diffusion/checkpoint-2500/gaussian_splat_decoder.pth")
gaussian_splat_decoder.load_state_dict(finetuned_state_dict)
gaussian_splat_decoder = gaussian_splat_decoder.to(device)
gaussian_splat_decoder.requires_grad_(False)

for i, datum in enumerate(tqdm(dataset, desc="Processing dataset")):
    data = {k: v.to(device) for k, v in datum.items()}
    input_images = data["gt_images"]
    features, skips = gt_encoder(input_images)
    gt_gaussian_splat = gt_decoder(features, skips)
    gt_gaussian_splat = gt_gaussian_splat.reshape(gt_gaussian_splat.size(0), -1).to("cpu")
    np.save(f"results/gt_gaussian_splats_outputs_{i}.npy", gt_gaussian_splat.numpy())

    pred_gaussian_splat = gaussian_splat_decoder(features)
    pred_gaussian_splat = pred_gaussian_splat.reshape(pred_gaussian_splat.size(0), -1).to("cpu")
    np.save(f"results/pred_gaussian_splats_outputs_{i}.npy", pred_gaussian_splat.numpy())
    if i == 100:
        break


# Load latent outputs
gt_gaussian_splat = []
pred_gaussian_splat = []
for i in tqdm(range(len(dataset)), desc="Loading gaussian splats"):
    gt_gaussian_splat.append(np.load(f"results/gt_gaussian_splats_outputs_{i}.npy"))
    pred_gaussian_splat.append(np.load(f"results/pred_gaussian_splats_outputs_{i}.npy"))
    if i == 100:
        break

# Plot histogram of gt gaussian splats and pred gaussian splats side by side
gt_gaussian_splat = np.concatenate(gt_gaussian_splat, axis=0).reshape(-1)
pred_gaussian_splat = np.concatenate(pred_gaussian_splat, axis=0).reshape(-1)

# Calculate mean and standard deviation
gt_mean = np.mean(gt_gaussian_splat)
gt_std = np.std(gt_gaussian_splat)

pred_mean = np.mean(pred_gaussian_splat)
pred_std = np.std(pred_gaussian_splat)

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].hist(gt_gaussian_splat, bins=100)
axs[0].set_title("Histogram of GT Gaussian splats, mean = {:.2f}, std = {:.2f}".format(gt_mean, gt_std))
axs[0].set_xlabel("Gaussian splat values")
axs[0].set_ylabel("Frequency")

axs[1].hist(pred_gaussian_splat, bins=100)
axs[1].set_title("Histogram of Pred Gaussian splats, mean = {:.2f}, std = {:.2f}".format(pred_mean, pred_std))
axs[1].set_xlabel("Gaussian splat values")
axs[1].set_ylabel("Frequency")

plt.savefig("results/finetuned_gaussian_splats_histogram.png")

# Delete latent outputs
for i in tqdm(range(len(dataset)), desc="Deleting gaussian splats temp files"):
    os.remove(f"results/gt_gaussian_splats_outputs_{i}.npy")
    os.remove(f"results/pred_gaussian_splats_outputs_{i}.npy")
    if i == 100:
        break

