import os
import numpy as np
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from tqdm import tqdm

from text2splatter.models import SongUNetEncoder, load_encoder_weights
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
encoder = SongUNetEncoder(cfg)
encoder = load_encoder_weights(encoder, cfg, "gso")
encoder = encoder.to(device)
encoder.requires_grad_(False)

for i, datum in enumerate(tqdm(dataset, desc="Processing dataset")):
    data = {k: v.to(device) for k, v in datum.items()}
    input_images = data["gt_images"]
    features, skips = encoder(input_images)
    features = features.reshape(features.size(0), -1).to("cpu")
    np.save(f"results/latent_outputs_{i}.npy", features.numpy())

# Load latent outputs
latent_outputs = []
for i in tqdm(range(len(dataset)), desc="Loading latent outputs"):
    latent_outputs.append(np.load(f"results/latent_outputs_{i}.npy"))

# Plot histogram of latent outputs
latent_outputs = np.concatenate(latent_outputs, axis=0).reshape(-1)

# Calculate mean and standard deviation
mean = np.mean(latent_outputs)
std = np.std(latent_outputs)

plt.hist(latent_outputs, bins=100)
plt.title("Histogram of GSO latent outputs, mean = {:.2f}, std = {:.2f}".format(mean, std))
plt.xlabel("Latent output values")
plt.ylabel("Frequency")
plt.savefig("results/latent_gso_outputs_histogram.png")

# Delete latent outputs
for i in tqdm(range(len(dataset)), desc="Deleting latent outputs"):
    os.remove(f"results/latent_outputs_{i}.npy")
