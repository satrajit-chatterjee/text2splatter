from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from text2splatter.splatter_image.datasets import gso
from text2splatter.splatter_image.datasets.dataset_factory import get_dataset

dataset_name = "gso"
cfg_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                                 filename="config_{}.yaml".format("objaverse"))
cfg = OmegaConf.load(cfg_path)
print(cfg.data.category)
cfg.data.category = "gso"
split = "test"
dataset = get_dataset(cfg, split)
print(len(dataset))

output = dataset[0]
print(output.keys())
print(len(output["gt_images"]))

# plot all 25 images in output["gt_images"]
import matplotlib.pyplot as plt
import numpy as np
fig, axs = plt.subplots(5, 5, figsize=(20, 20))
for i in range(25):
    ax = axs[i//5, i%5]
    ax.imshow(np.transpose(np.array(output["gt_images"][i]), (1, 2, 0)))
    ax.axis("off")

plt.savefig("results/gso.png")