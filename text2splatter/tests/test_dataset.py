import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

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

dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                            persistent_workers=True, pin_memory=True, num_workers=4)

output = next(iter(dataloader))
print(output.keys())
print(len(output["gt_images"]))

# plot all 25 images in output["gt_images"]
fig, axs = plt.subplots(5, 5, figsize=(20, 20))
for i in range(25):
    ax = axs[i//5, i%5]
    ax.imshow(np.transpose(output["gt_images"][0, i].cpu().numpy(), (1, 2, 0)))
    ax.axis("off")

plt.savefig("results/gso.png")