import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

from text2splatter.data.dataset_factory import get_dataset
import torchvision.transforms as transforms

resolution = 128
center_crop = True
random_flip = True

train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

def main():
    dataset_name = "gso"
    cfg_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", filename="config_{}.yaml".format("objaverse"))
    cfg = OmegaConf.load(cfg_path)
    print(cfg.data.category)
    cfg.data.category = dataset_name
    split = "test"
    dataset = get_dataset(cfg, split, transform=train_transforms)
    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
                            persistent_workers=True, pin_memory=True, num_workers=1)

    prompt, image = next(iter(dataloader))
    print(f"{prompt=}")
    print(f"{prompt.shape=}")
    print(f"{image.shape=}")


    # plot all 25 images in batch
    # fig, axs = plt.subplots(5, 5, figsize=(20, 20))
    # for i in range(25):
    #     ax = axs[i // 5, i % 5]
    #     ax.imshow(np.transpose(image["gt_images"][0, i].cpu().numpy(), (1, 2, 0)))
    #     ax.axis("off")
    

    plt.savefig("results/gso.png")

if __name__ == "__main__":
    main()