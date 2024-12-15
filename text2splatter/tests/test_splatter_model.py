import torch
import numpy as np
from copy import deepcopy
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import torchvision

from text2splatter.models import Text2SplatDecoder, SongUNetEncoder, CameraTransformGenerator, load_encoder_weights
from text2splatter.splatter_image.datasets.dataset_factory import get_dataset
from text2splatter.splatter_image.datasets.shared_dataset import SharedDataset
from text2splatter.splatter_image.scene.gaussian_predictor import GaussianSplatPredictor

device = "cuda:2"
cfg_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", filename="config_{}.yaml".format("objaverse"))
cfg = OmegaConf.load(cfg_path)
cfg.data.category = "gso"
num_views = 25
utils = SharedDataset()
# cfg.data.training_resolution = 512

image_path = "/workspace/project/text2splatter/text2splatter/splatter_image/tests/000.png"

'''
Splatter Image transform used is:

 img = Image.open(paths[i])
# resize to the training resolution
img = torchvision.transforms.functional.resize(img,
                                    self.cfg.data.training_resolution,
                                    interpolation=torchvision.transforms.InterpolationMode.LANCZOS)
img = torchvision.transforms.functional.pil_to_tensor(img) / 255.0
'''

img = Image.open(image_path).convert("RGB")
img = torchvision.transforms.functional.resize(img, cfg.data.training_resolution, interpolation=torchvision.transforms.InterpolationMode.LANCZOS)
img = torchvision.transforms.functional.pil_to_tensor(img) / 255.0

encoder = SongUNetEncoder(cfg)
encoder = load_encoder_weights(encoder, cfg, "gso")
encoder = encoder.to(device)

features, skips = encoder(img.unsqueeze(0).to(device))
print(f"encoded.shape: {features.shape}")

camera_transform_generator = CameraTransformGenerator(cfg, num_views, resolution=128)
metadata = camera_transform_generator.generate_camera_transforms()
text2splat_dec = Text2SplatDecoder(cfg).to(device)
split_dimensions, scale_inits, bias_inits = text2splat_dec.get_splits_and_inits(True, cfg)

v2w_T = metadata["view_world_transforms"]
src_cv2w_T = utils.get_source_cw2wT(v2w_T).unsqueeze(0).to(device)
# Since we are testing 1 image
v2w_T = v2w_T.unsqueeze(0).to(device)
v2w_T = v2w_T[:, :1, ...]
src_cv2w_T = src_cv2w_T[:, :1, ...]

reconstruction = text2splat_dec(
    features, 
    v2w_T,
    skips, 
    src_cv2w_T, 
    None,
)

# Save reconstruction and metadata. They are dictionaries of tensors
torch.save(reconstruction, "/workspace/project/text2splatter/text2splatter/splatter_image/tests/reconstruction_128.pth")
torch.save(metadata, "/workspace/project/text2splatter/text2splatter/splatter_image/tests/metadata_128.pth")
