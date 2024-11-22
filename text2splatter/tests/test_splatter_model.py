import torch
import torch.nn as nn
from copy import deepcopy
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader

from text2splatter.models import Text2SplatDecoder, SongUNetEncoder, SongUNetDecoder
from text2splatter.splatter_image.datasets.dataset_factory import get_dataset
from text2splatter.splatter_image.scene.gaussian_predictor import GaussianSplatPredictor

device = "cuda:1"

dataset_name = "gso"
cfg_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                                 filename="config_{}.yaml".format("objaverse"))
cfg = OmegaConf.load(cfg_path)
cfg.data.category = "gso"
split = "test"
dataset = get_dataset(cfg, split)
output = dataset[0]
dataloader = DataLoader(dataset, batch_size=4, shuffle=True,
                            persistent_workers=True, pin_memory=True, num_workers=1)
datum = next(iter(dataloader))
data = {k: v.to(device) for k, v in datum.items()}
rot_transform_quats = data["source_cv2wT_quat"][:, :cfg.data.input_images]

# input_images.shape: torch.Size([4, 1, 3, 128, 128])
input_images = data["gt_images"][:, :cfg.data.input_images, ...]
print(f"input_images.shape: {input_images.shape}")

# input_images.shape: torch.Size([4, 3, 128, 128])
input_images = input_images.reshape(-1, *input_images.shape[2:])
print(f"input_images.shape: {input_images.shape}")

focals_pixels_pred = None

model_path = hf_hub_download(repo_id="szymanowiczs/splatter-image-v1", 
                        filename="model_{}.pth".format("latest"))

gaussian_predictor = GaussianSplatPredictor(cfg)
ckpt_loaded = torch.load(model_path, map_location=device)
gaussian_predictor.load_state_dict(ckpt_loaded["model_state_dict"])
gaussian_predictor = gaussian_predictor.to(device)

encoder = SongUNetEncoder(cfg).to(device)

features, skips = encoder(input_images)
print(f"encoded.shape: {features.shape}")

decoder = Text2SplatDecoder(cfg).to(device)
reconstruction = decoder(
    features, 
    skips, 
    data["view_to_world_transforms"][:, :cfg.data.input_images, ...], 
    rot_transform_quats, 
    focals_pixels_pred
)
for k, v in reconstruction.items():
    print(f"reconstruction[{k}].shape: {v.shape}")

####################################################################################################
exit()
model = gaussian_predictor
model = model.to(device)
model.eval()

intermediate_outputs = {}

def hook_fn(module, input, output):
    intermediate_outputs[module] = output

# Only encoder
# for name, layer in model.network_with_offset.encoder.enc.items():
#     layer.register_forward_hook(hook_fn)

for name, layer in model.network_with_offset.encoder.dec.items():
    layer.register_forward_hook(hook_fn)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            persistent_workers=True, pin_memory=True, num_workers=10)

datum = next(iter(dataloader))
datum2 = next(iter(dataloader))
data = {k: v.to(device) for k, v in datum.items()}
data2 = {k: v.to(device) for k, v in datum2.items()}
rot_transform_quats = data["source_cv2wT_quat"][:, :cfg.data.input_images]
input_images = data["gt_images"][:, :cfg.data.input_images, ...]
input_images2 = data2["gt_images"][:, :cfg.data.input_images, ...]
assert torch.all(input_images == input_images2)
focals_pixels_pred = None

# print(f"input_images.shape: {input_images.shape}")

reconstruction = model(
                    input_images,
                    data["view_to_world_transforms"][:, :cfg.data.input_images, ...],
                    rot_transform_quats,
                    focals_pixels_pred
                )
exit()
for k, v in reconstruction.items():
    print(f"reconstruction[{k}].shape: {v.shape}")

final_layer = list(intermediate_outputs.keys())[-1]
final_layer_output = intermediate_outputs[final_layer]

print(f"Encoder final_layer name: {final_layer}")
print(f"Encoder final_layer_output.shape: {final_layer_output.shape}")

# Print the first decoder layer name and output
first_decoder_layer = list(intermediate_outputs.keys())[0]
first_decoder_layer_output = intermediate_outputs[first_decoder_layer]
print(f"Decoder first_layer name: {first_decoder_layer}")
print(f"Decoder first_layer_output.shape: {first_decoder_layer_output.shape}")

# Print all the output shapes of the intermediate layers
for k, v in intermediate_outputs.items():
    print(f"{k} output shape: {v.shape}")
