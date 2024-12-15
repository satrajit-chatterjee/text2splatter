import os
import math
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict
from PIL import Image
import torch
from copy import deepcopy
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline, AutoencoderKL
from text2splatter.models import Text2SplatDecoder, ExtendedDecoder, CameraTransformGenerator
from text2splatter.splatter_image.datasets.shared_dataset import SharedDataset
from gsplat import rasterization 

@dataclass
class Config:
    repo_id: str
    config_filename: str
    category: str
    training_resolution: int
    num_views: int
    render_resolution: int
    device: str
    pipeline_model: str
    unet_checkpoint: str
    decoder_checkpoint: str
    output_dir: str
    prompt: str
    conditioning: str
    negative_prompt: str

def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions.

    Args:
        q1 (torch.Tensor): A tensor of shape (4,) representing the first quaternion (w, x, y, z).
        q2 (torch.Tensor): A tensor of shape (4,) representing the second quaternion (w, x, y, z).

    Returns:
        torch.Tensor: A tensor of shape (4,) representing the resulting quaternion after multiplication.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.tensor([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], device=q1.device)

def generate_views() -> list:
    """
    Generate a list of view matrices for different camera angles.
    Returns:
        List of torch.Tensor: List of 4x4 camera view matrices.
    """
    views = []
    for angle in range(0, 360, 45):  # 360 degrees in 45-degree increments
        theta = math.radians(angle)
        viewmat = torch.tensor(
            [
                [math.cos(theta), 0.0, math.sin(theta), 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-math.sin(theta), 0.0, math.cos(theta), 12.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device="cuda:2",
        )
        views.append(viewmat)
    return views

def render_gaussian_splats(
    reconstruction, 
    batch_idx=0, 
    W=256, 
    H=256, 
    fov_x=math.pi / 1.2, 
    output_dir="output_images",
    views=None
):
    """
    Render Gaussian splats into 2D images for multiple views.
    
    Args:
        reconstruction (dict): Dictionary containing Gaussian parameters.
        batch_idx (int): Batch index to render.
        W (int): Width of the rendered image.
        H (int): Height of the rendered image.
        fov_x (float): Horizontal field of view in radians.
        output_dir (str): Directory to save output images.
        views (list of torch.Tensor): List of camera view matrices.
    """
    means = reconstruction["xyz"][batch_idx]
    quats = reconstruction["rotation"][batch_idx]
    scales = reconstruction["scaling"][batch_idx]
    opacities = reconstruction["opacity"][batch_idx].squeeze(-1)
    rgbs = reconstruction["features_dc"][batch_idx].squeeze(1)
    background = torch.tensor([0.0, 0.0, 0.0], device=means.device).unsqueeze(0)

    means[:, 0] = 2 * (means[:, 0] - means[:, 0].min()) / (means[:, 0].max() - means[:, 0].min()) - 1
    means[:, 1] = 2 * (means[:, 1] - means[:, 1].min()) / (means[:, 1].max() - means[:, 1].min()) - 1
    means[:, 0] *= 10
    means[:, 1] *= 10

    # Adjust z-coordinates
    # means[:, 2] = (means[:, 2] - means[:, 2].mean()) / 10.0
    # means[:, 2] = 2 * (means[:, 2] - means[:, 2].min()) / (means[:, 2].max() - means[:, 2].min()) - 1
    # means[:, 2] *= 10
    means[:, 2] = torch.abs(means[:, 2]) - 4
    means[:, 2] *= 3.0
    print("X min/max:", means[:, 0].min().item(), means[:, 0].max().item())
    print("Y min/max:", means[:, 1].min().item(), means[:, 1].max().item())
    print("Z min/max:", means[:, 2].min().item(), means[:, 2].max().item())

    opacities = opacities * 5.0
    opacities = opacities.clamp(0.0, 1.0)
    rgbs[opacities < 0.1] = 0.0

    quats = quats / quats.norm(dim=-1, keepdim=True)

    focal = 0.5 * W / math.tan(0.5 * fov_x)
    K = torch.tensor([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1],
    ], device=means.device)

    os.makedirs(output_dir, exist_ok=True)

    if views is None:
        views = [torch.eye(4, device=means.device)]

    for i, viewmat in enumerate(views):
        rendered = rasterization(
            means,
            quats,
            scales,
            opacities,
            torch.sigmoid(rgbs),
            viewmat[None], 
            K[None],    
            W,
            H,
            packed=False,
            backgrounds=background,
        )

        out_img = rendered[0].squeeze(0)
        out_img_np = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(out_img_np)
        img.save(os.path.join(output_dir, f"output_view_{i}.png"))
        print(f"Rendered image for view {i} saved to {output_dir}/output_view_{i}.png")

def load_configuration(repo_id: str, filename: str) -> Dict[str, Any]:
    """
    Load a configuration file from a specified repository.

    Args:
        repo_id (str): The ID of the repository where the configuration file is stored.
        filename (str): The name of the configuration file to be loaded.

    Returns:
        Dict[str, Any]: The loaded configuration as a dictionary.
    """
    config_path = hf_hub_download(repo_id=repo_id, filename=filename)
    config = OmegaConf.load(config_path)
    return config

def setup_pipeline(device: str, model: str, unet_checkpoint: str) -> StableDiffusionPipeline:
    """
    Sets up the Stable Diffusion pipeline with the specified device, model, and UNet checkpoint.

    Args:
        device (str): The device to run the pipeline on (e.g., 'cpu', 'cuda').
        model (str): The name or path of the pre-trained model to use.
        unet_checkpoint (str): The path to the UNet checkpoint to load attention processes from.

    Returns:
        StableDiffusionPipeline: The configured Stable Diffusion pipeline.
    """
    pipeline = StableDiffusionPipeline.from_pretrained(model)
    pipeline.unet.load_attn_procs(unet_checkpoint)
    pipeline.to(device)
    return pipeline

def generate_image(
    pipeline: StableDiffusionPipeline, 
    prompt: str, 
    negative_prompt: str, 
    guidance_scale: float = 7.5, 
    num_inference_steps: int = 30,
    height: int = 512,
    width: int = 512
) -> torch.Tensor:
    """
    Generates an image using the Stable Diffusion pipeline.

    Args:
        pipeline (StableDiffusionPipeline): The Stable Diffusion pipeline to use for image generation.
        prompt (str): The text prompt to guide the image generation.
        negative_prompt (str): The text prompt to guide what should not be in the image.
        guidance_scale (float, optional): The scale for classifier-free guidance. Default is 7.5.
        num_inference_steps (int, optional): The number of denoising steps. Default is 30.
        height (int, optional): The height of the generated image. Default is 512.
        width (int, optional): The width of the generated image. Default is 512.

    Returns:
        torch.Tensor: The generated image as a tensor.
    """
    result = pipeline(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
    )
    return result.images[0]

def generate_latent_image(
    pipeline: StableDiffusionPipeline, 
    prompt: str, 
    negative_prompt: str, 
    guidance_scale: float = 7.5, 
    num_inference_steps: int = 30,
    height: int = 512,
    width: int = 512
) -> torch.Tensor:
    """
    Generates a latent image using the Stable Diffusion pipeline.

    Args:
        pipeline (StableDiffusionPipeline): The Stable Diffusion pipeline to use for image generation.
        prompt (str): The text prompt to guide the image generation.
        negative_prompt (str): The negative text prompt to guide the image generation.
        guidance_scale (float, optional): The scale for classifier-free guidance. Default is 7.5.
        num_inference_steps (int, optional): The number of inference steps. Default is 30.
        height (int, optional): The height of the generated image. Default is 512.
        width (int, optional): The width of the generated image. Default is 512.

    Returns:
        torch.Tensor: The generated latent image tensor.
    """
    result = pipeline(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        output_type="latent"
    )
    return result.images[0]

def load_gaussian_splat_decoder(
    device: str, 
    vae_model: str, 
    decoder_checkpoint: str
) -> ExtendedDecoder:
    """
    Loads a Gaussian splat decoder from a pre-trained VAE model and a decoder checkpoint.

    Args:
        device (str): The device to load the model onto (e.g., 'cpu' or 'cuda').
        vae_model (str): The path or identifier of the pre-trained VAE model.
        decoder_checkpoint (str): The path to the checkpoint file for the decoder.

    Returns:
        ExtendedDecoder: The loaded and initialized Gaussian splat decoder.
    """
    vae = AutoencoderKL.from_pretrained(vae_model, subfolder="vae")
    vae.requires_grad_(False)

    decoder = ExtendedDecoder(deepcopy(vae.decoder))
    del vae

    decoder.load_state_dict(
        torch.load(
            decoder_checkpoint, 
            map_location=device, 
            weights_only=True
        )
    )
    decoder.requires_grad_(False)
    decoder.to(device)
    return decoder

def process_gaussian_splats(decoder: ExtendedDecoder, img_latent: torch.Tensor) -> torch.Tensor:
    """
    Processes Gaussian splats using the provided decoder and image latent tensor.

    Args:
        decoder (ExtendedDecoder): An instance of the ExtendedDecoder class used to decode the latent tensor.
        img_latent (torch.Tensor): A tensor representing the latent image to be processed.

    Returns:
        torch.Tensor: The decoded tensor after processing the Gaussian splats.
    """
    return decoder(img_latent.unsqueeze(0))

def setup_camera_transform_generator(
    cfg: Dict[str, Any], 
    num_views: int, 
    resolution: int
) -> CameraTransformGenerator:
    """
    Set up a CameraTransformGenerator with the given configuration.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary for the camera transform generator.
        num_views (int): Number of views to generate.
        resolution (int): Resolution for the camera transform generator.

    Returns:
        CameraTransformGenerator: An instance of CameraTransformGenerator configured with the provided parameters.
    """
    return CameraTransformGenerator(cfg, num_views, resolution=resolution)

def reconstruct_image(
    cfg: Dict[str, Any],
    gaussian_splats: torch.Tensor,
    metadata: Dict[str, torch.Tensor],
    shared_dataset: SharedDataset,
    device: str
) -> torch.Tensor:
    """
    Reconstruct an image from the given Gaussian splats and metadata using the Text2SplatDecoder.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary for the Text2SplatDecoder.
        gaussian_splats (torch.Tensor): Tensor containing the Gaussian splats.
        metadata (Dict[str, torch.Tensor]): Dictionary containing metadata, including view-world transforms.
        shared_dataset (SharedDataset): Shared dataset object to retrieve source camera world-to-world transforms.
        device (str): Device to run the reconstruction on (e.g., 'cpu' or 'cuda').

    Returns:
        torch.Tensor: The reconstructed image tensor.
    """
    text2splat_dec = Text2SplatDecoder(cfg, do_decode=False).to(device)

    v2w_T = metadata["view_world_transforms"]
    src_cv2w_T = shared_dataset.get_source_cw2wT(v2w_T).unsqueeze(0).to(device)

    # Reduce for single image
    v2w_T = v2w_T.unsqueeze(0).to(device)
    v2w_T = v2w_T[:, :1, ...]
    src_cv2w_T = src_cv2w_T[:, :1, ...]

    return text2splat_dec(
        x=None,
        source_cameras_view_to_world=v2w_T,
        skips=None,
        source_cv2wT_quat=src_cv2w_T,
        focals_pixels=None,
        gaussian_splats=gaussian_splats,
    )

def main():
    # Define configuration
    config = Config(
        repo_id="szymanowiczs/splatter-image-v1",
        config_filename="config_objaverse.yaml",
        category="gso",
        training_resolution=512,
        num_views=25,
        render_resolution=512,
        device="cuda:2",
        pipeline_model="stabilityai/stable-diffusion-2",
        unet_checkpoint="/data/satrajic/saved_models/text-3d-diffusion/lora/unet/checkpoint-32000/",
        decoder_checkpoint="/data/satrajic/saved_models/text-3d-diffusion/decoder/checkpoint-34500/gaussian_splat_decoder.pth",
        output_dir="results",
        # prompt="A sports shoe with a red sole and a white upper",
        conditioning=" high quality, 3D rendering, in the style of ttsplat",
        negative_prompt="disfigured, ugly, unrealistic, cartoon, anime, shadow, not in the style of ttsplat",
    )

    # Load and modify configuration
    cfg = load_configuration(config.repo_id, config.config_filename)
    cfg.data.category = config.category
    cfg.data.training_resolution = config.training_resolution

    # Initialize pipeline
    pipeline = setup_pipeline(config.device, config.pipeline_model, config.unet_checkpoint)

    # Generate and save image
    full_prompt = config.prompt + config.conditioning
    image = generate_image(
        pipeline, full_prompt, config.negative_prompt, height=512, width=512
    )
    os.makedirs(config.output_dir, exist_ok=True)
    image.save(os.path.join(config.output_dir, "generated_image.png"))

    # Generate latent image
    img_latent = generate_latent_image(
        pipeline, full_prompt, config.negative_prompt, height=512, width=512
    )
    del pipeline

    # Load Gaussian splat decoder
    gaussian_splat_decoder = load_gaussian_splat_decoder(
        config.device, config.pipeline_model, config.decoder_checkpoint
    )

    # Process Gaussian splats
    gaussian_splats = process_gaussian_splats(gaussian_splat_decoder, img_latent)

    # Generate camera metadata
    shared_dataset = SharedDataset()
    camera_transform_generator = setup_camera_transform_generator(
        cfg, config.num_views, config.render_resolution
    )
    metadata = camera_transform_generator.generate_camera_transforms()

    # Reconstruct image
    reconstruction = reconstruct_image(
        cfg, gaussian_splats, metadata, shared_dataset, config.device
    )

    # Generate multiple views
    views = generate_views()

    # Render Gaussian splats to 2D image
    render_gaussian_splats(
        reconstruction, 
        batch_idx=0, 
        W=512, 
        H=512, 
        output_dir=os.path.join(
            config.output_dir, 
            "views"
        ),
        views=views
    )

    print("Image, reconstruction, and rendered Gaussian splats saved successfully.")

if __name__ == "__main__":
    main()
