import math
import torch
import numpy as np
from PIL import Image
from gsplat import rasterization  # Assuming gsplat is installed and available

# Sample reconstruction dictionary (replace with your actual data)
# reconstruction = {...}

def render_gaussian_splats(reconstruction, batch_idx=0, W=256, H=256, fov_x=math.pi / 2.0, output_path="output_image.png"):
    """
    Render Gaussian splats into a 2D image.
    
    Args:
        reconstruction (dict): Dictionary containing Gaussian parameters.
        batch_idx (int): Batch index to render.
        W (int): Width of the rendered image.
        H (int): Height of the rendered image.
        fov_x (float): Horizontal field of view in radians.
        output_path (str): Path to save the output image.
    """
    # Extract Gaussian parameters for the selected batch
    means = reconstruction["xyz"][batch_idx]  # shape: [16384, 3]
    quats = reconstruction["rotation"][batch_idx]  # shape: [16384, 4]
    scales = reconstruction["scaling"][batch_idx]  # shape: [16384, 3]
    opacities = reconstruction["opacity"][batch_idx].squeeze(-1)  # shape: [16384]
    rgbs = reconstruction["features_dc"][batch_idx].squeeze(1)  # shape: [16384, 3]

    means[:, 2] = torch.abs(means[:, 2]) + 8.0  # Shift z-coordinates
    opacities = opacities * 5.0                 # Scale opacities
    opacities = opacities.clamp(0.1, 1.0)       # Clamp to minimum visibility
    rgbs = torch.tanh(rgbs)                     # Normalize colors
    rgbs = (rgbs + 1) / 2.0                     # Shift to [0, 1]
    fov_x = math.pi / 1.5                       # Wider field of view

    print("Means (xyz) min/max:", means.min(dim=0).values, means.max(dim=0).values)
    print("Colors (after sigmoid) min/max:", torch.sigmoid(rgbs).min(), torch.sigmoid(rgbs).max())
    print("Colors (before sigmoid) min/max:", rgbs.min(), rgbs.max())
    print("Opacities (after sigmoid) min/max:", opacities.min(), opacities.max())

    # Normalize quaternions
    quats = quats / quats.norm(dim=-1, keepdim=True)

    # Camera intrinsics (focal length and principal point)
    focal = 0.5 * W / math.tan(0.5 * fov_x)
    K = torch.tensor([
        [focal, 0, W / 2],
        [0, focal, H / 2],
        [0, 0, 1],
    ], device=means.device)

    # Camera extrinsics (view matrix)
    viewmat = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 10.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=means.device,
    )

    # Render using the rasterization function
    rendered = rasterization(
        means,
        quats,
        scales,
        # torch.sigmoid(opacities),  # Apply sigmoid for opacities
        # torch.sigmoid(rgbs),       # Apply sigmoid for colors
        opacities,
        rgbs,
        viewmat[None],             # Add batch dimension
        K[None],                   # Add batch dimension
        W,
        H,
        packed=False
    )
    print(f"Rendered image shape: {rendered[0].shape}")

    # Extract the rendered image (remove batch dimension)
    out_img = rendered[0].squeeze(0)  # shape: [H, W, 3]

    # Convert to NumPy array and save as an image
    out_img_np = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(out_img_np)
    img.save(output_path)
    print(f"Rendered image saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Assume 'reconstruction' dictionary is provided
    # reconstruction = {
    #     "xyz": torch.rand(4, 16384, 3).cuda(),
    #     "rotation": torch.rand(4, 16384, 4).cuda(),
    #     "features_dc": torch.rand(4, 16384, 1, 3).cuda(),
    #     "opacity": torch.rand(4, 16384, 1).cuda(),
    #     "scaling": torch.rand(4, 16384, 3).cuda(),
    #     "features_rest": torch.rand(4, 16384, 3, 3).cuda(),
    # }

    gaussian_splat_path = f"../splatter_image/tests/reconstruction_512.pth"
    reconstruction = torch.load(gaussian_splat_path, map_location='cuda:2', weights_only=True)
    for k, v in reconstruction.items():
        reconstruction[k] = v.requires_grad_(False)

    render_gaussian_splats(reconstruction, batch_idx=0, W=512, H=512, output_path="output_image.png")
