import math
import torch
import torchvision
import numpy as np

class CameraTransformGenerator:
    def __init__(self, cfg, num_views, resolution=512):
        self.cfg = cfg
        self.num_views = num_views
        self.resolution = resolution
        self.projection_matrix = self.getProjectionMatrix(
            znear=cfg.data.znear, zfar=cfg.data.zfar,
            fovX=cfg.data.fov * 2 * np.pi / 360, 
            fovY=cfg.data.fov * 2 * np.pi / 360).transpose(0,1)
        self.opengl_to_colmap = torch.tensor([[  1,  0,  0,  0],
                                              [  0, -1,  0,  0],
                                              [  0,  0, -1,  0],
                                              [  0,  0,  0,  1]], dtype=torch.float32)
        self.initial_twc = self.sample_twc()

    def getProjectionMatrix(self, znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))
        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right
        P = torch.zeros(4, 4)
        z_sign = 1.0
        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    def fov2focal(self, fov, pixels):
        return pixels / (2 * math.tan(fov / 2))

    def sample_twc(self):
        sample_tensor = torch.zeros((3, 4), dtype=torch.float32)
        sample_tensor[0, 0] = torch.mean(torch.rand(1000) * 2 - 1)
        sample_tensor[0, 1] = torch.mean(torch.rand(1000) * 2 - 1)
        sample_tensor[0, 2] = 0
        sample_tensor[0, 3] = torch.mean(torch.normal(0, 0.01, size=(1000,)))
        sample_tensor[1, 0] = torch.Tensor([np.mean(np.random.triangular(-1, 0, 1, size=1000))])
        sample_tensor[1, 1] = torch.mean(torch.normal(0, 0.41, size=(1000,)))
        lower, upper, mu, sigma = 0, 1, 0.79, 0.1
        truncated_samples = torch.clamp(torch.normal(mu, sigma, size=(1000,)), min=lower, max=upper)
        sample_tensor[1, 2] = torch.mean(truncated_samples)
        sample_tensor[1, 3] = torch.mean(torch.normal(0, 0.01, size=(1000,)))
        samples1 = torch.normal(-0.5, 0.1, size=(500,))
        samples2 = torch.normal(0.5, 0.1, size=(500,))
        sample_tensor[2, 0] = torch.mean(torch.cat([samples1, samples2]))
        samples1 = torch.normal(-0.5, 0.05, size=(500,))
        samples2 = torch.normal(0.5, 0.05, size=(500,))
        sample_tensor[2, 1] = torch.mean(torch.cat([samples1, samples2]))
        skew_param, loc, scale = 2, 0, 0.58
        skew_samples = loc + scale * torch.distributions.gamma.Gamma(skew_param, 1).sample((1000,))
        sample_tensor[2, 2] = torch.mean(skew_samples)
        sample_tensor[2, 3] = torch.mean(torch.empty(1000).uniform_(-2.2, -1.6))
        return sample_tensor

    def generate_camera_transforms(self):
        world_view_transforms = []
        view_world_transforms = []
        camera_centers = []
        for _ in range(self.num_views):
            w2c_cmo = self.sample_twc()
            w2c_cmo = torch.cat([w2c_cmo, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)], dim=0)
            w2c_cmo = torch.matmul(self.opengl_to_colmap, w2c_cmo)
            world_view_transform = w2c_cmo.transpose(0, 1)
            view_world_transform = w2c_cmo.inverse().transpose(0, 1)
            camera_center = view_world_transform[3, :3].clone()
            world_view_transforms.append(world_view_transform)
            view_world_transforms.append(view_world_transform)
            camera_centers.append(camera_center)
        world_view_transforms = torch.stack(world_view_transforms)
        view_world_transforms = torch.stack(view_world_transforms)
        camera_centers = torch.stack(camera_centers)
        focals_pixels = torch.full((self.num_views, 2), fill_value=self.fov2focal(self.cfg.data.fov, self.resolution))
        pps_pixels = torch.zeros((self.num_views, 2))
        translation_scaling_factor = 2.0 / torch.norm(camera_centers[0])
        world_view_transforms[:, 3, :3] *= translation_scaling_factor
        view_world_transforms[:, 3, :3] *= translation_scaling_factor
        camera_centers *= translation_scaling_factor
        full_proj_transforms = world_view_transforms.bmm(self.projection_matrix.unsqueeze(0).expand(
            world_view_transforms.shape[0], 4, 4))
        return {
            "world_view_transforms": world_view_transforms,
            "view_world_transforms": view_world_transforms,
            "camera_centers": camera_centers,
            "full_proj_transforms": full_proj_transforms,
            "focals_pixels": focals_pixels,
            "pps_pixels": pps_pixels
        }