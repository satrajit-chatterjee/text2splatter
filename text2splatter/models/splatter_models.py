import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from torch.nn.functional import silu

from text2splatter.splatter_image.scene.gaussian_predictor import (
    Linear,
    Conv2d,
    UNetBlock,
    PositionalEmbedding,
    GroupNorm,
)
from text2splatter.splatter_image.utils.graphics_utils import fov2focal
from text2splatter.splatter_image.utils.general_utils import quaternion_raw_multiply


class SongUNetEncoder(nn.Module):
    def __init__(self,
        splatter_cfg,                       # Splatter configuration.

        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        super().__init__()
        self.splatter_cfg = splatter_cfg

        if self.splatter_cfg.cam_embd.embedding is None:
            in_channels = 3
            emb_dim_in = 0
        else:
            in_channels = 3
            emb_dim_in = 6 * self.splatter_cfg.cam_embd.dimension

        self.img_resolution = splatter_cfg.data.training_resolution
        self.model_channels = splatter_cfg.model.base_dim
        self.num_blocks = splatter_cfg.model.num_blocks
        self.channel_mult_noise = 0
        self.num_blocks = splatter_cfg.model.num_blocks
        self.attn_resolutions = [16]

        self.label_dropout = label_dropout
        self.emb_dim_in = emb_dim_in
        if emb_dim_in > 0:
            emb_channels = self.model_channels * channel_mult_emb
        else:
            emb_channels = None

        noise_channels = self.model_channels * self.channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        if emb_dim_in > 0:
            self.map_layer0 = Linear(in_features=emb_dim_in, out_features=emb_channels, **init)
            self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        if noise_channels > 0:
            self.noise_map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
            self.noise_map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = self.img_resolution >> level
            if level == 0:
                cin = cout
                cout = self.model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(self.num_blocks):
                cin = cout
                cout = self.model_channels * mult
                attn = (res in self.attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)

    def forward(self, x, film_camera_emb=None, N_views_xa=1):
        emb = None

        if film_camera_emb is not None:
            if self.emb_dim_in != 1:
                film_camera_emb = film_camera_emb.reshape(
                    film_camera_emb.shape[0], 2, -1).flip(1).reshape(*film_camera_emb.shape) # swap sin/cos
            film_camera_emb = silu(self.map_layer0(film_camera_emb))
            film_camera_emb = silu(self.map_layer1(film_camera_emb))
            emb = film_camera_emb

        # Encoder.
        skips = []
        aux = x
        final_features = None
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux, N_views_xa)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux, N_views_xa)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux, N_views_xa)) / np.sqrt(2)
            else:
                x = block(x, emb=emb, N_views_xa=N_views_xa) if isinstance(block, UNetBlock) \
                    else block(x, N_views_xa=N_views_xa)
                skips.append(x)
                final_features = x
        return final_features, skips
    

class SongUNetDecoder(nn.Module):
    def __init__(self,
        splatter_cfg,                       # Splatter configuration.
        final_out_channels,                 # Number of output channels.
        final_out_bias,                     # Initial bias for the final output layer.
        final_out_scale,                    # Initial scale for the final output layer.

        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
    ):
        super().__init__()
        self.final_out_channels = final_out_channels
        self.final_out_bias = final_out_bias
        self.final_out_scale = final_out_scale
        self.splatter_cfg = splatter_cfg

        if self.splatter_cfg.cam_embd.embedding is None:
            in_channels = 3
            emb_dim_in = 0
        else:
            in_channels = 3
            emb_dim_in = 6 * self.splatter_cfg.cam_embd.dimension

        self.img_resolution = splatter_cfg.data.training_resolution
        self.model_channels = splatter_cfg.model.base_dim
        self.num_blocks = splatter_cfg.model.num_blocks
        self.channel_mult_noise = 0
        self.num_blocks = splatter_cfg.model.num_blocks
        self.attn_resolutions = [16]

        self.label_dropout = label_dropout
        self.emb_dim_in = emb_dim_in
        if emb_dim_in > 0:
            emb_channels = self.model_channels * channel_mult_emb
        else:
            emb_channels = None

        noise_channels = self.model_channels * self.channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        if emb_dim_in > 0:
            self.map_layer0 = Linear(in_features=emb_dim_in, out_features=emb_channels, **init)
            self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        if noise_channels > 0:
            self.noise_map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
            self.noise_map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = self.img_resolution >> level
            if level == 0:
                cin = cout
                cout = self.model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(self.num_blocks):
                cin = cout
                cout = self.model_channels * mult
                attn = (res in self.attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]
        del self.enc

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = self.img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(self.num_blocks + 1):
                cin = cout + skips.pop()
                cout = self.model_channels * mult
                attn = (idx == self.num_blocks and res in self.attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=sum(self.final_out_channels), out_channels=sum(self.final_out_channels), kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=sum(self.final_out_channels), kernel=3, init_weight=0.2, **init)# init_zero)
        
        self.out = nn.Conv2d(in_channels=sum(self.final_out_channels), 
                                 out_channels=sum(self.final_out_channels),
                                 kernel_size=1)
        
        start_channels = 0
        for out_channel, b, s in zip(
            self.final_out_channels, 
            self.final_out_bias, 
            self.final_out_scale
        ):
            nn.init.xavier_uniform_(
                self.out.weight[start_channels:start_channels+out_channel,
                                :, :, :], s)
            nn.init.constant_(
                self.out.bias[start_channels:start_channels+out_channel], b)
            start_channels += out_channel
    
    def forward(self, x, skips, film_camera_emb=None, N_views_xa=1):
        emb = None

        if film_camera_emb is not None:
            if self.emb_dim_in != 1:
                film_camera_emb = film_camera_emb.reshape(
                    film_camera_emb.shape[0], 2, -1).flip(1).reshape(*film_camera_emb.shape) # swap sin/cos
            film_camera_emb = silu(self.map_layer0(film_camera_emb))
            film_camera_emb = silu(self.map_layer1(film_camera_emb))
            emb = film_camera_emb

        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux, N_views_xa)
            elif 'aux_norm' in name:
                tmp = block(x, N_views_xa)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp), N_views_xa)
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    # skip connection is pixel-aligned which is good for
                    # foreground features
                    # but it's not good for gradient flow and background features
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb=emb, N_views_xa=N_views_xa)
        return self.out(aux)


class Text2SplatDecoder(nn.Module):
    def __init__(self, cfg, do_decode=True):
        super(Text2SplatDecoder, self).__init__()
        self.do_decode = do_decode
        self.cfg = cfg
        split_dimensions, scale_inits, bias_inits = self.get_splits_and_inits(True, cfg)
        self.gaussian_decoder = SongUNetDecoder(
                                cfg, 
                                split_dimensions,
                                final_out_bias=bias_inits,
                                final_out_scale=scale_inits,
                            )
        
        self.init_ray_dirs()
        
        self.depth_act = nn.Sigmoid()
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        if self.cfg.model.max_sh_degree > 0:
            self.init_sh_transform_matrices()

        if self.cfg.cam_embd.embedding is not None:
            if self.cfg.cam_embd.encode_embedding is None:
                self.cam_embedding_map = nn.Identity()
            elif self.cfg.cam_embd.encode_embedding == "positional":
                self.cam_embedding_map = PositionalEmbedding(self.cfg.cam_embd.dimension)
    
    def flatten_vector(self, x):
        # Gets rid of the image dimensions and flattens to a point list
        # B x C x H x W -> B x C x N -> B x N x C
        return x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    
    def get_splits_and_inits(self, with_offset, cfg):
        # Gets channel split dimensions and last layer initialisation
        split_dimensions = []
        scale_inits = []
        bias_inits = []

        if with_offset:
            split_dimensions = split_dimensions + [1, 3, 1, 3, 4, 3]
            scale_inits = scale_inits + [cfg.model.depth_scale, 
                           cfg.model.xyz_scale, 
                           cfg.model.opacity_scale, 
                           cfg.model.scale_scale,
                           1.0,
                           5.0]
            bias_inits = [cfg.model.depth_bias,
                          cfg.model.xyz_bias, 
                          cfg.model.opacity_bias,
                          np.log(cfg.model.scale_bias),
                          0.0,
                          0.0]
        else:
            split_dimensions = split_dimensions + [1, 1, 3, 4, 3]
            scale_inits = scale_inits + [cfg.model.depth_scale, 
                           cfg.model.opacity_scale, 
                           cfg.model.scale_scale,
                           1.0,
                           5.0]
            bias_inits = bias_inits + [cfg.model.depth_bias,
                          cfg.model.opacity_bias,
                          np.log(cfg.model.scale_bias),
                          0.0,
                          0.0]

        if cfg.model.max_sh_degree != 0:
            sh_num = (self.cfg.model.max_sh_degree + 1) ** 2 - 1
            sh_num_rgb = sh_num * 3
            split_dimensions.append(sh_num_rgb)
            scale_inits.append(0.0)
            bias_inits.append(0.0)

        if with_offset:
            self.split_dimensions_with_offset = split_dimensions
        else:
            self.split_dimensions_without_offset = split_dimensions

        return split_dimensions, scale_inits, bias_inits
    
    def get_pos_from_network_output(self, depth_network, offset, focals_pixels, const_offset=None):

        # expands ray dirs along the batch dimension
        # adjust ray directions according to fov if not done already
        ray_dirs_xy = self.ray_dirs.expand(depth_network.shape[0], 3, *self.ray_dirs.shape[2:])
        if self.cfg.data.category in ["hydrants", "teddybears"]:
            assert torch.all(focals_pixels > 0)
            ray_dirs_xy = ray_dirs_xy.clone()
            ray_dirs_xy[:, :2, ...] = ray_dirs_xy[:, :2, ...] / focals_pixels.unsqueeze(2).unsqueeze(3)

        # depth and offsets are shaped as (b 3 h w)
        depth = self.depth_act(depth_network) * (self.cfg.data.zfar - self.cfg.data.znear) + self.cfg.data.znear

        pos = ray_dirs_xy * depth + offset

        return pos
    
    def transform_rotations(self, rotations, source_cv2wT_quat):
        """
        Applies a transform that rotates the predicted rotations from 
        camera space to world space.
        Args:
            rotations: predicted in-camera rotation quaternions (B x N x 4)
            source_cameras_to_world: transformation quaternions from 
                camera-to-world matrices transposed(B x 4)
        Retures:
            rotations with appropriately applied transform to world space
        """

        Mq = source_cv2wT_quat.unsqueeze(1).expand(*rotations.shape)

        rotations = quaternion_raw_multiply(Mq, rotations) 
        
        return rotations
    
    def init_ray_dirs(self):
        x = torch.linspace(-self.cfg.data.training_resolution // 2 + 0.5, 
                            self.cfg.data.training_resolution // 2 - 0.5, 
                            self.cfg.data.training_resolution) 
        y = torch.linspace( self.cfg.data.training_resolution // 2 - 0.5, 
                           -self.cfg.data.training_resolution // 2 + 0.5, 
                            self.cfg.data.training_resolution)
        if self.cfg.model.inverted_x:
            x = -x
        if self.cfg.model.inverted_y:
            y = -y
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        ones = torch.ones_like(grid_x, dtype=grid_x.dtype)
        ray_dirs = torch.stack([grid_x, grid_y, ones]).unsqueeze(0)

        # for cars and chairs the focal length is fixed across dataset
        # so we can preprocess it
        # for co3d this is done on the fly
        if self.cfg.data.category not in ["hydrants", "teddybears"]:
            ray_dirs[:, :2, ...] /= fov2focal(self.cfg.data.fov * np.pi / 180, 
                                              self.cfg.data.training_resolution)
        self.register_buffer('ray_dirs', ray_dirs)

    def init_sh_transform_matrices(self):
        v_to_sh_transform = torch.tensor([[ 0, 0,-1],
                                          [-1, 0, 0],
                                          [ 0, 1, 0]], dtype=torch.float32)
        sh_to_v_transform = v_to_sh_transform.transpose(0, 1)
        self.register_buffer('sh_to_v_transform', sh_to_v_transform.unsqueeze(0))
        self.register_buffer('v_to_sh_transform', v_to_sh_transform.unsqueeze(0))
    
    def transform_SHs(self, shs, source_cameras_to_world):
        # shs: B x N x SH_num x 3
        # source_cameras_to_world: B 4 4
        assert shs.shape[2] == 3, "Can only process shs order 1"
        shs = rearrange(shs, 'b n sh_num rgb -> b (n rgb) sh_num')
        transforms = torch.bmm(
            self.sh_to_v_transform.expand(source_cameras_to_world.shape[0], 3, 3),
            # transpose is because source_cameras_to_world is
            # in row major order 
            source_cameras_to_world[:, :3, :3])
        transforms = torch.bmm(transforms, 
            self.v_to_sh_transform.expand(source_cameras_to_world.shape[0], 3, 3))
        
        shs_transformed = torch.bmm(shs, transforms)
        shs_transformed = rearrange(shs_transformed, 'b (n rgb) sh_num -> b n sh_num rgb', rgb=3)

        return shs_transformed
    
    def make_contiguous(self, tensor_dict):
        return {k: v.contiguous() for k, v in tensor_dict.items()}

    def multi_view_union(self, tensor_dict, B, N_view):
        for t_name, t in tensor_dict.items():
            t = t.reshape(B, N_view, *t.shape[1:])
            tensor_dict[t_name] = t.reshape(B, N_view * t.shape[2], *t.shape[3:])
        return tensor_dict

    def forward(self, 
                x, 
                source_cameras_view_to_world, 
                skips=None,
                source_cv2wT_quat=None,
                focals_pixels=None,
                activate_output=True,
                gaussian_splats=None,
            ):

        B = x.shape[0]
        N_views = 1

        const_offset = None

        source_cameras_view_to_world = source_cameras_view_to_world.reshape(B*N_views, *source_cameras_view_to_world.shape[2:])
        x = x.contiguous(memory_format=torch.channels_last)

        if not self.do_decode:
            assert gaussian_splats is None
            split_network_outputs = gaussian_splats
        else:
            split_network_outputs = self.gaussian_decoder(x, skips)

        split_network_outputs = split_network_outputs.split(self.split_dimensions_with_offset, dim=1)
        depth, offset, opacity, scaling, rotation, features_dc = split_network_outputs[:6]
        if self.cfg.model.max_sh_degree > 0:
            features_rest = split_network_outputs[6]

        pos = self.get_pos_from_network_output(depth, offset, focals_pixels, const_offset=const_offset)

        if self.cfg.model.isotropic:
            scaling_out = torch.cat([scaling[:, :1, ...], scaling[:, :1, ...], scaling[:, :1, ...]], dim=1)
        else:
            scaling_out = scaling

        # Pos prediction is in camera space - compute the positions in the world space
        pos = self.flatten_vector(pos)
        pos = torch.cat([pos, 
                         torch.ones((pos.shape[0], pos.shape[1], 1), device=pos.device, dtype=torch.float32)
                         ], dim=2)
        pos = torch.bmm(pos, source_cameras_view_to_world)
        pos = pos[:, :, :3] / (pos[:, :, 3:] + 1e-10)
        
        out_dict = {
            "xyz": pos, 
            "rotation": self.flatten_vector(self.rotation_activation(rotation)),
            "features_dc": self.flatten_vector(features_dc).unsqueeze(2)
                }

        if activate_output:
            out_dict["opacity"] = self.flatten_vector(self.opacity_activation(opacity))
            out_dict["scaling"] = self.flatten_vector(self.scaling_activation(scaling_out))
        else:
            out_dict["opacity"] = self.flatten_vector(opacity)
            out_dict["scaling"] = self.flatten_vector(scaling_out)

        assert source_cv2wT_quat is not None
        source_cv2wT_quat = source_cv2wT_quat.reshape(B*N_views, *source_cv2wT_quat.shape[2:])
        out_dict["rotation"] = self.transform_rotations(out_dict["rotation"], 
                    source_cv2wT_quat=source_cv2wT_quat)

        if self.cfg.model.max_sh_degree > 0:
            features_rest = self.flatten_vector(features_rest)
            # Channel dimension holds SH_num * RGB(3) -> renderer expects split across RGB
            # Split channel dimension B x N x C -> B x N x SH_num x 3
            out_dict["features_rest"] = features_rest.reshape(*features_rest.shape[:2], -1, 3)
            assert self.cfg.model.max_sh_degree == 1 # "Only accepting degree 1"
            out_dict["features_rest"] = self.transform_SHs(out_dict["features_rest"],
                                                           source_cameras_view_to_world)
        else:    
            out_dict["features_rest"] = torch.zeros((out_dict["features_dc"].shape[0], 
                                                     out_dict["features_dc"].shape[1], 
                                                     (self.cfg.model.max_sh_degree + 1) ** 2 - 1,
                                                     3), dtype=out_dict["features_dc"].dtype, device="cuda")
        
        out_dict = self.multi_view_union(out_dict, B, N_views)
        out_dict = self.make_contiguous(out_dict)

        return out_dict
