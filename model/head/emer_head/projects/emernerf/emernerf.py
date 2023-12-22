from nerfstudio.data.scene_box import SceneBox
from nerfstudio.models.base_model import Model, ModelConfig
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union, Callable
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
import numpy as np
# from nerfstudio.utils.external import TCNN_EXISTS, tcnn
import projects.emernerf.tcnn_modules as tcnn
# from nerfstudio.field_components.activations import trunc_exp
from projects.emernerf.nerf_utils import trunc_exp, contract
from projects.emernerf.nerfacc_prop_net import PropNetEstimator, ProposalNetworkSampler, get_proposal_requires_grad_fn
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.field_components.mlp import MLP
import torch.nn.functional as F
import projects.emernerf.emernerf_loss as emernerf_loss
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
# from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer, SemanticRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from collections import defaultdict
import itertools
from nerfacc import (
    accumulate_along_rays,
    render_transmittance_from_density,
    render_weight_from_density,
)


@dataclass
class EncoderConfig:
    n_input_dims: int = 4 # 3 for xyz, 1 for time
    n_levels: int = 10
    n_features_per_level: int = 4
    base_resolution:int = 32  # didn't do any ablation study on this
    max_resolution: int = 8192
    log2_hashmap_size: int = 18 


@dataclass
class DensityEncoderConfig:
    n_input_dims: int = 3
    n_levels_per_prop = [8, 8]
    base_resolutions_per_prop = [16, 16]
    max_resolution_per_prop = [512, 2048]
    log2_hashmap_size_per_prop = [20, 20]
    n_features_per_level = 1

    def get_config(self, idx):
        return EncoderConfig(self.n_input_dims, 
                             self.n_levels_per_prop[idx],
                             self.n_features_per_level,
                             self.base_resolutions_per_prop[idx],
                             self.max_resolution_per_prop[idx],
                             self.log2_hashmap_size_per_prop[idx]
                             )


@dataclass
class EmerNerfModelConfig(ModelConfig):
    """emernerf Model Config"""
    _target: Type = field(
        default_factory=lambda: EmerNerfModel
    ) 
    xyz_encoder: EncoderConfig = EncoderConfig(3, 10, 4, 16, 8192, 20)
    dynamic_xyz_encoder: EncoderConfig = EncoderConfig(4, 16, 4, 32, 8192, 18)
    flow_xyz_encoder: EncoderConfig = EncoderConfig(4, 10, 4, 16, 4096, 18)
    density_encoder: DensityEncoderConfig = DensityEncoderConfig()
    enable_flow_branch: bool = True 
    unbounded: bool = True
    enable_img_embedding: bool = True
    appearance_embedding_dim: int = 16
    enable_sky_head: bool = False
    feature_embedding_dim: int = 64
    feature_mlp_layer_width: int = 64
    enable_dynamic_branch: bool = True
    enable_shadow_head: bool = True
    base_mlp_layer_width: int = 64
    geometry_feature_dim: int = 64
    semantic_feature_dim: int = 64
    head_mlp_layer_width: int = 64
    check_nan: bool = False
    disable_scene_contraction: bool = False
    num_proposal_iterations: int = 2
    implementation: Literal["tcnn", "torch"] = "tcnn"
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    use_single_jitter: bool = True
    num_nerf_samples_per_ray: int = 64
    num_proposal_samples_per_ray: Tuple[int, ...] = (128, 64)
    use_proposal_weight_anneal: bool = True
    proposal_weights_anneal_max_num_iters: int = 1000
    proposal_weights_anneal_slope: float = 10.0
    background_color: Literal["random", "last_sample", "black", "white"] = "last_sample"
    use_gradient_scaling = False
    near_plane: float = 0.1
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    num_iters: int = 30000
    loss_scale: int = 1024


class XYZ_Encoder(nn.Module):
    encoder_type = "XYZ_Encoder"
    """Encode XYZ coordinates or directions to a vector."""

    def __init__(self, n_input_dims):
        super().__init__()
        self.n_input_dims = n_input_dims

    @property
    def n_output_dims(self) -> int:
        raise NotImplementedError


# class MLP(nn.Module):
#     """A simple MLP with skip connections."""

#     def __init__(
#         self,
#         in_dims: int,
#         out_dims: int,
#         num_layers: int = 3,
#         hidden_dims: Optional[int] = 256,
#         skip_connections: Optional[Tuple[int]] = [0],
#     ) -> None:
#         super().__init__()
#         self.in_dims = in_dims
#         self.hidden_dims = hidden_dims
#         self.n_output_dims = out_dims
#         self.num_layers = num_layers
#         self.skip_connections = skip_connections
#         layers = []
#         if self.num_layers == 1:
#             layers.append(nn.Linear(in_dims, out_dims))
#         else:
#             for i in range(self.num_layers - 1):
#                 if i == 0:
#                     layers.append(nn.Linear(in_dims, hidden_dims))
#                 elif i in skip_connections:
#                     layers.append(nn.Linear(in_dims + hidden_dims, hidden_dims))
#                 else:
#                     layers.append(nn.Linear(hidden_dims, hidden_dims))
#             layers.append(nn.Linear(hidden_dims, out_dims))
#         self.layers = nn.ModuleList(layers)

#     def forward(self, x: Tensor) -> Tensor:
#         input = x
#         for i, layer in enumerate(self.layers):
#             if i in self.skip_connections:
#                 x = torch.cat([x, input], -1)
#             x = layer(x)
#             if i < len(self.layers) - 1:
#                 x = nn.functional.relu(x)
#         return x
    

class HashEncoder(XYZ_Encoder):
    encoder_type = "HashEncoder"

    def __init__(
        self,
        cfg: EncoderConfig,
        dtype=torch.float32,
    ) -> None:
        super().__init__(cfg.n_input_dims)
        self.n_input_dims = cfg.n_input_dims
        self.num_levels = cfg.n_levels
        self.base_resolution = cfg.base_resolution
        self.max_resolution = cfg.max_resolution
        self.log2_hashmap_size = cfg.log2_hashmap_size
        self.n_features_per_level = cfg.n_features_per_level
        self.num_parameters = 2**cfg.log2_hashmap_size * cfg.n_features_per_level * cfg.n_levels

        self.growth_factor = np.exp(
            (np.log(self.max_resolution) - np.log(self.base_resolution)) / (self.num_levels - 1)
        )
        self.encoding_config = {
            "otype": "HashGrid",
            "n_levels": self.num_levels,
            "n_features_per_level": self.n_features_per_level,
            "log2_hashmap_size": self.log2_hashmap_size,
            "base_resolution": self.base_resolution,
            "per_level_scale": self.growth_factor,
            "interpolation": "linear",
        }
        self.tcnn_encoding = tcnn.Encoding(
            n_input_dims=self.n_input_dims,
            encoding_config=self.encoding_config,
            dtype=dtype,
        )

    @property
    def n_output_dims(self) -> int:
        return self.tcnn_encoding.n_output_dims

    def forward(self, in_tensor: Tensor) -> Tensor:
        return self.tcnn_encoding(in_tensor)


class SinusoidalEncoder(XYZ_Encoder):
    encoder_type = "SinusoidalEncoder"
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(
        self,
        n_input_dims: int = 3,
        min_deg: int = 0,
        max_deg: int = 10,
        enable_identity: bool = True,
    ):
        super().__init__(n_input_dims)
        self.n_input_dims = n_input_dims
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.enable_identity = enable_identity
        self.register_buffer(
            "scales", Tensor([2**i for i in range(min_deg, max_deg + 1)])
        )

    @property
    def n_output_dims(self) -> int:
        return (
            int(self.enable_identity) + (self.max_deg - self.min_deg + 1) * 2
        ) * self.n_input_dims

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., n_input_dims]
        Returns:
            encoded: [..., n_output_dims]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[..., None, :] * self.scales[:, None]),
            list(x.shape[:-1])
            + [(self.max_deg - self.min_deg + 1) * self.n_input_dims],
        )
        encoded = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
        if self.enable_identity:
            encoded = torch.cat([x] + [encoded], dim=-1)
        return encoded


def find_topk_nearby_timesteps(
    original: Tensor, query: Tensor, topk: int = 2, return_indices: bool = False
) -> Tensor:
    """
    Find the closest two closest in `original` tensor for each value in `query` tensor.

    Parameters:
    - original (torch.Tensor): Original tensor of timesteps.
    - query (torch.Tensor): Query tensor of timesteps for which closest indices are to be found.

    Returns:
    - torch.Tensor: Indices in the original tensor that are the two closest to each timestep in the query tensor.
    """

    # Expand dimensions of tensors to compute pairwise distances
    original_expanded = original.unsqueeze(0)
    query_expanded = query.unsqueeze(1)

    # Compute pairwise absolute differences
    abs_diffs = torch.abs(original_expanded - query_expanded)

    # Find indices of the two minimum absolute differences along the dimension of the original tensor
    _, closest_indices = torch.topk(abs_diffs, k=topk, dim=1, largest=False)
    if return_indices:
        return original[closest_indices], closest_indices
    return original[closest_indices]


def temporal_interpolation(
    normed_timestamps: Tensor,
    training_timesteps: Tensor,
    normed_positions: Tensor,
    hash_encoder: HashEncoder,
    mlp: nn.Module,
    interpolate_xyz_encoding: bool = False,
) -> Tensor:
    # to be studied
    if len(normed_timestamps.shape) == 2:
        timestep_slice = normed_timestamps[:, 0]
    else:
        timestep_slice = normed_timestamps[:, 0, 0]
    closest_timesteps = find_topk_nearby_timesteps(training_timesteps, timestep_slice)
    if torch.allclose(closest_timesteps[:, 0], timestep_slice):
        temporal_positions = torch.cat([normed_positions, normed_timestamps], dim=-1)
        xyz_encoding = hash_encoder(
            temporal_positions.view(-1, temporal_positions.shape[-1])
        ).view(list(temporal_positions.shape[:-1]) + [-1])
        encoded_feats = mlp(xyz_encoding)
    else:
        left_timesteps, right_timesteps = (
            closest_timesteps[:, 0],
            closest_timesteps[:, 1],
        )
        left_timesteps = left_timesteps.unsqueeze(-1).repeat(
            1, normed_positions.shape[1]
        )
        right_timesteps = right_timesteps.unsqueeze(-1).repeat(
            1, normed_positions.shape[1]
        )
        left_temporal_positions = torch.cat(
            [normed_positions, left_timesteps.unsqueeze(-1)], dim=-1
        )
        right_temporal_positions = torch.cat(
            [normed_positions, right_timesteps.unsqueeze(-1)], dim=-1
        )
        offset = (
            (
                (timestep_slice - left_timesteps[:, 0])
                / (right_timesteps[:, 0] - left_timesteps[:, 0])
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        left_xyz_encoding = hash_encoder(
            left_temporal_positions.view(-1, left_temporal_positions.shape[-1])
        ).view(list(left_temporal_positions.shape[:-1]) + [-1])
        right_xyz_encoding = hash_encoder(
            right_temporal_positions.view(-1, left_temporal_positions.shape[-1])
        ).view(list(right_temporal_positions.shape[:-1]) + [-1])
        if interpolate_xyz_encoding:
            encoded_feats = mlp(
                left_xyz_encoding * (1 - offset) + right_xyz_encoding * offset
            )
        else:
            encoded_feats = (
                mlp(left_xyz_encoding) * (1 - offset) + mlp(right_xyz_encoding) * offset
            )

    return encoded_feats


class EmerNerfField(Field):
    def __init__(self, cfg: EmerNerfModelConfig, 
                 aabb,
                 num_train_data, # these params are decided by dataset
                 spatial_distortion: Optional[SpatialDistortion] = None,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.aabb = aabb
        # ======== Static Field ======== #
        self.xyz_encoder = HashEncoder(self.cfg.xyz_encoder)
        self.base_mlp = nn.Sequential(
            nn.Linear(self.xyz_encoder.n_output_dims, self.cfg.base_mlp_layer_width),
            nn.ReLU(),
            nn.Linear(
                self.cfg.base_mlp_layer_width, self.cfg.geometry_feature_dim
            ),
        )

        # ======== Dynamic Field ======== #
        self.dynamic_xyz_encoder = None
        if self.cfg.enable_dynamic_branch:
            self.dynamic_xyz_encoder = HashEncoder(self.cfg.dynamic_xyz_encoder)
            self.register_buffer("training_timesteps", torch.zeros(num_train_data))
            self.dynamic_base_mlp = nn.Sequential(
                nn.Linear(self.dynamic_xyz_encoder.n_output_dims, self.cfg.base_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(
                    self.cfg.base_mlp_layer_width, self.cfg.geometry_feature_dim + self.cfg.semantic_feature_dim
                ),
            )
            
        # ======== Flow Field ======== #
        self.flow_xyz_encoder = None
        if self.cfg.enable_flow_branch:
            self.flow_xyz_encoder = HashEncoder(self.cfg.flow_xyz_encoder)
            self.flow_mlp = nn.Sequential(
                nn.Linear(
                    self.flow_xyz_encoder.n_output_dims,
                    self.cfg.base_mlp_layer_width,
                ),
                nn.ReLU(),
                nn.Linear(self.cfg.base_mlp_layer_width, self.cfg.base_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(self.cfg.base_mlp_layer_width, 6),  # 3 for forward, 3 for backward
                # no activation function for flow
            )

        # appearance embedding
        self.appearance_embedding = None
        if self.cfg.enable_img_embedding:
            self.appearance_embedding = nn.Embedding(
                num_train_data, self.cfg.appearance_embedding_dim
            )

        # direction encoding
        self.direction_encoding = SinusoidalEncoder(
            n_input_dims=3, min_deg=0, max_deg=4
        )


        # ======== Color Head ======== #
        self.rgb_head = MLP(
            in_dim = self.cfg.geometry_feature_dim+self.direction_encoding.n_output_dims
            +(self.cfg.appearance_embedding_dim if self.cfg.enable_img_embedding else 0),
            num_layers=3,
            out_dim=3,
            layer_width=self.cfg.head_mlp_layer_width,
            skip_connections=(1,),
        )

        # self.rgb_head = MLP(
        #     in_dims=self.cfg.geometry_feature_dim
        #     + self.direction_encoding.n_output_dims
        #     + (self.cfg.appearance_embedding_dim if self.cfg.enable_img_embedding else 0),
        #     out_dims=3,
        #     num_layers=3,
        #     hidden_dims=self.cfg.head_mlp_layer_width,
        #     skip_connections=[1],
        # )

        # ======== Shadow Head ======== #
        self.shadow_head = None
        if self.cfg.enable_shadow_head:
            self.shadow_head = nn.Sequential(
                nn.Linear(self.cfg.geometry_feature_dim, self.cfg.base_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(self.cfg.base_mlp_layer_width, 1),
                nn.Sigmoid(),
            )

        # ======== Sky Head ======== #
        self.sky_head = None
        if self.cfg.enable_sky_head:
            self.sky_head = MLP(
                in_dim=self.direction_encoding.n_output_dims
                + (
                    self.cfg.appearance_embedding_dim
                    if self.cfg.enable_img_embedding
                    else 0
                ),
                out_dim=3,
                num_layers=3,
                layer_width=self.cfg.head_mlp_layer_width,
                skip_connections=(1,),
            )

        self.spatial_distortion = spatial_distortion
        self.density_activation = lambda x: trunc_exp(x - 1)
        self.num_dims = 3

    def contract_points(self, positions)->torch.Tensor:
        if self.spatial_distortion is not None:
            normed_positions = self.spatial_distortion(positions)
            normed_positions = (normed_positions + 2.0) / 4.0
        else:
            normed_positions = SceneBox.get_normalized_positions(positions, self.aabb)

        selector = (
            ((normed_positions > 0.0) & (normed_positions < 1.0))
            .all(dim=-1)
            .to(positions)
        )
        normed_positions = normed_positions * selector.unsqueeze(-1)
        return normed_positions

    def forward_static_hash(
        self,
        ray_samples: RaySamples
    ):
        """
        forward pass for static hash encoding

        Returns:
            encoded_features: [..., geometry_feature_dim + (semantic_feature_dim)]
            normed_positions: [..., 3] in [0, 1]
        """
        # if self.spatial_distortion is not None:
        #     positions = ray_samples.frustums.get_positions()
        #     positions = self.spatial_distortion(positions)
        #     positions = (positions + 2.0) / 4.0
        # else:
        #     positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions = ray_samples.frustums.get_positions()
        positions = self.contract_points(positions)
        xyz_encoding = self.xyz_encoder(positions.view(-1, 3))
        encoded_features = self.base_mlp(xyz_encoding).view(
            list(positions.shape[:-1]) + [-1]
        )

        return encoded_features, positions

    def forward_dynamic_hash(
        self,
        normed_positions: Tensor,
        normed_timestamps: Tensor,
        return_hash_encodings: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """
        forward pass for dynamic hash encoding

        Returns:
            encoded_dynamic_feats: [..., geometry_feature_dim + (semantic_feature_dim)]
            dynamic_xyz_encoding: [..., n_output_dims] (optional)
        """
        if normed_timestamps.shape[-1] != 1:
            normed_timestamps = normed_timestamps.unsqueeze(-1)
        # To be fixed.
        # if self.training or not self.enable_temporal_interpolation:
        if True:
            temporal_positions = torch.cat(
                [normed_positions, normed_timestamps], dim=-1
            )
            dynamic_xyz_encoding = self.dynamic_xyz_encoder(
                temporal_positions.view(-1, self.num_dims + 1) # type: ignore
            ).view(list(temporal_positions.shape[:-1]) + [-1])
            encoded_dynamic_feats = self.dynamic_base_mlp(dynamic_xyz_encoding)

        if return_hash_encodings:
            return encoded_dynamic_feats, dynamic_xyz_encoding
        else:
            return encoded_dynamic_feats

    def temporal_aggregation(
        self,
        positions: Tensor,  # current world coordinates
        normed_timestamps: Tensor,  # current normalized timestamps
        forward_flow: Tensor,
        backward_flow: Tensor,
        dynamic_feats: Tensor,
    ):
        """
        temporal aggregation for dynamic features
        Eq. (8) in the emernerf paper
        """
        if normed_timestamps.shape[-1] != 1:
            normed_timestamps = normed_timestamps.unsqueeze(-1)
        if self.training:
            noise = torch.rand_like(forward_flow)[..., 0:1]
        else:
            noise = torch.ones_like(forward_flow)[..., 0:1]
        # forward and backward warped positions
        forward_warped_positions = self.contract_points(
            positions + forward_flow * noise
        )
        backward_warped_positions = self.contract_points(
            positions + backward_flow * noise
        )
        # forward and backward warped timestamps
        forward_warped_time = torch.clamp(
            normed_timestamps + self.time_diff * noise, 0, 1.0
        )
        backward_warped_time = torch.clamp(
            normed_timestamps - self.time_diff * noise, 0, 1.0
        )
        (
            forward_dynamic_feats,
            forward_dynamic_hash_encodings,
        ) = self.forward_dynamic_hash(
            forward_warped_positions,
            forward_warped_time,
            return_hash_encodings=True,
        )
        (
            backward_dynamic_feats,
            backward_dynamic_hash_encodings,
        ) = self.forward_dynamic_hash(
            backward_warped_positions,
            backward_warped_time,
            return_hash_encodings=True,
        )
        forward_pred_flow = self.forward_flow_hash(
            forward_warped_positions,
            forward_warped_time,
        )
        backward_pred_flow = self.forward_flow_hash(
            backward_warped_positions,
            backward_warped_time,
        )
        # simple weighted sum
        aggregated_dynamic_feats = (
            dynamic_feats + 0.5 * forward_dynamic_feats + 0.5 * backward_dynamic_feats
        ) / 2.0
        return {
            "dynamic_feats": aggregated_dynamic_feats,
            "forward_pred_backward_flow": forward_pred_flow[..., 3:],
            "backward_pred_forward_flow": backward_pred_flow[..., :3],
            # to be studied
            "forward_dynamic_hash_encodings": forward_dynamic_hash_encodings,
            "backward_dynamic_hash_encodings": backward_dynamic_hash_encodings,
        }

    def forward_flow_hash(
        self,
        normed_positions: Tensor,
        normed_timestamps: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        forward pass for flow hash encoding

        Returns:
            flow: [..., 6] (forward_flow, backward_flow)
        """
        assert self.flow_xyz_encoder is not None
        if normed_timestamps.shape[-1] != 1:
            normed_timestamps = normed_timestamps.unsqueeze(-1)
        if self.training:
            temporal_positions = torch.cat(
                [normed_positions, normed_timestamps], dim=-1
            )
            flow_xyz_encoding = self.flow_xyz_encoder(
                temporal_positions.view(-1, self.num_dims + 1)
            ).view(list(temporal_positions.shape[:-1]) + [-1])
            flow = self.flow_mlp(flow_xyz_encoding)
        else:
            flow = temporal_interpolation(
                normed_timestamps,
                self.training_timesteps,
                normed_positions,
                self.flow_xyz_encoder,
                self.flow_mlp,
                interpolate_xyz_encoding=True,
            )
        return flow

    def query_rgb(
        self,
        directions: Tensor,
        geo_feats: Tensor,
        dynamic_geo_feats: Tensor = None,
        img_idx = None,
    ) -> Tensor:
        
        directions = (directions + 1.0) / 2.0  # do we need this?
        h = self.direction_encoding(directions.reshape(-1, directions.shape[-1])).view(
            *directions.shape[:-1], -1
        )
        if self.cfg.enable_img_embedding:
            if img_idx is not None:
                # print(img_idx.device)
                img_idx = img_idx.to(directions.device)
                appearance_embedding = self.appearance_embedding(img_idx)
            else:
                # use mean appearance embedding
                # print("using mean appearance embedding")
                appearance_embedding = torch.ones(
                    (*directions.shape[:-1], self.cfg.appearance_embedding_dim),
                    device=directions.device,
                ) * self.appearance_embedding.weight.mean(dim=0)
            h = torch.cat([h, appearance_embedding], dim=-1)

        rgb = self.rgb_head(torch.cat([h, geo_feats], dim=-1))
        rgb = F.sigmoid(rgb)
        results = {"rgb": rgb}

        if self.dynamic_xyz_encoder is not None:
            assert (
                dynamic_geo_feats is not None
            ), "Dynamic geometry features are not provided."
            dynamic_rgb = self.rgb_head(torch.cat([h, dynamic_geo_feats], dim=-1))
            dynamic_rgb = F.sigmoid(dynamic_rgb)
            results["dynamic_rgb"] = dynamic_rgb
        return results

    def query_sky(
        self, directions: Tensor, img_idx=None
    ) -> Dict[str, Tensor]:
        if len(directions.shape) == 2:
            dd = self.direction_encoding(directions).to(directions)
        else:
            dd = self.direction_encoding(directions[:, 0]).to(directions)
        if self.cfg.enable_img_embedding:
            if img_idx is not None:
                appearance_embedding = self.appearance_embedding(img_idx)
            else:
                # use mean appearance embedding
                appearance_embedding = torch.ones(
                    (*directions.shape[:-1], self.cfg.appearance_embedding_dim),
                    device=directions.device,
                ) * self.appearance_embedding.weight.mean(dim=0)
            dd = torch.cat([dd, appearance_embedding], dim=-1)
        rgb_sky = self.sky_head(dd).to(directions)
        rgb_sky = F.sigmoid(rgb_sky)
        results = {"rgb_sky": rgb_sky}
        # if self.enable_feature_head:
        #     self.dino_sky_head(dd).to(directions)
        #     results["dino_sky_feat"] = self.dino_sky_head(dd).to(directions)
        return results

    def register_normalized_training_timesteps(
        self, normalized_timesteps: Tensor, time_diff: float = None, device = 'cuda'
    ) -> None:
        """
        register normalized timesteps for temporal interpolation

        Args:
            normalized_timesteps (Tensor): normalized timesteps in [0, 1]
            time_diff (float, optional): time difference between two consecutive timesteps. Defaults to None.
        """
        if self.dynamic_xyz_encoder is not None:
            # register timesteps for temporal interpolation
            # self.training_timesteps.copy_(normalized_timesteps)
            self.training_timesteps = torch.nn.Parameter(normalized_timesteps).to(device)
            if time_diff is not None:
                # use the provided time difference if available
                self.time_diff = time_diff
            else:
                if len(self.training_timesteps) > 1:
                    # otherwise, compute the time difference from the provided timesteps
                    # it's important to make sure the provided timesteps are consecutive
                    self.time_diff = (
                        self.training_timesteps[1] - self.training_timesteps[0]
                    )
                else:
                    self.time_diff = 0

    def forward(self, ray_samples: RaySamples, combine_static_dynamic=False, compute_normals: bool = False) -> Dict[str, Tensor]:
        """
        Args:
            positions: [..., 3]
            directions: [..., 3]
            data_dict: a dictionary containing additional data
            return_density_only: if True, only return density without querying other heads
            combine_static_dynamic: if True, combine static and dynamic predictions based on static and dynamic density
            in addition to returning separate results for static and dynamic fields
            query_feature_head: if True, query feature head
            query_pe_head: if True, query PE head. Disable this if we want to directly query 3D features.
        Returns:
            results_dict: a dictionary containing everything
        """
        results_dict = {}
        # forward static branch
        encoded_features, normed_positions = self.forward_static_hash(ray_samples)
        geo_feats = encoded_features
        # geo_feats, semantic_feats = torch.split(
        #     encoded_features,
        #     [self.cfg.geometry_feature_dim, self.cfg.semantic_feature_dim],
        #     dim=-1,
        # )
        # static_density = self.density_activation(geo_feats[..., 0:1])
        static_density = self.density_activation(geo_feats[..., 0])

        # directions = get_normalized_directions(ray_samples.frustums.directions)
        directions = ray_samples.frustums.directions
        camera_indices = ray_samples.camera_indices.squeeze()
        if self.dynamic_xyz_encoder is not None and ray_samples.times is not None:
            # forward dynamic branch
            normed_timestamps = ray_samples.times
            dynamic_feats, dynamic_hash_encodings = self.forward_dynamic_hash(
                normed_positions, normed_timestamps, return_hash_encodings=True
            )

            if self.flow_xyz_encoder is not None:
                flow = self.forward_flow_hash(normed_positions, normed_timestamps)
                forward_flow, backward_flow = flow[..., :3], flow[..., 3:]
                results_dict["forward_flow"] = forward_flow
                results_dict["backward_flow"] = backward_flow
                temporal_aggregation_results = self.temporal_aggregation(
                    ray_samples.frustums.get_positions(),
                    normed_timestamps,
                    forward_flow,
                    backward_flow,
                    dynamic_feats,
                )
                # overwrite dynamic feats using temporal aggregation results
                dynamic_feats = temporal_aggregation_results["dynamic_feats"]
                # to be studied
                temporal_aggregation_results[
                    "current_dynamic_hash_encodings"
                ] = dynamic_hash_encodings
                results_dict.update(temporal_aggregation_results)
            (dynamic_geo_feats, dynamic_semantic_feats,) = torch.split(
                dynamic_feats,
                [self.cfg.geometry_feature_dim, self.cfg.semantic_feature_dim],
                dim=-1,
            )
            # dynamic_density = self.density_activation(dynamic_geo_feats[..., 0:1])
            dynamic_density = self.density_activation(dynamic_geo_feats[..., 0])
            # blend static and dynamic density to get the final density
            density = static_density + dynamic_density
            results_dict.update(
                {
                    "density": density,
                    "static_density": static_density,
                    "dynamic_density": dynamic_density,
                }
            )
            # if return_density_only:
            #     # skip querying other heads
            #     return results_dict

            if directions is not None:
                rgb_results = self.query_rgb(
                    directions, geo_feats, dynamic_geo_feats, img_idx=camera_indices
                )
                results_dict["dynamic_rgb"] = rgb_results["dynamic_rgb"]
                results_dict["static_rgb"] = rgb_results["rgb"]
                if combine_static_dynamic:
                    static_ratio = static_density / (density + 1e-6)
                    dynamic_ratio = dynamic_density / (density + 1e-6)
                    results_dict["rgb"] = (
                        static_ratio[..., None] * results_dict["static_rgb"]
                        + dynamic_ratio[..., None] * results_dict["dynamic_rgb"]
                    )
            if self.cfg.enable_shadow_head:
                shadow_ratio = self.shadow_head(dynamic_geo_feats)
                results_dict["shadow_ratio"] = shadow_ratio
                if combine_static_dynamic and "rgb" in results_dict:
                    results_dict["rgb"] = (
                        static_ratio[..., None]
                        * results_dict["rgb"]
                        * (1 - shadow_ratio)
                        + dynamic_ratio[..., None] * results_dict["dynamic_rgb"]
                    )
        else:
            # if no dynamic branch, use static density
            results_dict["density"] = static_density
            # if return_density_only:
            #     # skip querying other heads
            #     return results_dict
            if directions is not None:
                rgb_results = self.query_rgb(directions, geo_feats, img_idx=camera_indices)
                results_dict["rgb"] = rgb_results["rgb"]
        #TODO: Feature head

        # query sky if not in lidar mode
        if (
            self.cfg.enable_sky_head
            # and "lidar_origin" not in data_dict
            # and directions is not None
        ):
            directions = directions[:, 0]
            # reduced_data_dict = {k: v[:, 0] for k, v in data_dict.items()}
            sky_results = self.query_sky(directions, img_idx=camera_indices[:, 0])
            results_dict.update(sky_results)

        return results_dict


class SceneAABBContraction(SpatialDistortion):
    def __init__(self, order, aabb) -> None:
        super().__init__()
        self.order = order
        self.aabb = aabb

    def forward(self, positions):
        def my_contract(x):
            aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)  # 0~1
            x = x * 2 - 1  # aabb is at [-1, 1]
            mag = torch.linalg.norm(x, ord=float("inf"), dim=-1, keepdim=True)
            x = torch.where(mag < 1, x, (2 - 1 / mag) * (x / mag))
            return x
        # aabb_lengths = self.aabb[1] - self.aabb[0]
        # normalized_positions = (positions - self.aabb[0]) / aabb_lengths
        # normalized_positions = normalized_positions * 2 - 1 
        normalized_positions = my_contract(positions) # -2~2
        return normalized_positions


class DensityField(nn.Module):
    def __init__(
        self,
        xyz_encoder: HashEncoder,
        aabb: Union[Tensor, List[float]] = [[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]],
        num_dims: int = 3,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        base_mlp_layer_width: int = 64,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dims = num_dims
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.xyz_encoder = xyz_encoder

        # density head
        self.base_mlp = nn.Sequential(
            nn.Linear(self.xyz_encoder.n_output_dims, base_mlp_layer_width),
            nn.ReLU(),
            nn.Linear(base_mlp_layer_width, 1),
        )

    @property
    def device(self) -> torch.device:
        return self.aabb.device

    def set_aabb(self, aabb: Union[Tensor, List[float]]) -> None:
        if not isinstance(aabb, Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.aabb.copy_(aabb)
        self.aabb = self.aabb.to(self.device)

    def forward(
        self, positions: Tensor, data_dict: Dict[str, Tensor] = None
    ) -> Dict[str, Tensor]:
        if self.unbounded:
            # use infinte norm to contract the positions for cuboid aabb
            positions = contract(positions, self.aabb, ord=float("inf"))
        else:
            aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
            positions = (positions - aabb_min) / (aabb_max - aabb_min)
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1).to(positions)
        positions = positions * selector.unsqueeze(-1)
        xyz_encoding = self.xyz_encoder(positions.view(-1, self.num_dims))
        density_before_activation = self.base_mlp(xyz_encoding).view(
            list(positions.shape[:-1]) + [-1]
        )
        density = self.density_activation(density_before_activation)
        return {"density": density}
    

class EmerNerfModel(Model):
    """Instant NGP model

    Args:
        config: instant NGP configuration to instantiate model
    """

    config: EmerNerfModelConfig

    def __init__(self, config: EmerNerfModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def register_normalized_training_timesteps(
        self, normalized_timesteps: Tensor, time_diff: float = None
    ) -> None:
        self.field.register_normalized_training_timesteps(normalized_timesteps, time_diff, self.device)

    def build_propnet(self, num_iters):
        proposal_networks = nn.ModuleList([
            DensityField(
                HashEncoder(self.config.density_encoder.get_config(i)),
                aabb = self.scene_aabb[None, ...],
                unbounded=self.config.unbounded
            )
            for i in range(len(self.config.num_proposal_samples_per_ray))])
        
        # hard code
        prop_optimizer = torch.optim.Adam(
            itertools.chain(*[p.parameters() for p in proposal_networks]),
            lr=0.01,
            eps=1e-15,
            weight_decay=1e-5,
            betas=(0.9, 0.99),
        )

        scheduler_milestones = [
            num_iters // 2,
            num_iters * 3 // 4,
            num_iters * 9 // 10,
        ]
        if num_iters >= 10000:
            scheduler_milestones.insert(0, num_iters // 4)
        prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    prop_optimizer,
                    start_factor=0.01,
                    total_iters=num_iters // 10,
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    prop_optimizer,
                    milestones=scheduler_milestones,
                    gamma=0.33,
                ),
            ]
        )
        estimator = PropNetEstimator(
            prop_optimizer,
            prop_scheduler,
            enable_anti_aliasing_loss=True,
            anti_aliasing_pulse_width=[0.03, 0.003]
        )
        self.proposal_sampler = ProposalNetworkSampler(estimator, 
                                                       proposal_networks,
                                                       num_samples=self.config.num_nerf_samples_per_ray,
                                                       num_samples_per_prop=self.config.num_proposal_samples_per_ray,
                                                       near_plane=self.config.near_plane,
                                                       far_plane=self.config.far_plane,
                                                       sampling_type='uniform_lindisp')
        
        # self.proposal_requires_grad_fn = get_proposal_requires_grad_fn()

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.scene_aabb = Parameter(self.scene_box.aabb.flatten(), requires_grad=False)

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneAABBContraction(order=float("inf"), aabb=self.scene_aabb)
            # scene_contraction = SceneContraction(order=float("inf"))
        self.field = EmerNerfField(self.config, self.scene_aabb, self.num_train_data, scene_contraction)
        
        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations

        # losses
        self.rgb_loss_fn = emernerf_loss.RealValueLoss(
            loss_type='l2',
            coef=1.0,
            name="rgb",
            check_nan=self.config.check_nan,
        )

        self.sky_loss_fn = emernerf_loss.SkyLoss(
            loss_type='opacity_based',
            coef=0.001,
            check_nan=self.config.check_nan,
        )

        self.dynamic_reg_loss_fn = emernerf_loss.DynamicRegularizationLoss(
            loss_type='sparsity',
            coef=0.01,
            entropy_skewness=1.1,
            check_nan=self.config.check_nan,
        )

        self.shadow_loss_fn = emernerf_loss.DynamicRegularizationLoss(
            name="shadow",
            loss_type='sparsity',
            coef=0.01,
            check_nan=self.config.check_nan,
        )

        # self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        # self.renderer_accumulation = AccumulationRenderer()
        # self.renderer_depth = DepthRenderer(method="median")
        # self.renderer_expected_depth = DepthRenderer(method="expected")
        # self.renderer_normals = NormalsRenderer()

        # shaders
        # self.normals_shader = NormalsShader()

        # losses
        self.step = 0
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        self.build_propnet(self.config.num_iters)

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.proposal_sampler.step_cb,
            )
        )
        return callbacks
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        # self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle, combine_static_dynamic: bool = False):
        # apply the camera optimizer pose tweaks
        if self.training:
            proposal_requires_grad = self.proposal_sampler.is_required_grad()
        else:
            proposal_requires_grad = False

        ray_samples = self.proposal_sampler.forward(ray_bundle, proposal_requires_grad)
        results = self.field.forward(ray_samples, combine_static_dynamic=combine_static_dynamic)
        t_starts = ray_samples.frustums.starts.squeeze(-1)
        t_ends = ray_samples.frustums.ends.squeeze(-1)
        trans, alphas = render_transmittance_from_density(
            t_starts, t_ends, results["density"].squeeze(-1)
        )
        # Calculate weights for each point along the rays based on the transmittance and alpha values
        weights = trans * alphas

        extras = {
            "weights": weights,
            # the transmittance of the intervals
            "trans": trans,
            # the midpoints of the intervals
            "t_vals": (t_starts + t_ends) / 2.0,
            # the lengths of the intervals
            "t_dist": (t_ends - t_starts),
        }

        for k in [
            # predicted forward flow
            "forward_flow",
            # predicted backward flow
            "backward_flow",
            # the predicted backward flow from the forward-warpped points
            "forward_pred_backward_flow",
            # the predicted forward flow from the backward-warpped points
            "backward_pred_forward_flow",
        ]:
            if k in results:
                extras[k] = results[k]

        # =============== Geometry ================ #
        opacities = accumulate_along_rays(weights, values=None).clamp(1e-6, 1.0)
        # expected depth
        depths = accumulate_along_rays(weights, values=(t_starts + t_ends)[..., None] / 2.0)
        depths = depths / opacities
        # median depth
        steps = (t_starts + t_ends)[..., None] / 2.0
        cumulative_weights = torch.cumsum(weights, dim=-1)  # [..., num_samples]
        # [..., 1]
        split = torch.ones((*weights.shape[:-1], 1), device=weights.device) * 0.5
        # [..., 1]
        median_index = torch.searchsorted(cumulative_weights, split, side="left")
        median_index = torch.clamp(median_index, 0, steps.shape[-2] - 1)  # [..., 1]
        median_depth = torch.gather(steps[..., 0], dim=-1, index=median_index)  # [..., 1]

        results_dict = {
            "density": results["density"].squeeze(-1),
            "depth": depths,
            "accumulation": opacities,
            "median_depth": median_depth,
        }

        # =========== Geometry Decomposition =========== #
        if "static_density" in results and "dynamic_density" in results:
            extras["static_density"] = results["static_density"]
            extras["dynamic_density"] = results["dynamic_density"]
            # blend static and dynamic densities
            static_ratio = results["static_density"] / (results["density"] + 1e-6)
            dynamic_ratio = results["dynamic_density"] / (results["density"] + 1e-6)

        # =========== RGB =========== #
        if "rgb" in results:
            # static-only scene
            results_dict["rgb"] = accumulate_along_rays(weights, values=results["rgb"])
        elif "static_rgb" in results and "dynamic_rgb" in results:
            # default to no shadow
            shadow_ratio = 0.0
            if "shadow_ratio" in results:
                shadow_ratio = results["shadow_ratio"]
                results_dict["shadow_ratio"] = accumulate_along_rays(
                    weights,
                    values=shadow_ratio.square(),
                )
            rgb = (
                static_ratio[..., None] * results["static_rgb"] * (1 - shadow_ratio)
                + dynamic_ratio[..., None] * results["dynamic_rgb"]
            )
            results_dict["rgb"] = accumulate_along_rays(weights, values=rgb)

        # Sky composition.
        if "rgb_sky" in results:
            results_dict["rgb"] = results_dict["rgb"] + results["rgb_sky"] * (
                1.0 - results_dict["accumulation"]
            )
            if "static_rgb" in results_dict:
                # add sky to static rgb
                results_dict["static_rgb"] = results_dict["static_rgb"] + results[
                    "rgb_sky"
                ] * (1.0 - results_dict["static_opacity"])

        results_dict["extras"] = extras


        if self.training:
            self.proposal_sampler.update_every_n_steps(
                results_dict["extras"]["trans"],
                proposal_requires_grad,
                loss_scaler=1024,
            )
        return results_dict
    
    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)
        # image = self.renderer_rgb.blend_background(image)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        # metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"][..., :3].to(self.device)
        # pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
        #     pred_image=outputs["rgb"],
        #     pred_accumulation=outputs["accumulation"],
        #     gt_image=image,
        # )
        pred_rgb = outputs['rgb']
        # print(torch.cat((pred_rgb[:5], image[:5]), 1))
        pixel_loss_dict = {}

        rgb_loss = self.rgb_loss_fn(image, pred_rgb)
        pixel_loss_dict.update(rgb_loss)
        if self.sky_loss_fn is not None:  # if sky loss is enabled
            pixel_loss_dict.update(
                self.sky_loss_fn(outputs["accumulation"].squeeze(), batch["sky_mask"].squeeze().to(self.device))
            )

        if self.config.enable_dynamic_branch:
            if self.dynamic_reg_loss_fn is not None:
                pixel_loss_dict.update(
                    self.dynamic_reg_loss_fn(
                        dynamic_density=outputs["extras"]["dynamic_density"],
                        static_density=outputs["extras"]["static_density"],
                    )
                )

        if self.config.enable_shadow_head:
            if self.shadow_loss_fn is not None:
                pixel_loss_dict.update(
                    self.shadow_loss_fn(
                        outputs["shadow_ratio"],
                    )
                )

        if "forward_flow" in outputs["extras"]:
            cycle_loss = (
                0.5
                * (
                    (
                        outputs["extras"]["forward_flow"].detach()
                        + outputs["extras"]["forward_pred_backward_flow"]
                    )
                    ** 2
                    + (
                        outputs["extras"]["backward_flow"].detach()
                        + outputs["extras"]["backward_pred_forward_flow"]
                    )
                    ** 2
                ).mean()
            )
            pixel_loss_dict.update({"cycle_loss": cycle_loss * 0.01})

        # if self.config.use_gradient_scaling:
        #     field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)
        for k in pixel_loss_dict.keys():
            pixel_loss_dict[k] = self.config.loss_scale * pixel_loss_dict[k]
        return pixel_loss_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                outputs_lists[output_name].append(output.to(input_device))
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        # image = self.renderer_rgb.blend_background(image)
        rgb = outputs["rgb"]
        # acc = colormaps.apply_colormap(outputs["accumulation"])
        # depth = colormaps.apply_depth_colormap(
        #     outputs["depth"],
        #     accumulation=outputs["accumulation"],
        # )

        combined_rgb = torch.cat([image, rgb], dim=1)
        # combined_acc = torch.cat([acc], dim=1)
        # combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {
            "img": combined_rgb,
            # "accumulation": combined_acc,
            # "depth": combined_depth,
        }

        return metrics_dict, images_dict


