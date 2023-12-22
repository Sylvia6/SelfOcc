from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Type
from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.data.scene_box import SceneBox
from torch.utils.data._utils.collate import collate, default_collate_fn_map

import torch
from jaxtyping import Float, Shaped
from torch import Tensor, nn
import torch.nn.functional as F

from mmengine.registry import MODELS
from mmengine.model import BaseModule
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstuido.fields.base_field import Field
from nerfacc import (
    accumulate_along_rays,
    render_transmittance_from_density,
    render_weight_from_density,
)

from .utils import MLP
from ...utils import interpolation 

class DynamicVoxLinearInterpolationField(Field):
    def __init__(self, feat_dim, head_mlp_layer_width, aabb, *args, **kwargs):
        super(DynamicVoxLinearInterpolationField, self).__init__(*args, **kwargs)

        self.static_feats = None
        self.dynamic_feats = None
        self.dense_activation = lambda x: trunc_exp(x - 1)
        self.aabb = aabb
        xmin, ymin, zmin, tmin, xmax, ymax, zmax, tmax = aabb
        self.aabb_min = torch.nn.Parameter(torch.tensor([xmin, ymin, zmin, tmin], dtype=torch.float32), return_grad=False)
        self.aabb_max = torch.nn.Parameter(torch.tensor([xmin, ymin, zmin, tmin], dtype=torch.float32), return_grad=False)

        # position encoding
        self.direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
        )

        # ======== Color Head ======== #
        self.rgb_head = MLP(
            in_dims=feat_dim+self.direction_encoding.n_output_dims,
            out_dims=3,
            num_layers=3,
            hidden_dims=head_mlp_layer_width,
            skip_connections=[1],
        )
        # TODO flow head, shadow head, sky head, feature head
        self.enable_shadow_head = False
        self.enable_sky_head = False
        self.enable_feature_head = False


    def build_representation(self, static_feats, dynamic_feats):
        # static_feats: B, C, D, H, W
        # dynamic_feats: B, C, T, D, H, W
        self.static_feats = static_feats
        self.dynamic_feats = dynamic_feats

    def clear_representation(self):
        self.static_feats = None
        self.dynamic_feats = None

    def get_outputs(self, ray_samples_batch: List[RaySamples],
        return_density_only: bool = False,
        combine_static_dynamic: bool = True,
        query_feature_head: bool = False,
        query_pe_head: bool = False)->dict:
        results_dict = dict()
        positions = []
        times = []
        for ray_samples in ray_samples_batch:
            positions.append(ray_samples.frustums.get_positions())
            times = times.append(ray_samples.times)
        positions = torch.stack(positions, dim=0) # B, N, 3
        times = torch.stack(times, dim=0) # B, N, 1
        normed_positions = self.contract_points(positions)
        normed_times = self.contract_times(times)
        # ======== Static Field ======== #
        geo_feats = interpolation.trilinear_interpolation(self.static_feats, normed_positions) # B, feat_dim, N
        static_density = self.density_activation(geo_feats[..., 0]) # B, N, 1

        # ======== Dynamic Field ======== #
        dynamic_geo_feats = interpolation.quadlinear_interpolation(self.dynamic_feats, torch.cat([normed_positions, normed_times], dim=-1)) # B, feat_dim, N
        dynamic_density = self.density_activation(dynamic_geo_feats[..., 0])
        density = static_density + dynamic_density
        results_dict.update(
            {
                "density": density,
                "static_density": static_density,
                "dynamic_density": dynamic_density,
            }
        )  
        if return_density_only:
            # skip querying other heads
            return results_dict
        directions = []
        for ray_samples in ray_samples_batch:
            directions.append(ray_samples.frustums.directions)
        directions = torch.stack(directions, dim=0) # B, N, 3
        rgb_results = self.query_rgb(
            directions, geo_feats, dynamic_geo_feats)
        results_dict["dynamic_rgb"] = rgb_results["dynamic_rgb"]
        results_dict["static_rgb"] = rgb_results["rgb"]
        if combine_static_dynamic:
            static_ratio = static_density / (density + 1e-6)
            dynamic_ratio = dynamic_density / (density + 1e-6)
            results_dict["rgb"] = (
                static_ratio[..., None] * results_dict["static_rgb"]
                + dynamic_ratio[..., None] * results_dict["dynamic_rgb"]
            )

        # shadow TODO
        if self.enable_shadow_head:
            shadow_ratio = self.shadow_head(dynamic_geo_feats)
            results_dict["shadow_ratio"] = shadow_ratio
            if combine_static_dynamic and "rgb" in results_dict:
                results_dict["rgb"] = (
                    static_ratio[..., None]
                    * results_dict["rgb"]
                    * (1 - shadow_ratio)
                    + dynamic_ratio[..., None] * results_dict["dynamic_rgb"])
        if self.enable_sky_head:
            pass

        if self.enable_feature_head and query_feature_head:
            pass
        return results_dict

    def query_rgb(
        self,
        directions: Tensor,
        geo_feats: Tensor,
        dynamic_geo_feats: Tensor = None,
    ) -> Tensor:
        directions = (directions + 1.0) / 2.0  # do we need this?
        h = self.direction_encoding(directions.reshape(-1, directions.shape[-1])).view(
            *directions.shape[:-1], -1
        )
        rgb = self.rgb_head(torch.cat([h, geo_feats], dim=-1))
        rgb = F.sigmoid(rgb)
        results = {"rgb": rgb}
        if dynamic_geo_feats is not None:
            dynamic_rgb = self.rgb_head(torch.cat([h, dynamic_geo_feats], dim=-1))
            dynamic_rgb = F.sigmoid(dynamic_rgb)
            results["dynamic_rgb"] = dynamic_rgb
        return results

    def contract_times(self, times):
        """
        contract [-inf, inf] points to the range [-1, 1] for quadlinear encoding
        positions: B, N, 1

        Returns:
            normed_times: [..., 1] in [-1, 1]
        """
        aabb_min = self.aabb_min[3:4].unsqueeze(0)
        aabb_max = self.aabb_max[3:4].unsqueeze(0)
        normed_times = (times - aabb_max) / (aabb_max - aabb_min) * 2 - 1
        return normed_times

    def contract_points(self, positions):
        """
        contract [-inf, inf] points to the range [-1, 1] for quadlinear encoding
        positions: B, N, 4

        Returns:
            normed_positions: [..., 3] in [-1, 1]
        """
        # use infinte norm to contract the positions for cuboid aabb
        aabb_min = self.aabb_min[:3].unsqueeze(0)
        aabb_max = self.aabb_max[:3].unsqueeze(0)
        normed_positions = (positions - aabb_min) / (aabb_max - aabb_min) * 2 - 1
        return normed_positions




# Reference to EmerNerf
@MODELS.register_module()
class EmerNerfHead(BaseTaskHead):
    def __init__(self,
                 feat_dim,
                 static_mapper,
                 dynamic_mapper,
                 aabb,
                 head_mlp_layer_width=64,
                 render_chunk_size=2**24,
                 ray_sampler=None):
        super(EmerNerfHead, self).__init__()
        self.feat_dim = feat_dim
        self.render_chunk_size = render_chunk_size
        self.static_mapper = MODELS.build(**static_mapper)
        self.dynamic_mapper = MODELS.build(**dynamic_mapper)
        self.field = DynamicVoxLinearInterpolationField(
            feat_dim, head_mlp_layer_width=head_mlp_layer_width, aabb=aabb)
        self.ray_sampler = RaySampler(**ray_sampler)

    def get_feats(self, feat3ds):
        return (self.static_mapper(feat3ds), self.dynamic_mapper(feat3ds))

    def get_ray_samples_batch(self, metas):
        meta = metas["gt"]
        n_imgs = meta["n_imgs"] # B, N_gt

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
            self, 
            representation, 
            metas=None,
            **kwargs):
        self.field.build_representation(*self.get_feats(representation))
        ray_samples_batch = self.get_ray_samples_batch(metas)
        rst = self.field.get_outputs(ray_samples_batch)
        return self.render(rst, ray_samples_batch)


    def render(self, rst_batch, ray_samples_batch)->dict:
        B = len(ray_samples_batch)
        out_rsts = []
        for i in range(B):
            rst = dict()
            for key, value in rst_batch.items():
                rst[key] = value[i]
            ray_samples = ray_samples_batch[i]
            out_rsts.append(self.rendering(rst, ray_samples))
        return default_collate_fn_map(out_rsts)

    def rendering(self, results, ray_samples)->dict:
        t_starts = ray_samples.frustums.starts.squeeze(-1)
        t_ends = ray_samples.frustums.ends.squeeze(-1)
        num_rays = len(t_starts)
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
        return results_dict


    def prepare(
            self, 
            representation, 
            metas=None, 
            **kwargs):
        assert False

    @torch.cuda.amp.autocast(enabled=False)
    def forward_occ(
            self, 
            representation, 
            metas=None,
            **kwargs):
        assert False