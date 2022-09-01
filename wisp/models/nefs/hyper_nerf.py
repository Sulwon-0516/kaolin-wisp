# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging as log
import time
import math
import copy

from wisp.ops.spc import sample_spc
from wisp.utils import PsDebugger, PerfTimer
from wisp.ops.geometric import sample_unif_sphere

from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders import get_positional_embedder
from wisp.accelstructs import OctreeAS
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.models.decoders import BasicDecoder, SE3Decoder, SurfDecoder
from wisp.models.grids import *


import kaolin.ops.spc as spc_ops


USE_SEPARATE_GRID = True
USE_DIFF_COLOR_DECODER = False
DIM_APPEARANCE = 8
DIM_LATENT = 8

# DeformNet
DEPTH_DEFORM = 2
WIDTH_DEFORM = 128
SE3_EXP_DIM = 6

# Surface Modify Net
DEPTH_SURF = 2
WIDTH_SURF = 64





class HyperNeuralRadianceField(BaseNeuralField):
    """Model for encoding radiance fields (density and plenoptic color)
    """
    def __init__(self, 
        grid_type          : str = 'OctreeGrid',
        interpolation_type : str = 'trilinear',
        multiscale_type    : str = 'none',

        as_type            : str = 'octree',
        raymarch_type      : str = 'voxel',

        decoder_type       : str = 'none',
        embedder_type      : str = 'none', 
        activation_type    : str = 'relu',
        layer_type         : str = 'none',

        base_lod         : int   = 2,
        num_lods         : int   = 1, 

        # grid args
        sample_tex       : bool  = False,
        dilate           : int   = None,
        feature_dim      : int   = 16,

        # decoder args
        hidden_dim       : int   = 128,
        pos_multires     : int   = 10,
        view_multires    : int   = 4,
        num_layers       : int   = 1,
        position_input   : bool  = False,

        # num_images for feasture
        num_images       : int   = 0,
        **kwargs
    ):
    # I just copied and little modified the __init__ functions.
        super().__init__()

        self.grid_type = grid_type
        self.interpolation_type = interpolation_type
        self.raymarch_type = raymarch_type
        self.embedder_type = embedder_type
        self.activation_type = activation_type
        self.layer_type = layer_type
        self.decoder_type = decoder_type
        self.multiscale_type = multiscale_type

        self.base_lod = base_lod
        self.num_lods = num_lods

        self.sample_tex = sample_tex
        self.dilate = dilate
        self.feature_dim = feature_dim
        
        self.hidden_dim = hidden_dim
        self.pos_multires = pos_multires
        self.view_multires = view_multires
        self.num_layers = num_layers
        self.position_input = position_input

        self.kwargs = kwargs

        self.grid = None
        self.decoder = None
        
        self.init_grid()
        self.init_embedder()
        self.init_decoder()
        torch.cuda.empty_cache()
        self._forward_functions = {}
        self.register_forward_functions()
        self.supported_channels = set([channel for channels in self._forward_functions.values() for channel in channels])

        if num_images == 0:
            print("default # of input images")
            assert(0)
        else:
            self.num_imagse = num_images
    

        # set it as parameters
        self.appearance_param = nn.Embedding(
            num_embeddings=self.num_imagse,
            embedding_dim=DIM_APPEARANCE
        )
        self.latent_param = nn.Parameter(
            num_embeddings=self.num_imagse,
            embedding_dim=DIM_LATENT
        )

        # initialize embeddings
        nn.init.uniform_(self.appearance_param.weight, a=0.0, b=0.05)
        nn.init.uniform_(self.latent_param.weight, a=0.0, b=0.05)


    def init_embedder(self):
        """Creates positional embedding functions for the position and view direction.
        """
        self.pos_embedder, self.pos_embed_dim = get_positional_embedder(self.pos_multires, 
                                                                       self.embedder_type == "positional")
        self.view_embedder, self.view_embed_dim = get_positional_embedder(self.view_multires, 
                                                                         self.embedder_type == "positional")
        log.info(f"Position Embed Dim: {self.pos_embed_dim}")
        log.info(f"View Embed Dim: {self.view_embed_dim}")

    def init_decoder(self):
        """Initializes the decoder object. 
        """
        if self.multiscale_type == 'cat':
            self.effective_feature_dim = self.grid.feature_dim * self.num_lods
        else:
            self.effective_feature_dim = self.grid.feature_dim

        self.input_dim = self.effective_feature_dim + DIM_LATENT

        if self.position_input:
            self.input_dim += self.pos_embed_dim

        # original final decoder (template decoder)
        self.decoder = BasicDecoder(
            input_dim=self.input_dim, 
            output_dim=1, 
            activation=get_activation_class(self.activation_type), 
            bias=True,
            layer=get_layer_class(self.layer_type), 
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim, 
            skip=[])

        # additional color layer
        if USE_DIFF_COLOR_DECODER:
            self.rgb_decode = nn.Linear(
                in_features=self.hidden_dim + self.view_embed_dim + DIM_APPEARANCE,
                out_features=3
            )

        # deformation encoder (input encoder)
        self.deform_decoder = SE3Decoder(
            input_dim=self.effective_feature_dim+DIM_LATENT, 
            output_dim=SE3_EXP_DIM, 
            activation=get_activation_class(self.activation_type), 
            bias=True,
            layer=get_layer_class(self.layer_type), 
            num_layers=self.num_layers,
            hidden_dim=WIDTH_DEFORM, 
            skip=[])

        # deformation encoder (surface plane encoder)
        self.surf_decoder = BasicDecoder(
            input_dim=self.effective_feature_dim+DIM_LATENT, 
            output_dim=DIM_LATENT, 
            activation=get_activation_class(self.activation_type), 
            bias=True,
            layer=get_layer_class(self.layer_type), 
            num_layers=self.num_layers,
            hidden_dim=WIDTH_SURF, 
            skip=[])

    def init_grid(self):
        """Initialize the grid object.
        """
        if self.grid_type == "OctreeGrid":
            grid_class = OctreeGrid
        elif self.grid_type == "CodebookOctreeGrid":
            grid_class = CodebookOctreeGrid
        elif self.grid_type == "TriplanarGrid":
            grid_class = TriplanarGrid
        elif self.grid_type == "HashGrid":
            grid_class = HashGrid
        else:
            raise NotImplementedError

        self.grid = grid_class(self.feature_dim,
                               base_lod=self.base_lod, num_lods=self.num_lods,
                               interpolation_type=self.interpolation_type, multiscale_type=self.multiscale_type,
                               **self.kwargs)

        self.deform_grid = grid_class(self.feature_dim,
                               base_lod=self.base_lod, num_lods=self.num_lods,
                               interpolation_type=self.interpolation_type, multiscale_type=self.multiscale_type,
                               **self.kwargs)


        if USE_SEPARATE_GRID:
            self.surface_grid = grid_class(self.feature_dim,
                               base_lod=self.base_lod, num_lods=self.num_lods,
                               interpolation_type=self.interpolation_type, multiscale_type=self.multiscale_type,
                               **self.kwargs)
        else:
            self.surface_grid = self.deform_grid

    def prune(self):
        """Prunes the blas based on current state.
        """
        grids = [self.grid, self.deform_grid]
        if USE_SEPARATE_GRID:
            grids.append(self.surface_grid)

        for i_grid in grids:
            if i_grid is not None:
                
                if self.grid_type == "HashGrid":
                    # TODO(ttakikawa): Expose these parameters. 
                    # This is still an experimental feature for the most part. It does work however.
                    density_decay = 0.6
                    min_density = ((0.01 * 512)/np.sqrt(3))

                    i_grid.occupancy = i_grid.occupancy.cuda()
                    i_grid.occupancy = i_grid.occupancy * density_decay
                    points = i_grid.dense_points.cuda()
                    #idx = torch.randperm(points.shape[0]) # [:N] to subsample
                    res = 2.0**i_grid.blas_level
                    samples = torch.rand(points.shape[0], 3, device=points.device)
                    samples = points.float() + samples
                    samples = samples / res
                    samples = samples * 2.0 - 1.0
                    sample_views = torch.FloatTensor(sample_unif_sphere(samples.shape[0])).to(points.device)
                    with torch.no_grad():
                        density = self.forward(coords=samples[:,None], ray_d=sample_views, channels="density")
                    i_grid.occupancy = torch.stack([density[:, 0, 0], i_grid.occupancy], -1).max(dim=-1)[0]

                    mask = i_grid.occupancy > min_density
                    
                    #print(density.mean())
                    #print(density.max())
                    #print(mask.sum())
                    #print(self.grid.occupancy.max())

                    _points = points[mask]
                    octree = spc_ops.unbatched_points_to_octree(_points, i_grid.blas_level, sorted=True)
                    i_grid.blas.init(octree)
                else:
                    raise NotImplementedError

    def get_nef_type(self):
        """Returns a text keyword of the neural field type.

        Returns:
            (str): The key type
        """
        return 'hyper_nerf'

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.rgba, ["density", "rgb"])

    def rgba(self, coords, ray_d, step, img_idx, pidx=None, lod_idx=None):
        """Compute color and density [particles / vol] for the provided coordinates.

        Args:
            coords (torch.FloatTensor): packed tensor of shape [batch, num_samples, 3]
            ray_d (torch.FloatTensor): packed tensor of shape [batch, 3]
            pidx (torch.LongTensor): SPC point_hierarchy indices of shape [batch].
                                     Unused in the current implementation.
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
        
        Returns:
            {"rgb": torch.FloatTensor, "density": torch.FloatTensor}:
                - RGB tensor of shape [batch, num_samples, 3] 
                - Density tensor of shape [batch, num_samples, 1]
        """
        timer = PerfTimer(activate=False, show_memory=True)
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, num_samples, _ = coords.shape
        timer.check("rf_rgba_preprocess")
        
        if batch != 1:
            print("currrent batch size : ", batch)
            print("only single batch can be handled now")
            assert(0)

        ##########################
        # First, apply deform plane
        def_feats = self.deform_grid.interpolate(coords, lod_idx).reshape(-1, self.effective_feature_dim)
        timer.check("deform_grid_interpolate")

        # As original HyperNeRF includes "residual connection", here I added residual inputs
        fdir = torch.cat([
            def_feats,
            self.def_pos_embedder(coords.reshape(-1, 3), step),
            self.latent_param[img_idx]
            ],dim = -1)
        
        se3_deforms = self.deform_decoder(fdir)
        timer.check("deform_decoder")
        deformed_coords = se3_deforms(se3_deforms, coords)
        ###############################

        ##########################
        # Second, change surface plane
        if USE_SEPARATE_GRID:
            surf_feats = self.deform_grid.interpolate(coords, lod_idx).reshape(-1, self.effective_feature_dim)
        else:
            surf_feats = def_feats
        timer.check("surface_grid_interpolate")

        # As original HyperNeRF includes "residual connection", here I added residual inputs
        fdir = torch.cat([
            surf_feats,
            self.surf_pos_embedder(coords.reshape(-1, 3), step),
            self.latent_param[img_idx]
            ],dim = -1)
        
        deformed_inputs = self.surf_decoder(fdir)
        timer.check("surface_decoder")
        ###############################

        ##########################
        # Finally, apply template network
        # Embed coordinates into high-dimensional vectors with the grid.
        feats = self.grid.interpolate(deformed_coords, lod_idx).reshape(-1, self.effective_feature_dim)
        timer.check("template_grid_interpolate")

        # As original HyperNeRF includes "residual connection", here I added residual inputs
        fdir = torch.cat([
            feats,
            self.nerf_pos_embedder(coords.reshape(-1, 3), step),
            self.latent_param[img_idx],
            self.appearance_param[img_idx]
            ],dim = -1)
        timer.check("rf_rgba_embed_cat")
        
        # Decode high-dimensional vectors to RGBA.
        rgba = self.decoder(fdir)
        timer.check("rf_rgba_decode")

        # Colors are values [0, 1] floats
        colors = torch.sigmoid(rgba[...,:3]).reshape(batch, num_samples, -1)

        # Density is [particles / meter], so need to be multiplied by distance
        density = torch.relu(rgba[...,3:4]).reshape(batch, num_samples, -1)
        timer.check("rf_rgba_activation")
        
        return dict(rgb=colors, density=density)



'''
(08/25)
I just followed HyperNeRF-torch's implementation here.
To be accurate, I need to do double-check & re-implmenet here.

'''
def se3deforms(se3_inputs, coords):
    print("We need to check here (se3 deformation)")
    w, v = se3_inputs[...,:3], se3_inputs[...,3:]

    theta = torch.norm(w, dim=-1)
    w = w / theta.unsqueeze(-1)
    v = v / theta.unsqueeze(-1)
    screw_axis = torch.cat([w, v], dim=-1)
    transform = exp_se3(screw_axis, theta)

    warped_points = from_homogenous(
        torch.matmul(transform, to_homogenous(coords)))

    return warped_points

'''
below codes are from unofficial HyperNeRF implementations
https://github.com/songrise/HyperNeRF-torch/blob/7b9650fa3c0d5a28371bfeba7d3a37b2c913e573/hypernerf/rigid_body.py#L59
'''
from torch import matmul

def rp_to_se3(r, p):
    """Build a SE3 matrix from a rotation matrix and a translation vector.
    Args:
        r: (3, 3) A rotation matrix
        p: (3,) A translation vector
    Returns:
        T: (4, 4) A homogeneous transformation matrix
    """
    p = p.view(3,1)
    up =  torch.cat([r, p], dim=1)
    lower = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).cuda()
    return torch.cat([up, lower], dim=0)


def skew(w):
    """Build a skew matrix ("cross product matrix") for vector w.
    Modern Robotics Eqn 3.30.
    Args:
        w: (3,) A 3-vector
    Returns:
        W: (3, 3) A skew matrix such that W @ v == w x v
    """
    w = w.view(3)
    return torch.tensor([[0, -w[2], w[1]],
                        [w[2], 0.0, -w[0]],
                        [-w[1], w[0], 0.0]], dtype=torch.float32).cuda()

def exp_so3(w, theta):
    w = skew(w)
    return torch.eye(3).cuda() + torch.sin(theta) * w + (1.0-torch.cos(theta)) * (w @ w)

def exp_se3(S, theta):
    """Exponential map from Lie algebra so3 to Lie group SO3.
    Modern Robotics Eqn 3.88.
    Args:
        S: (,6) A screw axis of motion.
        theta: Magnitude of motion.
    Returns:
        a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
        motion of magnitude theta about S for one second.
    """
    w,v = S[...,:3], S[...,3:]
    v = v.squeeze(0)#todo check
    W = skew(w)
    R = exp_so3(w, theta).cuda()
    a = (theta * torch.eye(3).cuda() + (1.0 - torch.cos(theta)) * W +
              (theta - torch.sin(theta)) * matmul(W, W))

    # reshape v to match the shape
    v = v.view(v.shape[1], v.shape[0])
    p = matmul((theta * torch.eye(3).cuda() + (1.0 - torch.cos(theta)) * W +
              (theta - torch.sin(theta)) * matmul(W, W)), v)
    return rp_to_se3(R, p)

def to_homogenous(v):
    ones = torch.ones_like(v[..., :1]).cuda()
    #convert from (N,1,4) to (4,N)
    res = torch.cat([v, ones], dim=-1).squeeze(0)
    return res.reshape(res.shape[1], res.shape[0])

def from_homogenous(v):

    return v[..., :3] / v[..., -1:]