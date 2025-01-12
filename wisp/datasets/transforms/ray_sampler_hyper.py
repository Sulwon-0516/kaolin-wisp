# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch


class HyperSampleRays:
    """ A dataset transform for sub-sampling a fixed amount of rays. """
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, inputs):
        ray_idx = torch.randperm(
            inputs['imgs'].shape[0],
            device=inputs['imgs'].device)[:self.num_samples]

        out = {}
        out['rays'] = inputs['rays'][ray_idx].contiguous()
        out['imgs'] = inputs['imgs'][ray_idx].contiguous()
        out['idx'] = inputs['idx']
        return out
