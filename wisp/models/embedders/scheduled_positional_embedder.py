# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import torch
import torch.nn as nn
from .schedulers import *

# original NeRF scheduling
hyper_point_max_deg = 1
hyper_alpha_schedule = {
  'type': 'piecewise',
  'schedules': [
    (1000, ('constant', 0.0)),
    (0, ('linear', 0.0, hyper_point_max_deg, 10000))
  ],
}


# slice 
hyper_slice_max_deg = 6
slice_alpha_schedule = ('constant', hyper_slice_max_deg)

# deform 
warp_min_deg = 0
warp_max_deg = 4
warp_alpha_schedule = {
  'type': 'linear',
  'initial_value': warp_min_deg,
  'final_value': warp_max_deg,
  'num_steps': 50000,
}

class ScheduledPositionalEmbedder(nn.Module):
    """PyTorch implementation of positional embedding.
    """
    def __init__(
        self, 
        num_freq, 
        max_freq_log2, 
        log_sampling=True, 
        include_input=False,            # in hypernerf. we doesn't include input point (no identity concat) 
        input_dim=3,
        scheduler=None
        ):
        """Initialize the module.

        Args:
            num_freq (int): The number of frequency bands to sample. 
            max_freq_log2 (int): The maximum frequency. The bands will be sampled between [0, 2^max_freq_log2].
            log_sampling (bool): If true, will sample frequency bands in log space.
            include_input (bool): If true, will concatenate the input.
            input_dim (int): The dimension of the input coordinate space.

        Returns:
            (void): Initializes the encoding.
        """
        super().__init__()

        self.num_freq = num_freq
        self.max_freq_log2 = max_freq_log2
        self.log_sampling = log_sampling
        self.include_input = include_input
        self.out_dim = 0
        if include_input:
            self.out_dim += input_dim

        if self.log_sampling:
            self.bands = 2.0**torch.linspace(0.0, max_freq_log2, steps=num_freq)
        else:
            self.bands = torch.linspace(1, 2.0**max_freq_log2, steps=num_freq)
        

        # The out_dim is really just input_dim + num_freq * input_dim * 2 (for sin and cos)
        self.out_dim += self.bands.shape[0] * input_dim * 2
        self.bands = nn.Parameter(self.bands).requires_grad_(False)
        
        self.scheduler_type = scheduler
        if isinstance(scheduler, type(None)):
            print("no scheduler on positional encoder")
        elif scheduler == 'template':
            self.scheduler = from_dict(hyper_alpha_schedule)
        elif scheduler == 'slice':
            self.scheduler = from_dict(slice_alpha_schedule)
        elif scheduler == 'deform':
            self.scheduler = from_dict(warp_alpha_schedule)
        else:
            print("wront scheduler type for masking on positional encoding")
            print("use no mask")
            self.scheduler_type = None
        
    
    def forward(self, coords, step):
        """Embded the coordinates.

        Args:
            coords (torch.FloatTensor): Coordinates of shape [N, input_dim]

        Returns:
            (torch.FloatTensor): Embeddings of shape [N, input_dim + out_dim] or [N, out_dim].
        """
        N = coords.shape[0]
        winded = coords[:,None] * self.bands[None,:,None]                          # (N, 4, 3)
        encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)        # (N, 4, 6)

        encoded = encoded * self.get_screen(step).unsqueeze(-1)
        encoded = encoded.reshape(N,-1)
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)
        return encoded

    def get_screen(self, step):
        """Get screen window of coordinate

        Args:
            step : step of the function (n_iteration)
        Returns:
            (torch.FloatTensor): screen of shape [1, out_dim]
        
        Here I don't care about (# of samples). You should handle it in forward().
        """
        if isinstance(self.scheduler_type, type(None)):
            return torch.ones_like(self.bands, dtype=torch.float32).reshape(1, -1)

        alpha = self.scheduler.get(step)
        window = torch.linspace(0.0, self.num_freq-1, steps=self.num_freq)
        window = (1-torch.cos(torch.pi * min(max(alpha-window,0.0),1.0)))/2.0
        window = window.view(1, -1)

        return window




def get_scheduled_positional_embedder(multires, input_dim=3):
    """Utility function to get the embedding from the NeRF-style multires input.

    Thin wrapper function

    Args:
        multires (int): The multires.
        input_dim (int): The input coordinate dimension.

    Returns:
        (nn.Module, int):
        - The embedding module.
        - The output dimension of the encoder.
    """

    encoder = ScheduledPositionalEmbedder(multires, multires-1, input_dim=input_dim)
    return encoder, encoder.out_dim
