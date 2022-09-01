# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import ortho_group
from wisp.models.decoders.basic_decoders import BasicDecoder

ROT_FCL_WIDTH = 128
TRANS_FCL_WIDTH = 128



class OrgSE3Decoder(nn.Module):
    """SE3 Decoder with specific initialization
    """
    def __init__(self, 
        input_dim, 
        activation,
        output_dim = 6,
        bias = True,
        layer = nn.Linear,
        num_layers = 6, 
        hidden_dim = 128, 
        skip       = [5]
    ):
        """Initialize the BasicDecoder.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.

        Returns:
            (void): Initializes the class.
        """

        # Currently, it only decodes 
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim        
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip = skip
        if self.skip is None:
            self.skip = []
        
        self.make()
        self.initialize()



    def make(self):
        """Builds the actual MLP.
        """
        # I algined the concatenation of input in same direction
        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(self.layer(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim+self.input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)

        self.r_net = self.layer(self.hidden_dim, self.output_dim//2, bias=self.bias)
        self.t_net = self.layer(self.hidden_dim, self.output_dim//2, bias=self.bias)

    def forward(self, x, return_h=False):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        N = x.shape[0]

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i in self.skip:
                h = torch.cat([x, h], dim=-1)           #### Original code has issues here
                h = self.activation(l(h))
            else:
                h = self.activation(l(h))
        
        r_out = self.r_net(h)
        t_out = self.t_net(h)

        out = torch.cat([r_out, t_out], dim=-1)
        
        if return_h:
            return out, h
        else:
            return out

    def initialize(self):
        """Initializes the MLP layers with some initialization functions.

        Args:
            get_weight (function): A function which returns a matrix given a matrix.

        Returns:
            (void): Initializes the layer weights.
        """
        
        # initialize with default xavier uniform
        for i, w in enumerate(self.layers):
            torch.nn.init.xavier_uniform(w.weight)
        
        # initialize rot / trans
        torch.nn.init.uniform_(self.r_net.weight, a=0, b=1e-4)
        torch.nn.init.uniform_(self.t_net.weight, a=0, b=1e-4)
        



class OrgSliceDecoder(nn.Module):
    """Super basic but super useful MLP class.
    """
    def __init__(self, 
        input_dim, 
        output_dim, 
        activation,
        bias,
        layer = nn.Linear,
        num_layers = 7, 
        hidden_dim = 64, 
        skip       = [5]
    ):
        """Initialize the BasicDecoder.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim        
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip = skip
        if self.skip is None:
            self.skip = []
        
        self.make()
        self.initialize()

    def make(self):
        """Builds the actual MLP.
        """
        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(self.layer(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim+self.input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.lout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)

    def forward(self, x, return_h=False):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        N = x.shape[0]

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i in self.skip:
                h = torch.cat([x, h], dim=-1)
                h = self.activation(l(h))
            else:
                h = self.activation(l(h))
        
        out = self.lout(h)
        
        if return_h:
            return out, h
        else:
            return out

    def initialize(self):
        """Initializes the MLP layers with some initialization functions.

        Args:
            get_weight (function): A function which returns a matrix given a matrix.

        Returns:
            (void): Initializes the layer weights.
        """
        
        # initialize with default xavier uniform
        for i, w in enumerate(self.layers):
            torch.nn.init.xavier_uniform(w.weight)
        
        # initialize rot / trans
        torch.nn.init.uniform_(self.lout.weight, a=0, b=1e-5)



class OrgNGPDecoder(nn.Module):
    """The NGP decoder is little different from original kaolin-wisp, so here I reimplemente it.
    """
    def __init__(self, 
        input_dim,
        direction_dim, 
        activation,
        bias = True,
        layer = nn.Linear,
        num_layers = 8, 
        hidden_dim = 256, 
        skip       = [5]
    ):
        """Initialize the BasicDecoder.

        Args:
            input_dim (int): Input dimension of the MLP.
            direction_dim (int) : The additional dimensions should be concatenated to MLP (the last layers) -> It's main difference
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        
        self.input_dim = input_dim     
        self.direction_dim = direction_dim
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip = skip
        if self.skip is None:
            self.skip = []
        
        self.make()
        self.initialize()

    def make(self):
        """Builds the actual MLP.
        """
        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(self.layer(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim+self.input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.lout = self.layer(self.hidden_dim, 1, bias=self.bias)

        self.color = self.layer(self.hidden_dim + self.direction_dim, self.hidden_dim//2, bias=self.bias)
        self.color_out = self.layer(self.hidden_dim//2, 3, bias=self.bias)

    def forward(self, x, color_latent, return_h=False):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        N = x.shape[0]

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i in self.skip:    
                h = torch.cat([x, h], dim=-1)
                h = self.activation(l(h))
            else:
                h = self.activation(l(h))
        
        out = self.lout(h)

        sigma = out[:,0:1]
        color_feature = out[:,1:]
        color_input = torch.cat([
            color_feature,
            color_latent
        ], dim=-1)

        rgb = self.color(color_input)

        # Kaolin framework requires Batch x 4 returns (rgba)
        # It returns the raw output & handling the scale at the output
        out = torch.cat([
            rgb,
            sigma
        ], dim=-1)
        
        if return_h:
            return out, h
        else:
            return out

    def initialize(self, get_weight):
        """Initializes the MLP layers with some initialization functions.

        Args:
            get_weight (function): A function which returns a matrix given a matrix.

        Returns:
            (void): Initializes the layer weights.
        """
        
        # initialize with default xavier uniform
        for i, w in enumerate(self.layers):
            torch.nn.init.xavier_uniform(w.weight)
        
        # initialize lout
        torch.nn.init.uniform_(self.lout.weight, a=0, b=1e-5)
        torch.nn.init.uniform_(self.color_out.weight, a=0, b=1e-5)

        # initialize color mlp
        torch.nn.init.xavier_uniform_(self.color.weight)
        torch.nn.init.uniform_(self.lout.weight, a=0, b=1e-5)
