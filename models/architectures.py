import torch
import torch.nn as nn
from lib.pointops.functions import pointops
import torch.nn.functional as F

import os
import sys
import numpy as np
from blocks import *

class SceneSegNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_in = 0
        if config.with_intensity:
            d_in += 1 
        d_out = config.d_out_initial 
        n_cls = config.num_classes
        nsample = config.nsample
        stride_list = config.strides
        stride = 1
        stride_idx = 0
        d_prev = d_in 
        level = 0

        # construct encoder 
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        layer_ind_major = 0
        layer_ind_sub = -1
        
        for block_i, block_name in enumerate(config.architecture):
            layer_ind_sub += 1
            # Detect change to next layer for skip connection
            if np.any([tmp in block_name for tmp in ['strided', 'downsample']]):
                self.encoder_skip_dims.append(d_prev)
                self.encoder_skips.append(block_i)
                layer_ind_major += 1
                layer_ind_sub = 0 
                level += 1

            # Detect upsampling block to stop
            if 'upsample' in block_name:
                break

            # update feature dim            
            d_in = d_prev 
            # if subsample
            if 'strided' in block_name or 'downsample' in block_name:
                stride = stride_list[stride_idx]
                stride_idx += 1
                d_out *= 2
            else:
                stride = 1
            
            # stack modules
            if 'unit' in block_name:
                nsample = config.nsample
            else:
                nsample = config.nsample_conv
            if level == 0:
                nsample = nsample # for dd_partseg 

            layer_ind = f'e_{str(layer_ind_major)}_{str(layer_ind_sub)}'
            self.encoder_blocks.append(
                block_decider(block_name)(
                    d_in, d_out, nsample, stride, config, layer_ind 
                )
            )

            d_prev = d_out


        # construct decoder 
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsample = []

        # Find first upsampling block
        start_i = 0
        for block_i, block_name in enumerate(config.architecture):
            if 'upsample' in block_name:
                start_i = block_i
                break

        # Loop over consecutive blocks
        layer_ind_major = 0
        layer_ind_sub = -1 
        for block_i, block_name in enumerate(config.architecture[start_i:]):
            layer_ind_sub += 1

            d_in = d_out

            # detect the upsample layer
            if 'upsample' in block_name:
                layer_ind_major += 1
                layer_ind_sub = 0
                level -= 1

                self.decoder_upsample.append(block_i)
                
                # if upsample, out_dim / 2 
                d_out = max(d_out // 2, config.decoder_out_dim)                

                # Apply the good block function defining tf ops
                if 'unit' in block_name:
                    nsample = config.nsample
                else:
                    nsample = config.nsample_conv
                if level == 0:
                    nsample = nsample # for dd_partseg 

                layer_ind = f'd_{str(layer_ind_major)}_{str(layer_ind_sub)}'
                self.decoder_blocks.append(
                    block_decider(block_name)(
                        [d_in, self.encoder_skip_dims.pop()], 
                        d_out, nsample, stride, config, layer_ind 
                    )
                )
            else:
                # if not upsample, then dim remain same
                if 'unit' in block_name:
                    nsample = config.nsample
                else:
                    nsample = config.nsample_conv
                if level == 0:
                    nsample = nsample 

                layer_ind = f'd_{str(layer_ind_major)}_{str(layer_ind_sub)}'
                self.decoder_blocks.append(
                    block_decider(block_name)(
                        d_in, d_in, nsample, stride, config, layer_ind 
                    )
                )
         
        # classification layers
        self.classifier = nn.Sequential(
            nn.Linear(d_out, d_out), 
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(d_out, n_cls)
        )        
    
    def forward(self, p, x, o, save_path=None):
        # p, x, o, one_hot: (n,3), (n,c), (b), (b,16)
        p_from_encoder = []
        x_from_encoder = []
        o_from_encoder = []
        idx = None

        # encoder
        for block_i, block in enumerate(self.encoder_blocks):
            idx = None
            if block_i in self.encoder_skips:
                p_from_encoder.append(p) 
                x_from_encoder.append(x)
                o_from_encoder.append(o)
            p, x, o, idx = block(p, x, o, idx, save_path=save_path)

                     
        # decoder
        idx = None
        for block_i, block in enumerate(self.decoder_blocks):
            idx = None
            if block_i in self.decoder_upsample:
                p_dense = p_from_encoder.pop()
                x_dense = x_from_encoder.pop()
                o_dense = o_from_encoder.pop()
                p, x, o, idx = block(p_dense, x_dense, o_dense, p, x, o, idx, save_path=save_path) 
            else:
                p, x, o, idx = block(p, x, o, idx, save_path=save_path) 
            
        # classification     
        x = self.classifier(x)
        return x




