# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Network architectures from the paper
"Analyzing and Improving the Image Quality of StyleGAN".
Matches the original implementation of configs E-F by Karras et al. at
https://github.com/NVlabs/stylegan2/blob/master/training/networks_stylegan2.py"""

import numpy as np
import torch
import math
from torch import nn
from .networks_styleunet import DecoderBlock_new, Conv2dLayer
from functools import partial
from .mmseg.mix_transformer import MLP, MixVisionTransformer, transformer_block
from .helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
from .unet_encoders import recurrent_Up, Up, DoubleConv, ConvGRU


class CondSynthesisNetwork_vitEnc(torch.nn.Module):
    def __init__(self,
                 img_resolution,  # Output image resolution.
                 img_channels,  # Number of color channels.
                 cond_channels=3,  # Number of condition input channels.
                 in_size=64,  # Input size of condition image.
                 final_size=4,  # Lowest resolution of encoding.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=4,  # Use FP16 for the N highest resolutions.
                 num_cond_res=64,  # Highest resolution of condition injection.
                 use_gru=False,
                 **block_kwargs,  # Arguments for SynthesisBlock.
                 ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.cond_channels = cond_channels
        self.in_size = in_size
        self.final_size = final_size
        self.final_size_log2 = int(np.log2(final_size))
        self.block_resolutions = [2 ** i for i in range(self.final_size_log2 + 1, self.img_resolution_log2 + 1)]
        self.num_cond_res = num_cond_res
        self.use_gru = use_gru
        channels_dict = {res: min(channel_base // res, channel_max) for res in ([final_size] + self.block_resolutions)}

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            block = DecoderBlock_new(img_channels, in_channels, out_channels, cond_channel=out_channels if res < self.img_resolution else 0,
                                     use_gru=use_gru, architecture='skip' if res==32 else 'orig')
            setattr(self, f'b{res}', block)

        # encoder
        self.encoder_res = [2 ** i for i in range(int(np.log2(self.in_size)), int(np.log2(self.final_size)) - 1, -1)]
        # self.vit_encoder = MixVisionTransformer(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
        #     qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        #     drop_rate=0.0, drop_path_rate=0.1, in_chans=7)  # mit_b3
        # self.vit_encoder = MixVisionTransformer(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
        #     qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        #     drop_rate=0.0, drop_path_rate=0.1, in_chans=7)  # mit_b4
        self.vit_encoder = MixVisionTransformer(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, in_chans=7)  # mit_b5
        self.final_encoder_layer = Conv2dLayer(channels_dict[final_size*2], channels_dict[final_size], 3, down=2, activation='lrelu')
        embed_dims = [64, 128, 320, 512]
        self.linear_128 = MLP(input_dim=sum(embed_dims)//4, embed_dim=channels_dict[128])
        self.linear_64 = MLP(input_dim=sum(embed_dims[1:])//4, embed_dim=channels_dict[64])
        self.linear_32 = MLP(input_dim=sum(embed_dims[2:])//4, embed_dim=channels_dict[32])
        self.linear_16 = MLP(input_dim=sum(embed_dims[3:])//4, embed_dim=channels_dict[16])
        self.pixel_shuffle = torch.nn.PixelShuffle(upscale_factor=2)


    def forward(self, x_cond, r_list=None, return_list=False):
        if x_cond.dim() == 5:
            B, T = x_cond.shape[:2]
            x_in = x_cond.flatten(0, 1)
        else:
            x_in = x_cond

        # obtain multi-scale content features
        vit_out = self.vit_encoder(x_in)
        vit_last_out = vit_out[-1]
        last_feat = self.final_encoder_layer(vit_last_out)
        vit_out = [self.pixel_shuffle(o) for o in vit_out]
        cond_list = [None]
        for idx in range(len(vit_out)):
            res = vit_out[idx].shape[-1]
            feat = [vit_out[idx]]
            for yidx in range(idx+1, len(vit_out)):
                feat.append(torch.nn.functional.interpolate(vit_out[yidx], size=(res, res), mode='bilinear'))
            feat = torch.cat(feat, dim=1)
            feat = getattr(self, f'linear_{res}')(feat).permute(0, 2, 1).reshape(feat.shape[0], -1, res, res)
            cond_list.append(feat)
        cond_list.append(vit_last_out)
        cond_list.append(last_feat)
        cond_list = cond_list[::-1]
        x = cond_list[0]
        img = None
        x_list, _index = [], 0
        start_layer = int(np.log2(32)) - self.final_size_log2 - 1  # 从32分辨率开始
        if r_list is None: r_list = [None for _ in range(len(self.block_resolutions))]
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            # print(res, x.requires_grad, block.cond_channel, (cond_list[_index+1].shape, cond_list[_index+1].requires_grad) if cond_list[_index+1] is not None else None)
            x, img = block(x, img, cond_list[_index+1])
            if _index >= start_layer:
                if return_list:
                    if _index == start_layer: x_list.append(img.clone())
                    x_list.append(x.clone())
            _index += 1

        if return_list:
            x_list.append(img)
            if self.use_gru:
                return x_list, r_list
            else:
                return x_list
        else:
            return img, r_list

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])


class TriPlaneSFTfeat_SegformerEncoder(nn.Module):
    def __init__(self, inp_ch, sft_half=True, res=None, use_gru=False):
        super(TriPlaneSFTfeat_SegformerEncoder, self).__init__()
        self.sft_half = sft_half
        self.res = res
        self.use_gru = use_gru
        self.face_pool = None if res is None else torch.nn.AdaptiveAvgPool2d((res, res))

        self.vit_encoder = MixVisionTransformer(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, in_chans=6)  # mit_b5

        embed_dims = [64, 128, 320, 512]
        self.linear_128 = MLP(input_dim=sum(embed_dims[0:2])//4, embed_dim=64)
        self.linear_64 = MLP(input_dim=sum(embed_dims[1:3])//4, embed_dim=128)
        self.linear_32 = MLP(input_dim=sum(embed_dims[2:4])//4, embed_dim=256)
        self.linear_16 = MLP(input_dim=sum(embed_dims[3:5])//4, embed_dim=512)

        self.up0 = nn.Conv2d(24, 96, kernel_size=3, padding=1)

        if use_gru:
            self.up1 = (recurrent_Up(1024, 512, upscale_factor=1))
            self.up2 = (recurrent_Up(384, 384))
            self.up3 = (recurrent_Up(224, 256))
            self.up4 = (recurrent_Up(128, 96))
        else:
            self.up1 = (Up(1024, 512, upscale_factor=1))
            self.up2 = (Up(384, 384))
            self.up3 = (Up(224, 256))
            self.up4 = (Up(128, 96))

        self.head = nn.PixelShuffle(upscale_factor=2)
        self.final_head = nn.Sequential(
            nn.Conv2d(24, 96, kernel_size=3, padding=1),
            nn.PReLU(96),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.PReLU(96)
        )

        self.block_resolutions = [2 ** i for i in range(int(np.log2(16)), int(np.log2(256)) + 1)]
        channels_dict = {res: min(32768 // res, 512) for res in self.block_resolutions}
        body_outchannels_dict = {16: 512, 32: 384, 64: 256, 128: 96, 256: 96}

        for res in self.block_resolutions:
            out_channels = body_outchannels_dict[res]
            sft_out_channels = channels_dict[res] // 2 if self.sft_half else channels_dict[res]
            condition_scale = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1))
            setattr(self, f'condition_scale{res}', condition_scale)
            condition_shift = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1))
            setattr(self, f'condition_shift{res}', condition_shift)

    def forward(self, x, r_list=None):
        if x.dim() == 5:
            B, T = x.shape[:2]
            x = x.flatten(0, 1)
        else:
            T = 1
        if self.face_pool is not None and x.shape[-1] != self.res:
            x = self.face_pool(x)
        x = self.input_layer(x)
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                c0 = x
            if i == 6:
                c1 = x
            if i == 20:
                c2 = x
            elif i == 21:
                c3 = x
        out_list_dict = {}
        if self.use_gru:
            if r_list is None: r_list = [None for _ in range(4)]
            tri_plane, r_list[0] = self.up1(x, c3, T, r_list[0])             # 512*16 + 512*16 -> 512*16
            out_list_dict[16] = torch.stack([getattr(self, f'condition_scale{16}')(tri_plane), getattr(self, f'condition_shift{16}')(tri_plane)])

            tri_plane = tri_plane.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)    # [B, C, H, W] -> [B*T, C, H, W]
            tri_plane, r_list[1] = self.up2(tri_plane, c2, T, r_list[1])     # 512*16 -> 128*32 + 256*32 -> 384*32
            out_list_dict[32] = torch.stack([getattr(self, f'condition_scale{32}')(tri_plane), getattr(self, f'condition_shift{32}')(tri_plane)])

            tri_plane = tri_plane.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
            tri_plane, r_list[2] = self.up3(tri_plane, c1, T, r_list[2])     # 384*32 -> 96*64 + 128*64 -> 256*64
            out_list_dict[64] = torch.stack([getattr(self, f'condition_scale{64}')(tri_plane), getattr(self, f'condition_shift{64}')(tri_plane)])

            tri_plane = tri_plane.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
            tri_plane, r_list[3] = self.up4(tri_plane, c0, T, r_list[3])     # 256*64 -> 64*128 + 64*128 -> 96*128
            out_list_dict[128] = torch.stack([getattr(self, f'condition_scale{128}')(tri_plane), getattr(self, f'condition_shift{128}')(tri_plane)])

            tri_plane = self.final_head(self.head(tri_plane))
            out_list_dict[256] = torch.stack([getattr(self, f'condition_scale{256}')(tri_plane), getattr(self, f'condition_shift{256}')(tri_plane)])

            return out_list_dict, r_list
        else:
            tri_plane = self.up1(x, c3)  # 512*16 + 512*16 -> 512*16
            out_list_dict[16] = torch.stack([getattr(self, f'condition_scale{16}')(tri_plane), getattr(self, f'condition_shift{16}')(tri_plane)])

            tri_plane = self.up2(tri_plane, c2)  # 512*16 -> 128*32 + 256*32 -> 384*32
            out_list_dict[32] = torch.stack([getattr(self, f'condition_scale{32}')(tri_plane), getattr(self, f'condition_shift{32}')(tri_plane)])

            tri_plane = self.up3(tri_plane, c1)  # 384*32 -> 96*64 + 128*64 -> 256*64
            out_list_dict[64] = torch.stack([getattr(self, f'condition_scale{64}')(tri_plane), getattr(self, f'condition_shift{64}')(tri_plane)])

            tri_plane = self.up4(tri_plane, c0)  # 256*64 -> 64*128 + 64*128 -> 96*128
            out_list_dict[128] = torch.stack([getattr(self, f'condition_scale{128}')(tri_plane), getattr(self, f'condition_shift{128}')(tri_plane)])

            tri_plane = self.final_head(self.head(tri_plane))
            out_list_dict[256] = torch.stack([getattr(self, f'condition_scale{256}')(tri_plane), getattr(self, f'condition_shift{256}')(tri_plane)])

            return  out_list_dict


class TriPlanefeat_SegformerDecoder(nn.Module):
    def __init__(self, inp_ch, sft_half=True, res=None, use_gru=False):
        super(TriPlanefeat_SegformerDecoder, self).__init__()
        self.sft_half = sft_half
        blocks = get_blocks(num_layers=50)
        self.res = res
        self.use_gru = use_gru
        self.face_pool = None if res is None else torch.nn.AdaptiveAvgPool2d((res, res))
        self.input_layer = nn.Sequential(nn.Conv2d(inp_ch, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck_IR_SE(bottleneck.in_channel,
                                                bottleneck.depth,
                                                bottleneck.stride))
        self.body = nn.Sequential(*modules)

        self.up1 = (UpLayer(1024, 512, upscale_factor=1, use_gru=use_gru, num_vit=4))
        self.up2 = (UpLayer(384, 384, use_gru=use_gru, num_vit=4))
        self.up3 = (UpLayer(224, 256, use_gru=use_gru, num_vit=3))
        self.up4 = (UpLayer(128, 96, use_gru=use_gru, num_vit=3))

        self.outconv0 = nn.Conv2d(384, 32, kernel_size=1, padding=0)  # 16
        self.outconv1 = nn.Conv2d(384, 512, kernel_size=1, padding=0)  # 32
        self.outconv2 = nn.Conv2d(256, 512, kernel_size=1, padding=0)  # 64
        self.outconv3 = nn.Conv2d(96, 256, kernel_size=1, padding=0)  # 128

    def forward(self, x, r_list=None, return_list=True):
        if x.dim() == 5:
            B, T = x.shape[:2]
            x = x.flatten(0, 1)
        else:
            T = 1
        if self.face_pool is not None and x.shape[-1] != self.res:
            x = self.face_pool(x)
        x = self.input_layer(x)
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                c0 = x
            if i == 6:
                c1 = x
            if i == 20:
                c2 = x
            elif i == 21:
                c3 = x
        out_list = []
        if self.use_gru:
            if r_list is None: r_list = [None for _ in range(4)]
            tri_plane, r_list[0] = self.up1(x, c3, T, r_list[0])             # 512*16 + 512*16 -> 512*16
            # out_list.append(self.outconv0(tri_plane))

            tri_plane = tri_plane.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)    # [B, C, H, W] -> [B*T, C, H, W]
            tri_plane, r_list[1] = self.up2(tri_plane, c2, T, r_list[1])     # 512*16 -> 128*32 + 256*32 -> 384*32
            out_list.append(self.outconv0(tri_plane))
            out_list.append(self.outconv1(tri_plane))

            tri_plane = tri_plane.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
            tri_plane, r_list[2] = self.up3(tri_plane, c1, T, r_list[2])     # 384*32 -> 96*64 + 128*64 -> 256*64
            out_list.append(self.outconv2(tri_plane))

            tri_plane = tri_plane.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
            tri_plane, r_list[3] = self.up4(tri_plane, c0, T, r_list[3])     # 256*64 -> 64*128 + 64*128 -> 96*128
            out_list.append(self.outconv3(tri_plane))

            return out_list, r_list
        else:
            tri_plane = self.up1(x, c3)  # 512*16 + 512*16 -> 512*16

            tri_plane = self.up2(tri_plane, c2)  # 512*16 -> 128*32 + 256*32 -> 384*32
            out_list.append(self.outconv0(tri_plane))
            out_list.append(self.outconv1(tri_plane))

            tri_plane = self.up3(tri_plane, c1)  # 384*32 -> 96*64 + 128*64 -> 256*64
            out_list.append(self.outconv2(tri_plane))

            tri_plane = self.up4(tri_plane, c0)  # 256*64 -> 64*128 + 64*128 -> 96*128
            out_list.append(self.outconv3(tri_plane))

            return out_list


class TriPlaneSFTfeat_SegformerDecoder(nn.Module):
    def __init__(self, inp_ch, sft_half=True, res=None, use_gru=False):
        super(TriPlaneSFTfeat_SegformerDecoder, self).__init__()
        self.sft_half = sft_half
        blocks = get_blocks(num_layers=50)
        self.res = res
        self.use_gru = use_gru
        self.face_pool = None if res is None else torch.nn.AdaptiveAvgPool2d((res, res))
        self.input_layer = nn.Sequential(nn.Conv2d(inp_ch, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck_IR_SE(bottleneck.in_channel,
                                                bottleneck.depth,
                                                bottleneck.stride))
        self.body = nn.Sequential(*modules)

        self.up1 = (UpLayer(1024, 512, upscale_factor=1, use_gru=use_gru, num_vit=4))
        self.up2 = (UpLayer(384, 384, use_gru=use_gru, num_vit=4))
        self.up3 = (UpLayer(224, 256, use_gru=use_gru, num_vit=3))
        self.up4 = (UpLayer(128, 96, use_gru=use_gru, num_vit=2))

        self.head = nn.PixelShuffle(upscale_factor=2)
        self.final_head = nn.Sequential(
            nn.Conv2d(24, 96, kernel_size=3, padding=1),
            nn.PReLU(96),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.PReLU(96)
        )

        self.block_resolutions = [2 ** i for i in range(int(np.log2(16)), int(np.log2(256)) + 1)]
        channels_dict = {res: min(32768 // res, 512) for res in self.block_resolutions}
        body_outchannels_dict = {16: 512, 32: 384, 64: 256, 128: 96, 256: 96}

        for res in self.block_resolutions:
            out_channels = body_outchannels_dict[res]
            sft_out_channels = channels_dict[res] // 2 if self.sft_half else channels_dict[res]
            condition_scale = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1))
            setattr(self, f'condition_scale{res}', condition_scale)
            condition_shift = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1))
            setattr(self, f'condition_shift{res}', condition_shift)

    def forward(self, x, r_list=None):
        if x.dim() == 5:
            B, T = x.shape[:2]
            x = x.flatten(0, 1)
        else:
            T = 1
        if self.face_pool is not None and x.shape[-1] != self.res:
            x = self.face_pool(x)
        x = self.input_layer(x)
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                c0 = x
            if i == 6:
                c1 = x
            if i == 20:
                c2 = x
            elif i == 21:
                c3 = x
        out_list_dict = {}
        if self.use_gru:
            if r_list is None: r_list = [None for _ in range(4)]
            tri_plane, r_list[0] = self.up1(x, c3, T, r_list[0])             # 512*16 + 512*16 -> 512*16
            out_list_dict[16] = torch.stack([getattr(self, f'condition_scale{16}')(tri_plane), getattr(self, f'condition_shift{16}')(tri_plane)])

            tri_plane = tri_plane.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)    # [B, C, H, W] -> [B*T, C, H, W]
            tri_plane, r_list[1] = self.up2(tri_plane, c2, T, r_list[1])     # 512*16 -> 128*32 + 256*32 -> 384*32
            out_list_dict[32] = torch.stack([getattr(self, f'condition_scale{32}')(tri_plane), getattr(self, f'condition_shift{32}')(tri_plane)])

            tri_plane = tri_plane.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
            tri_plane, r_list[2] = self.up3(tri_plane, c1, T, r_list[2])     # 384*32 -> 96*64 + 128*64 -> 256*64
            out_list_dict[64] = torch.stack([getattr(self, f'condition_scale{64}')(tri_plane), getattr(self, f'condition_shift{64}')(tri_plane)])

            tri_plane = tri_plane.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
            tri_plane, r_list[3] = self.up4(tri_plane, c0, T, r_list[3])     # 256*64 -> 64*128 + 64*128 -> 96*128
            out_list_dict[128] = torch.stack([getattr(self, f'condition_scale{128}')(tri_plane), getattr(self, f'condition_shift{128}')(tri_plane)])

            tri_plane = self.final_head(self.head(tri_plane))
            out_list_dict[256] = torch.stack([getattr(self, f'condition_scale{256}')(tri_plane), getattr(self, f'condition_shift{256}')(tri_plane)])

            return out_list_dict, r_list
        else:
            tri_plane = self.up1(x, c3)  # 512*16 + 512*16 -> 512*16
            out_list_dict[16] = torch.stack([getattr(self, f'condition_scale{16}')(tri_plane), getattr(self, f'condition_shift{16}')(tri_plane)])

            tri_plane = self.up2(tri_plane, c2)  # 512*16 -> 128*32 + 256*32 -> 384*32
            out_list_dict[32] = torch.stack([getattr(self, f'condition_scale{32}')(tri_plane), getattr(self, f'condition_shift{32}')(tri_plane)])

            tri_plane = self.up3(tri_plane, c1)  # 384*32 -> 96*64 + 128*64 -> 256*64
            out_list_dict[64] = torch.stack([getattr(self, f'condition_scale{64}')(tri_plane), getattr(self, f'condition_shift{64}')(tri_plane)])

            tri_plane = self.up4(tri_plane, c0)  # 256*64 -> 64*128 + 64*128 -> 96*128
            out_list_dict[128] = torch.stack([getattr(self, f'condition_scale{128}')(tri_plane), getattr(self, f'condition_shift{128}')(tri_plane)])

            tri_plane = self.final_head(self.head(tri_plane))
            out_list_dict[256] = torch.stack([getattr(self, f'condition_scale{256}')(tri_plane), getattr(self, f'condition_shift{256}')(tri_plane)])

            return out_list_dict

class TriPlane_SegformerDecoder(nn.Module):

    def __init__(self, inp_ch, res=None, use_gru=False):
        super(TriPlane_SegformerDecoder, self).__init__()
        blocks = get_blocks(num_layers=50)
        self.res = res
        self.use_gru = use_gru
        self.face_pool = None if res is None else torch.nn.AdaptiveAvgPool2d((res, res))
        self.input_layer = nn.Sequential(nn.Conv2d(inp_ch, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck_IR_SE(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules)

        self.up1 = (UpLayer(1024, 512, upscale_factor=1, use_gru=use_gru, num_vit=4))
        self.up2 = (UpLayer(384, 384, use_gru=use_gru, num_vit=4))
        self.up3 = (UpLayer(224, 256, use_gru=use_gru, num_vit=4))
        self.up4 = (UpLayer(128, 96, use_gru=use_gru, num_vit=4))
        self.head = nn.PixelShuffle(upscale_factor=2)
        self.final_head = nn.Sequential(
            nn.Conv2d(24, 96, kernel_size=3, padding=1),
            nn.PReLU(96),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.PReLU(96),
            nn.Conv2d(96, 96, kernel_size=1)
        )

    def forward(self, x, r_list=None):
        if x.dim() == 5:
            B, T = x.shape[:2]
            x = x.flatten(0, 1)
        else:
            T = 1
        if self.face_pool is not None and x.shape[-1] != self.res:
            x = self.face_pool(x)
        x = self.input_layer(x)
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                c0 = x
            if i == 6:
                c1 = x
            if i == 20:
                c2 = x
            elif i == 21:
                c3 = x

        if self.use_gru:
            if r_list is None: r_list = [None for _ in range(4)]
            tri_plane, r_list[0] = self.up1(x, c3, T, r_list[0], seq2seq=self.seq2seq)  # 512*16 + 512*16 -> 512*16

            if not self.seq2seq: tri_plane = tri_plane.unsqueeze(1).expand(-1, T, -1, -1, -1)  # [B, C, H, W] -> [B*T, C, H, W]
            tri_plane, r_list[1] = self.up2(tri_plane.flatten(0, 1), c2, T, r_list[1], seq2seq=self.seq2seq)  # 512*16 -> 128*32 + 256*32 -> 384*32

            if not self.seq2seq: tri_plane = tri_plane.unsqueeze(1).expand(-1, T, -1, -1, -1)
            tri_plane, r_list[2] = self.up3(tri_plane.flatten(0, 1), c1, T, r_list[2], seq2seq=self.seq2seq)  # 384*32 -> 96*64 + 128*64 -> 256*64

            if not self.seq2seq: tri_plane = tri_plane.unsqueeze(1).expand(-1, T, -1, -1, -1)
            tri_plane, r_list[3] = self.up4(tri_plane.flatten(0, 1), c0, T, r_list[3], seq2seq=self.seq2seq)  # 256*64 -> 64*128 + 64*128 -> 96*128
            if self.seq2seq: tri_plane = tri_plane.flatten(0, 1)

            tri_plane = self.head(tri_plane)  # 256
            tri_plane = self.final_head(tri_plane)
            return tri_plane, r_list
        else:
            tri_plane = self.up1(x, c3)             # 1024 * 16 -> 512 * 16
            tri_plane = self.up2(tri_plane, c2)     # -> 384 * 32
            tri_plane = self.up3(tri_plane, c1)     # -> 384 * 64
            tri_plane = self.up4(tri_plane, c0)     # -> 384 * 128
            tri_plane = self.head(tri_plane)        # -> 384 * 256
            tri_plane = self.final_head(tri_plane)
            return tri_plane

class UpLayer(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=2, use_gru=False, num_vit=0):
        super().__init__()

        self.up = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.conv = DoubleConv(in_channels, out_channels)
        self.conv_gru = ConvGRU(out_channels, out_act_prelu=False) if use_gru else None
        self.use_vit = num_vit > 0
        self.transformer = transformer_block(in_chans=in_channels, num_vit=num_vit) if self.use_vit else None

    def forward(self, x1, x2=None, T=0, r=None):
        x1 = self.up(x1)
        x = x1 if x2 is None else torch.cat([x2, x1], dim=1)
        if self.use_vit: x = self.transformer(x)
        x = self.conv(x)
        if self.conv_gru is None:
            return x
        else:
            x, r = self.conv_gru(x.unflatten(0, (-1, T)), r, seq2seq=False)  # [B*T, C, H, W] -> [B, C, H, W]
            return x, r
