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
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True)) # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class EncoderResBlock(torch.nn.Module):
    def __init__(self, img_channel, in_channel, out_channel, resample_filter=[1, 3, 3, 1], downsample=True):
        super().__init__()
        self.fromrgb = Conv2dLayer(img_channel, in_channel, 1, activation='lrelu')
        self.conv1 = Conv2dLayer(in_channel, in_channel, 3, activation='lrelu')
        self.conv2 = Conv2dLayer(in_channel, out_channel, 3, down=2, activation='lrelu')
        # self.skip  = Conv2dLayer(in_channel, out_channel, 1, down=2, activation='linear', bias=False)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.downsample = downsample

    def forward(self, input, skip=None):
        if self.downsample:
            input = upfirdn2d.downsample2d(input, self.resample_filter)
        out = self.fromrgb(input)
        if skip is not None:
            out = out + skip
        out  = self.conv1(out)
        out  = self.conv2(out)
        return input, out

#----------------------------------------------------------------------------

@persistence.persistent_class
class DecoderBlock(torch.nn.Module):
    def __init__(self, img_channel, in_channel, out_channel, architecture='skip', resample_filter=[1, 3, 3, 1]):
        super().__init__()
        assert architecture in ['orig', 'skip', 'resnet']
        self.conv0 = Conv2dLayer(in_channel, out_channel, 3, up=2, activation='lrelu')
        self.conv1 = Conv2dLayer(out_channel, out_channel, 3, activation='lrelu')
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.architecture = architecture
        if self.architecture == 'skip':
            self.torgb = Conv2dLayer(out_channel, img_channel, 1, activation='lrelu')

    def forward(self, x, img):
        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        # ToRGB.
        if img is not None:
            img = upfirdn2d.upsample2d(img, self.resample_filter)

        if self.architecture == 'skip':
            y = self.torgb(x)
            img = img.add_(y) if img is not None else y

        return x, img


class DecoderBlock_new(torch.nn.Module):    # 增加了gru，vit的选项
    def __init__(self, img_channel, in_channel, out_channel, cond_channel=0, architecture='skip', resample_filter=[1, 3, 3, 1], use_gru=False):
        super().__init__()
        assert architecture in ['orig', 'skip']
        self.cond_channel = cond_channel
        self.conv0 = Conv2dLayer(in_channel, out_channel, 3, up=2, activation='lrelu')
        self.conv1 = Conv2dLayer(out_channel + cond_channel, out_channel, 3, activation='lrelu')
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.architecture = architecture
        if self.architecture == 'skip':
            self.torgb = Conv2dLayer(out_channel, img_channel, 1, activation='lrelu')
        self.gru = ConvGRU(out_channel) if use_gru else None
        self.use_gru = use_gru
        # self.use_vit = num_vit > 0
        # self.transformer = transformer_block(in_chans=out_channel, num_vit=num_vit) if self.use_vit else None
        # self.fusion = Conv2dLayer(out_channel + cond_channel, out_channel, kernel_size=3, activation='linear', bias=True) \
        #     if cond_channel > 0 else None

    def forward(self, x, img, cond=None, T=0, r=None):
        # Main layers.
        x = self.conv0(x)
        # if self.use_vit: x = self.transformer(x)
        if self.cond_channel > 0: x = torch.cat([x, cond], dim=1)#self.fusion(torch.cat([x, cond], dim=1))
        x = self.conv1(x)

        if self.use_gru:
            x, r = self.gru(x.unflatten(0, (-1, T)), r)    # [B, C, H, W]

        # ToRGB.
        if img is not None:
            img = upfirdn2d.upsample2d(img, self.resample_filter)

        if self.architecture == 'skip':
            y = self.torgb(x)
            img = img.add_(y) if img is not None else y

        if self.use_gru:
            return x, img, r
        else:
            return x, img

@persistence.persistent_class
class ConvFusionDecoderBlock(torch.nn.Module):
    def __init__(self, img_channel, in_channel, out_channel, architecture='skip', resample_filter=[1, 3, 3, 1], T=4):
        super().__init__()
        assert architecture in ['orig', 'skip', 'resnet']
        self.conv0 = Conv2dLayer(in_channel, out_channel, 3, up=2, activation='lrelu')
        self.conv1 = Conv2dLayer(out_channel, out_channel, 3, activation='lrelu')
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.architecture = architecture
        if self.architecture == 'skip':
            self.torgb = Conv2dLayer(out_channel, img_channel, 1, activation='lrelu')
        self.conv_fusion = Conv2dLayer(out_channel * T, out_channel, 3, activation='lrelu')

    def forward(self, x, img, T, r=None):
        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)
        x_time = x.unflatten(0, (-1, T))
        x_time = self.conv_fusion(x_time.flatten(1, 2))    # [B, C, H, W]

        # ToRGB.
        if img is not None:
            img = upfirdn2d.upsample2d(img, self.resample_filter)

        if self.architecture == 'skip':
            y = self.torgb(x_time)
            img = img.add_(y) if img is not None else y

        return x_time, img, r


class RecurrentDecoderBlock(torch.nn.Module):
    def __init__(self, img_channel, in_channel, out_channel, architecture='skip', resample_filter=[1, 3, 3, 1]):
        super().__init__()
        assert architecture in ['orig', 'skip', 'resnet']
        self.conv0 = Conv2dLayer(in_channel, out_channel, 3, up=2, activation='lrelu')
        self.conv1 = Conv2dLayer(out_channel, out_channel, 3, activation='lrelu')
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.architecture = architecture
        if self.architecture == 'skip':
            self.torgb = Conv2dLayer(out_channel, img_channel, 1, activation='lrelu')
        self.gru = ConvGRU(out_channel)

    def forward(self, x, img, T, r=None):
        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)
        x_time = x.unflatten(0, (-1, T))
        x_time, r = self.gru(x_time, r)    # [B, C, H, W]

        # ToRGB.
        if img is not None:
            img = upfirdn2d.upsample2d(img, self.resample_filter)

        if self.architecture == 'skip':
            y = self.torgb(x_time)
            img = img.add_(y) if img is not None else y

        return x_time, img, r


class DecoderBlock_SFT(torch.nn.Module):
    def __init__(self, img_channel, in_channel, out_channel, architecture='skip', resample_filter=[1, 3, 3, 1], use_gru=False, out_sft=False):
        super().__init__()
        assert architecture in ['orig', 'skip', 'resnet']
        self.conv0 = Conv2dLayer(in_channel, out_channel, 3, up=2, activation='lrelu')
        self.conv1 = Conv2dLayer(out_channel, out_channel, 3, activation='lrelu')
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.architecture = architecture
        if self.architecture == 'skip':
            self.torgb = Conv2dLayer(out_channel, img_channel, 1, activation='lrelu')
        self.gru = ConvGRU(out_channel) if use_gru else None
        self.out_sft = out_sft
        if out_sft:
            sft_out_channels = out_channel // 2 # sft_half
            self.condition_scale = Conv2dLayer(out_channel, sft_out_channels, kernel_size=3, activation='linear', bias=True)
            self.condition_shift = Conv2dLayer(out_channel, sft_out_channels, kernel_size=3, activation='linear', bias=True)

    def forward(self, x, img, T=None, r=None):
        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            if self.gru is not None:
                x_time = x.unflatten(0, (-1, T))
                x, r = self.gru(x_time, r)  # [B, C, H, W]
            x = self.conv1(x)

        # ToRGB.
        if img is not None:
            img = upfirdn2d.upsample2d(img, self.resample_filter)

        if self.architecture == 'skip':
            y = self.torgb(x)
            img = img.add_(y) if img is not None else y

        sft = torch.stack([self.condition_scale(x), self.condition_shift(x)]) if self.out_sft else None
        if self.gru:
            return x, sft, img, r
        else:
            return x, sft, img
#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},',
            f'up={self.up}, down={self.down}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if update_emas and self.w_avg_beta is not None:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, self.in_channels, in_resolution, in_resolution])
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},',
            f'resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        return x

    def extra_repr(self):
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        is_last,                                # Is this the last block?
        architecture            = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = True,         # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(x, [None, self.in_channels, self.resolution // 2, self.resolution // 2])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

class CondSynthesisNetwork(torch.nn.Module):
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
                 residual_arch=False,
                 **block_kwargs,  # Arguments for SynthesisBlock.
                 ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.residual_arch = residual_arch
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
        channels_dict = {res: min(channel_base // res, channel_max) for res in ([final_size] + self.block_resolutions)}

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            block = DecoderBlock(img_channels, in_channels, out_channels)
            setattr(self, f'b{res}', block)

        # encoder
        self.encoder_res = [2 ** i for i in range(int(np.log2(self.in_size)), int(np.log2(self.final_size)) - 1, -1)]
        self.encoder = torch.nn.ModuleList()
        for res in self.encoder_res[:-1]:
            in_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            block = EncoderResBlock(self.cond_channels, in_channels, out_channels, downsample=(res < in_size))
            self.encoder.append(block)
        # trainable fusion module
        self.fusion = torch.nn.ModuleList()
        for res in self.encoder_res[::-1]:
            num_channels = channels_dict[res]
            if res > self.final_size:
                self.fusion.append(
                    Conv2dLayer(num_channels * 2, num_channels, kernel_size=3, activation='linear', bias=True))
            else:
                self.fusion.append(
                    Conv2dLayer(num_channels, num_channels, kernel_size=3, activation='linear', bias=True))

    def forward(self, x_cond, return_list=False):
        # obtain multi-scale content features
        x_in = x_cond
        cond_list = []
        cond_out = None
        for i, res in enumerate(self.encoder_res[:-1][::-1]):
            x_in, cond_out = self.encoder[i](x_in, cond_out)
            cond_list.append(cond_out)

        cond_list = cond_list[::-1]

        x = img = None
        x_list, _index = [], 0
        start_layer = int(np.log2(32)) - self.final_size_log2 - 1  # 从32分辨率开始
        for res in self.block_resolutions:
            # pass the mid-layer features of E to the corresponding resolution layers of G
            if 2 ** (_index + int(np.log2(self.final_size))) < self.num_cond_res:
                if res == self.block_resolutions[0]:
                    x = self.fusion[_index](cond_list[_index])
                else:
                    x = torch.cat([x, cond_list[_index]], dim=1)
                    x = self.fusion[_index](x)
            block = getattr(self, f'b{res}')
            x, img = block(x, img)
            if _index >= start_layer:
                if return_list:
                    if _index == start_layer: x_list.append(img.clone())
                    x_list.append(x.clone())
            _index += 1

        if self.residual_arch:
            assert img.shape[1] == x_cond.shape[1]
            img = img + (x_cond if x_cond.shape[-2:] == img.shape[-2:] else
                         torch.nn.functional.interpolate(x_cond, size=img.shape[-2:], mode='bilinear', antialias=True))
        if return_list:
            x_list.append(img)
            return x_list
        else:
            return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])


class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
                 img_resolution,  # Output image resolution.
                 img_channels,  # Number of color channels.
                 final_size=4,  # Lowest resolution of encoding.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=4,  # Use FP16 for the N highest resolutions.
                 **block_kwargs,  # Arguments for SynthesisBlock.
                 ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.final_size = final_size
        self.final_size_log2 = int(np.log2(final_size))
        self.block_resolutions = [2 ** i for i in range(self.final_size_log2 + 1, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in ([final_size] + self.block_resolutions)}

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            block = DecoderBlock(img_channels, in_channels, out_channels)
            setattr(self, f'b{res}', block)

    def forward(self, x, img, return_list=False):
        assert x.shape[-1] == img.shape[-1] == self.final_size, (x.shape, img.shape, self.final_size)
        x_list, _index = [], 0
        start_out_res = 32
        assert start_out_res >= self.final_size
        if start_out_res == self.final_size:
            x_list.append(img.clone())
            x_list.append(x.clone())
        start_layer = int(np.log2(start_out_res)) - self.final_size_log2 - 1  # 从32分辨率开始输出
        for res in self.block_resolutions:
            # pass the mid-layer features of E to the corresponding resolution layers of G
            block = getattr(self, f'b{res}')
            x, img = block(x, img)
            if _index >= start_layer:
                if return_list:
                    if _index == start_layer: x_list.append(img.clone())
                    x_list.append(x.clone())
            _index += 1

        if return_list:
            x_list.append(img)
            return x_list
        else:
            return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])

# ----------------------------------------------------------------------------
class ConvGRU(torch.nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = torch.nn.Sequential(
            # nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            Conv2dLayer(channels * 2, channels * 2, kernel_size, activation='linear'),
            torch.nn.Sigmoid()
        )
        self.hh = torch.nn.Sequential(
            # nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            Conv2dLayer(channels * 2, channels, kernel_size, activation='linear'),
            torch.nn.Tanh()
        )

    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h

    def forward_time_series(self, x, h, seq2seq=False):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            if seq2seq: o.append(ot)
        o = torch.stack(o, dim=1) if seq2seq else ot
        return o, h

    def forward(self, x, h):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)

class CondSynthesisNetwork_new(torch.nn.Module):    # 和CondSynthesisNetwork的区别在于：去掉了residual_arch的选项和fusion网络，将condition放在了decoder的两个卷积层中间，此外支持了其它接口
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
        self.use_gru = use_gru
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
        channels_dict = {res: min(channel_base // res, channel_max) for res in ([final_size] + self.block_resolutions)}

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            block = DecoderBlock_new(img_channels, in_channels, out_channels, cond_channel=out_channels if res < self.img_resolution else 0,
                                     architecture='skip' if res==32 else 'orig')
            setattr(self, f'b{res}', block)

        # encoder
        self.encoder_res = [2 ** i for i in range(int(np.log2(self.in_size)), int(np.log2(self.final_size)) - 1, -1)]
        self.encoder = torch.nn.ModuleList()
        for res in self.encoder_res[:-1]:
            in_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            block = EncoderResBlock(self.cond_channels, in_channels, out_channels, downsample=(res < in_size))
            self.encoder.append(block)
        # # trainable fusion module
        # self.fusion = torch.nn.ModuleList()
        # self.fusion.append(
        #     Conv2dLayer(channels_dict[final_size], channels_dict[final_size], kernel_size=3, activation='linear', bias=True))

    def forward(self, x_cond, return_list=False):
        # obtain multi-scale content features
        x_in = x_cond
        cond_list = [None]
        cond_out = None
        for i, res in enumerate(self.encoder_res[:-1][::-1]):
            x_in, cond_out = self.encoder[i](x_in, cond_out)
            cond_list.append(cond_out)

        cond_list = cond_list[::-1]
        x = cond_list[0] #self.fusion[0](cond_list[0])
        img = None
        x_list, _index = [], 0
        start_layer = int(np.log2(32)) - self.final_size_log2 - 1  # 从32分辨率开始
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cond_list[_index+1])
            if _index >= start_layer:
                if return_list:
                    if _index == start_layer: x_list.append(img.clone())
                    x_list.append(x.clone())
            _index += 1

        if return_list:
            x_list.append(img)
            return x_list
        else:
            return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])

class CondSynthesisNetwork_SFT(torch.nn.Module):
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
                 sft_half=True,
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
        channels_dict = {res: min(channel_base // res, channel_max) for res in ([final_size] + self.block_resolutions)}
        self.use_gru = use_gru
        self.sft_half = sft_half
        self.num_ws = 0
        self.out_start_res = 16
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            block = DecoderBlock_SFT(img_channels, in_channels, out_channels, architecture='orig', use_gru=use_gru, out_sft=res>=self.out_start_res)
            setattr(self, f'b{res}', block)

        # encoder
        self.encoder_res = [2 ** i for i in range(int(np.log2(self.in_size)), int(np.log2(self.final_size)) - 1, -1)]
        self.encoder = torch.nn.ModuleList()
        for res in self.encoder_res[:-1]:
            in_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            block = EncoderResBlock(self.cond_channels, in_channels, out_channels, downsample=(res < in_size))
            self.encoder.append(block)
        # trainable fusion module
        self.fusion = torch.nn.ModuleList()
        for res in self.encoder_res[::-1]:
            num_channels = channels_dict[res]
            if res > self.final_size:
                self.fusion.append(
                    Conv2dLayer(num_channels * 2, num_channels, kernel_size=3, activation='linear', bias=True))
            else:
                self.fusion.append(
                    Conv2dLayer(num_channels, num_channels, kernel_size=3, activation='linear', bias=True))

    def forward_onlyEncoder(self, x_cond):
        assert x_cond.dim() == 5
        x_in = x_cond.flatten(0, 1)
        cond_list = []
        cond_out = None
        for i, res in enumerate(self.encoder_res[:-1][::-1]):
            x_in, cond_out = self.encoder[i](x_in, cond_out)
            cond_list.append(cond_out)
        cond_list = cond_list[::-1]
        return cond_list

    def forward_onlyDecoder(self, T, cond_list, r_list=None):
        x = img = None
        out_list_dict, _index = {}, 0
        start_layer = int(np.log2(self.out_start_res)) - self.final_size_log2 - 1  # 从16分辨率开始
        if r_list is None: r_list = [None for _ in range(len(self.block_resolutions))]
        for res in self.block_resolutions:
            # pass the mid-layer features of E to the corresponding resolution layers of G
            if 2 ** (_index + int(np.log2(self.final_size))) < self.num_cond_res:
                if res == self.block_resolutions[0]:
                    x = self.fusion[_index](cond_list[_index])
                else:
                    x = torch.cat([x, cond_list[_index]], dim=1)
                    x = self.fusion[_index](x)
            block = getattr(self, f'b{res}')
            if self.use_gru:
                x, sft, img, r_list[_index] = block(x, img, T, r_list[_index])
            else:
                x, sft, img, _ = block(x, img)
            if _index >= start_layer:
                out_list_dict[res] = sft
            x = x.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
            _index += 1

        if self.use_gru:
            return out_list_dict, r_list
        else:
            return out_list_dict

    def forward(self, x_cond, r_list=None):
        if x_cond.dim() == 5:
            B, T = x_cond.shape[:2]
            x_in = x_cond.flatten(0, 1)
        else:
            T = 1
            x_in = x_cond

        cond_list = []
        cond_out = None
        for i, res in enumerate(self.encoder_res[:-1][::-1]):
            x_in, cond_out = self.encoder[i](x_in, cond_out)
            cond_list.append(cond_out)

        cond_list = cond_list[::-1]

        x = img = None
        out_list_dict, _index = {}, 0
        start_layer = int(np.log2(self.out_start_res)) - self.final_size_log2 - 1  # 从16分辨率开始
        if r_list is None: r_list = [None for _ in range(len(self.block_resolutions))]
        for res in self.block_resolutions:
            # pass the mid-layer features of E to the corresponding resolution layers of G
            if 2 ** (_index + int(np.log2(self.final_size))) < self.num_cond_res:
                if res == self.block_resolutions[0]:
                    x = self.fusion[_index](cond_list[_index])
                else:
                    x = torch.cat([x, cond_list[_index]], dim=1)
                    x = self.fusion[_index](x)
            block = getattr(self, f'b{res}')
            if self.use_gru:
                x, sft, img, r_list[_index] = block(x, img, T, r_list[_index])
            else:
                x, sft, img = block(x, img)
            if _index >= start_layer:
                out_list_dict[res] = sft
            x = x.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
            _index += 1

        if self.use_gru:
            return out_list_dict, r_list
        else:
            return out_list_dict

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])

class CondSynthesisNetwork_withGRU(torch.nn.Module):
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
        channels_dict = {res: min(channel_base // res, channel_max) for res in ([final_size] + self.block_resolutions)}

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            block = RecurrentDecoderBlock(img_channels, in_channels, out_channels)
            # block = ConvFusionDecoderBlock(img_channels, in_channels, out_channels)
            setattr(self, f'b{res}', block)

        # encoder
        self.encoder_res = [2 ** i for i in range(int(np.log2(self.in_size)), int(np.log2(self.final_size)) - 1, -1)]
        self.encoder = torch.nn.ModuleList()
        for res in self.encoder_res[:-1]:
            in_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            block = EncoderResBlock(self.cond_channels, in_channels, out_channels, downsample=(res < in_size))
            self.encoder.append(block)
        # trainable fusion module
        self.fusion = torch.nn.ModuleList()
        for res in self.encoder_res[::-1]:
            num_channels = channels_dict[res]
            if res > self.final_size:
                self.fusion.append(
                    Conv2dLayer(num_channels * 2, num_channels, kernel_size=3, activation='linear', bias=True))
            else:
                self.fusion.append(
                    Conv2dLayer(num_channels, num_channels, kernel_size=3, activation='linear', bias=True))

    def forward_onlyEncoder(self, x_cond):
        assert x_cond.dim() == 5
        x_in = x_cond.flatten(0, 1)
        cond_list = []
        cond_out = None
        for i, res in enumerate(self.encoder_res[:-1][::-1]):
            x_in, cond_out = self.encoder[i](x_in, cond_out)
            cond_list.append(cond_out)
        cond_list = cond_list[::-1]
        return cond_list

    def forward_onlyDecoder(self, T, cond_list, r_list=None):

        x = img = None
        x_list, _index = [], 0
        start_layer = int(np.log2(32)) - self.final_size_log2 - 1  # 从32分辨率开始
        if r_list is None: r_list = [None for _ in range(len(self.block_resolutions))]
        for res in self.block_resolutions:
            # pass the mid-layer features of E to the corresponding resolution layers of G
            if 2 ** (_index + int(np.log2(self.final_size))) < self.num_cond_res:
                if res == self.block_resolutions[0]:
                    x = self.fusion[_index](cond_list[_index])
                else:
                    x = torch.cat([x, cond_list[_index]], dim=1)
                    x = self.fusion[_index](x)
            block = getattr(self, f'b{res}')
            x, img, r_list[_index] = block(x, img, T, r_list[_index])
            if _index >= start_layer:
                if _index == start_layer: x_list.append(img.clone())
                x_list.append(x.clone())
            x = x.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
            _index += 1

        x_list.append(img)
        return x_list, r_list


    def forward(self, x_cond, r_list=None, fix_encoder=False, return_list=False):
        assert x_cond.dim() == 5
        B, T = x_cond.shape[:2]
        # obtain multi-scale content features
        x_in = x_cond.flatten(0, 1)
        cond_list = []
        cond_out = None
        for i, res in enumerate(self.encoder_res[:-1][::-1]):
            x_in, cond_out = self.encoder[i](x_in, cond_out)
            cond_list.append(cond_out)
        if fix_encoder: cond_list = [cond_out.detach() for cond_out in cond_list]

        cond_list = cond_list[::-1]

        x = img = None
        x_list, _index = [], 0
        start_layer = int(np.log2(32)) - self.final_size_log2 - 1  # 从32分辨率开始
        if r_list is None: r_list = [None for _ in range(len(self.block_resolutions))]
        for res in self.block_resolutions:
            # pass the mid-layer features of E to the corresponding resolution layers of G
            if 2 ** (_index + int(np.log2(self.final_size))) < self.num_cond_res:
                if res == self.block_resolutions[0]:
                    x = self.fusion[_index](cond_list[_index])
                else:
                    x = torch.cat([x, cond_list[_index]], dim=1)
                    x = self.fusion[_index](x)
            block = getattr(self, f'b{res}')
            x, img, r_list[_index] = block(x, img, T, r_list[_index])
            if _index >= start_layer:
                if return_list:
                    if _index == start_layer: x_list.append(img.clone())
                    x_list.append(x.clone())
            x = x.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
            _index += 1

        if return_list:
            x_list.append(img)
            return x_list, r_list
        else:
            return img, r_list

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])

class CondSynthesisNetwork_withConvFusion(torch.nn.Module):
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
        channels_dict = {res: min(channel_base // res, channel_max) for res in ([final_size] + self.block_resolutions)}

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            # block = RecurrentDecoderBlock(img_channels, in_channels, out_channels)
            block = ConvFusionDecoderBlock(img_channels, in_channels, out_channels)
            setattr(self, f'b{res}', block)

        # encoder
        self.encoder_res = [2 ** i for i in range(int(np.log2(self.in_size)), int(np.log2(self.final_size)) - 1, -1)]
        self.encoder = torch.nn.ModuleList()
        for res in self.encoder_res[:-1]:
            in_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            block = EncoderResBlock(self.cond_channels, in_channels, out_channels, downsample=(res < in_size))
            self.encoder.append(block)
        # trainable fusion module
        self.fusion = torch.nn.ModuleList()
        for res in self.encoder_res[::-1]:
            num_channels = channels_dict[res]
            if res > self.final_size:
                self.fusion.append(
                    Conv2dLayer(num_channels * 2, num_channels, kernel_size=3, activation='linear', bias=True))
            else:
                self.fusion.append(
                    Conv2dLayer(num_channels, num_channels, kernel_size=3, activation='linear', bias=True))

    def forward_onlyEncoder(self, x_cond):
        assert x_cond.dim() == 5
        x_in = x_cond.flatten(0, 1)
        cond_list = []
        cond_out = None
        for i, res in enumerate(self.encoder_res[:-1][::-1]):
            x_in, cond_out = self.encoder[i](x_in, cond_out)
            cond_list.append(cond_out)
        cond_list = cond_list[::-1]
        return cond_list

    def forward_onlyDecoder(self, T, cond_list, r_list=None):

        x = img = None
        x_list, _index = [], 0
        start_layer = int(np.log2(32)) - self.final_size_log2 - 1  # 从32分辨率开始
        if r_list is None: r_list = [None for _ in range(len(self.block_resolutions))]
        for res in self.block_resolutions:
            # pass the mid-layer features of E to the corresponding resolution layers of G
            if 2 ** (_index + int(np.log2(self.final_size))) < self.num_cond_res:
                if res == self.block_resolutions[0]:
                    x = self.fusion[_index](cond_list[_index])
                else:
                    x = torch.cat([x, cond_list[_index]], dim=1)
                    x = self.fusion[_index](x)
            block = getattr(self, f'b{res}')
            x, img, r_list[_index] = block(x, img, T, r_list[_index])
            if _index >= start_layer:
                if _index == start_layer: x_list.append(img.clone())
                x_list.append(x.clone())
            x = x.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
            _index += 1

        x_list.append(img)
        return x_list, r_list


    def forward(self, x_cond, r_list=None, fix_encoder=False, return_list=False):
        assert x_cond.dim() == 5
        B, T = x_cond.shape[:2]
        # obtain multi-scale content features
        x_in = x_cond.flatten(0, 1)
        cond_list = []
        cond_out = None
        for i, res in enumerate(self.encoder_res[:-1][::-1]):
            x_in, cond_out = self.encoder[i](x_in, cond_out)
            cond_list.append(cond_out)
        if fix_encoder: cond_list = [cond_out.detach() for cond_out in cond_list]

        cond_list = cond_list[::-1]

        x = img = None
        x_list, _index = [], 0
        start_layer = int(np.log2(32)) - self.final_size_log2 - 1  # 从32分辨率开始
        if r_list is None: r_list = [None for _ in range(len(self.block_resolutions))]
        for res in self.block_resolutions:
            # pass the mid-layer features of E to the corresponding resolution layers of G
            if 2 ** (_index + int(np.log2(self.final_size))) < self.num_cond_res:
                if res == self.block_resolutions[0]:
                    x = self.fusion[_index](cond_list[_index])
                else:
                    x = torch.cat([x, cond_list[_index]], dim=1)
                    x = self.fusion[_index](x)
            block = getattr(self, f'b{res}')
            x, img, r_list[_index] = block(x, img, T, r_list[_index])
            if _index >= start_layer:
                if return_list:
                    if _index == start_layer: x_list.append(img.clone())
                    x_list.append(x.clone())
            x = x.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1)
            _index += 1

        if return_list:
            x_list.append(img)
            return x_list, r_list
        else:
            return img, r_list

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])