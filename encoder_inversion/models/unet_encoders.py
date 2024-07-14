import torch
import numpy as np
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Flatten, Sequential, Module
from .helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE


class ConvGRU(torch.nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 out_act_prelu=False):
        super().__init__()
        self.channels = channels
        self.ih = torch.nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            # Conv2dLayer(channels * 2, channels * 2, kernel_size, activation='linear'),
            torch.nn.Sigmoid()
        )
        self.hh = torch.nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            # Conv2dLayer(channels * 2, channels, kernel_size, activation='linear'),
            nn.PReLU(channels) if out_act_prelu else torch.nn.Tanh()
        )

    def forward_single_frame(self, x, h):
        # h = torch.zeros_like(x)
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h

    def forward_time_series(self, x, h, seq2seq):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            if seq2seq: o.append(ot)
        o = torch.stack(o, dim=1) if seq2seq else ot
        return o, h

    def forward(self, x, h, seq2seq=False):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        if x.ndim == 5:
            return self.forward_time_series(x, h, seq2seq)
        else:
            return self.forward_single_frame(x, h)


class ConvFusion(torch.nn.Module):
    def __init__(self, channels: int, T: int = 4):
        super().__init__()
        self.channels = channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(channels * T, channels, kernel_size=3, padding=1),
            nn.PReLU(channels)
        )
        self.T = T

    def forward(self, x):
        if x.ndim == 5:
            assert x.shape[1] == self.T
            return self.double_conv(x.flatten(1, 2))    # [B, T, C, H, W] -> [B, T*C, H, W] -> [B, C, H, W]
        else:
            return self.double_conv(x)


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, use_instnorm=False):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.InstanceNorm2d(in_channels) if use_instnorm else nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.PReLU(out_channels),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()

        self.up = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class recurrent_Up(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()

        self.up = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.conv = DoubleConv(in_channels, out_channels, use_instnorm=False)
        self.conv_gru = ConvGRU(out_channels, out_act_prelu=False)
        # self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0) # gru的输出经过tanh被限制在(-1,1)，需要一个映射   # 似乎不需要也行

    def forward(self, x1, x2, T, r=None, seq2seq=False):
        x1 = self.up(x1)
        x = self.conv(torch.cat([x2, x1], dim=1))
        x, r = self.conv_gru(x.unflatten(0, (-1, T)), r, seq2seq)    # [B*T, C, H, W] -> [B, C, H, W]
        return x, r


class TimeFusion_Up(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2, T=4):
        super().__init__()

        self.up = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.conv = DoubleConv(in_channels, out_channels, use_instnorm=False)
        self.conv_gru = ConvFusion(out_channels, T)
        # self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0) # gru的输出经过tanh被限制在(-1,1)，需要一个映射   # 似乎不需要也行

    def forward(self, x1, x2, T, r=None):
        x1 = self.up(x1)
        x = self.conv(torch.cat([x2, x1], dim=1))
        x = self.conv_gru(x.unflatten(0, (-1, T)))    # [B*T, C, H, W] -> [B, C, H, W]
        return x, r


class TriPlane_Encoder(Module):

    def __init__(self, inp_ch, res=None):
        super(TriPlane_Encoder, self).__init__()

        blocks = get_blocks(num_layers=50)
        self.res = res
        self.face_pool = None if res is None else torch.nn.AdaptiveAvgPool2d((res, res))
        self.input_layer = Sequential(Conv2d(inp_ch, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck_IR_SE(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.up1 = (Up(1024, 512, upscale_factor=1))
        self.up2 = (Up(384, 384))
        self.up3 = (Up(224, 256))
        self.up4 = (Up(128, 96))
        self.head = nn.PixelShuffle(upscale_factor=2)
        self.final_head = nn.Sequential(
            nn.Conv2d(24, 96, kernel_size=3, padding=1),
            nn.PReLU(96),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.PReLU(96),
            nn.Conv2d(96, 96, kernel_size=1)
        )

    def forward(self, x):
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

        tri_plane = self.up1(x, c3)             # 1024 * 16 -> 512 * 16
        tri_plane = self.up2(tri_plane, c2)     # -> 384 * 32
        tri_plane = self.up3(tri_plane, c1)     # -> 384 * 64
        tri_plane = self.up4(tri_plane, c0)     # -> 384 * 128
        tri_plane = self.head(tri_plane)        # -> 384 * 256
        tri_plane = self.final_head(tri_plane)
        return tri_plane

class TriPlane_Encoder_withGRU(Module):

    def __init__(self, inp_ch, res=None, seq2seq=False):
        super(TriPlane_Encoder_withGRU, self).__init__()

        blocks = get_blocks(num_layers=50)
        self.seq2seq = seq2seq
        self.res = res
        self.face_pool = None if res is None else torch.nn.AdaptiveAvgPool2d((res, res))
        self.input_layer = Sequential(Conv2d(inp_ch, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck_IR_SE(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.up1 = (recurrent_Up(1024, 512, upscale_factor=1))
        self.up2 = (recurrent_Up(384, 384))
        self.up3 = (recurrent_Up(224, 256))
        self.up4 = (recurrent_Up(128, 96))
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
        out_list = []
        if r_list is None: r_list = [None for _ in range(4)]
        tri_plane, r_list[0] = self.up1(x, c3, T, r_list[0], seq2seq=self.seq2seq)             # 512*16 + 512*16 -> 512*16
        if not self.seq2seq: tri_plane = tri_plane.unsqueeze(1).expand(-1, T, -1, -1, -1)    # [B, C, H, W] -> [B*T, C, H, W]
        tri_plane, r_list[1] = self.up2(tri_plane.flatten(0, 1), c2, T, r_list[1], seq2seq=self.seq2seq)     # 512*16 -> 128*32 + 256*32 -> 384*32
        if not self.seq2seq: tri_plane = tri_plane.unsqueeze(1).expand(-1, T, -1, -1, -1)
        tri_plane, r_list[2] = self.up3(tri_plane.flatten(0, 1), c1, T, r_list[2], seq2seq=self.seq2seq)     # 384*32 -> 96*64 + 128*64 -> 256*64
        if not self.seq2seq: tri_plane = tri_plane.unsqueeze(1).expand(-1, T, -1, -1, -1)
        tri_plane, r_list[3] = self.up4(tri_plane.flatten(0, 1), c0, T, r_list[3], seq2seq=self.seq2seq)     # 256*64 -> 64*128 + 64*128 -> 96*128
        if self.seq2seq: tri_plane = tri_plane.flatten(0, 1)
        tri_plane = self.head(tri_plane)        # 256
        tri_plane = self.final_head(tri_plane)
        return tri_plane, r_list


class TriPlanefeat_Encoder(Module):

    def __init__(self, inp_ch, seq2seq=False, res=None, use_gru=False):
        super(TriPlanefeat_Encoder, self).__init__()
        self.seq2seq = seq2seq
        blocks = get_blocks(num_layers=50)
        self.res = res
        self.use_gru = use_gru
        self.face_pool = None if res is None else torch.nn.AdaptiveAvgPool2d((res, res))
        self.input_layer = Sequential(Conv2d(inp_ch, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck_IR_SE(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
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

        self.outconv0 = nn.Conv2d(384, 32, kernel_size=1, padding=0)  # 16
        self.outconv1 = nn.Conv2d(384, 512, kernel_size=1, padding=0)  # 32
        self.outconv2 = nn.Conv2d(256, 512, kernel_size=1, padding=0)  # 64
        self.outconv3 = nn.Conv2d(96, 256, kernel_size=1, padding=0)  # 128

    def forward_onlyEncoder(self, x):
        assert x.dim() == 5
        x = x.flatten(0, 1)

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
        return [x, c0, c1, c2, c3]

    def forward_onlyDecoder(self, T, cond_list, r_list=None):
        x, c0, c1, c2, c3 = cond_list
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


class TriPlaneSFTfeat_Encoder(Module):

    def __init__(self, inp_ch, sft_half=True, res=None, use_gru=False):
        super(TriPlaneSFTfeat_Encoder, self).__init__()
        self.sft_half = sft_half
        blocks = get_blocks(num_layers=50)
        self.res = res
        self.use_gru = use_gru
        self.face_pool = None if res is None else torch.nn.AdaptiveAvgPool2d((res, res))
        self.input_layer = Sequential(Conv2d(inp_ch, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck_IR_SE(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

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

class TriPlaneSFTfeatConvFusion_Encoder(Module):

    def __init__(self, inp_ch, sft_half=True, res=None, use_gru=False):
        super(TriPlaneSFTfeatConvFusion_Encoder, self).__init__()
        self.sft_half = sft_half
        blocks = get_blocks(num_layers=50)
        self.res = res
        self.use_gru = use_gru
        assert use_gru
        self.face_pool = None if res is None else torch.nn.AdaptiveAvgPool2d((res, res))
        self.input_layer = Sequential(Conv2d(inp_ch, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck_IR_SE(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.up1 = (TimeFusion_Up(1024, 512, upscale_factor=1))
        self.up2 = (TimeFusion_Up(384, 384))
        self.up3 = (TimeFusion_Up(224, 256))
        self.up4 = (TimeFusion_Up(128, 96))

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
            raise NotImplementedError