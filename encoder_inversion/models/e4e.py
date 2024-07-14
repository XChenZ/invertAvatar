# import torch
# from torch import nn
# from torch_utils import misc
# import yaml
# from enum import Enum
# from mmap import ACCESS_DEFAULT
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
# from torch_utils import persistence

# import dnnlib
from .helpers import get_blocks, Flatten, bottleneck_IR_SE
from training.networks_stylegan2 import FullyConnectedLayer
# import legacy
# from training.triplane import TriPlaneGenerator as Generator


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [
            Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        ]
        for _ in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = FullyConnectedLayer(in_features=out_c, out_features=out_c, bias=True, activation='linear', lr_multiplier=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


def _upsample_add(x, y):
    """Upsample and add two feature maps.
    Args:
      x: (Variable) top feature map to be upsampled.
      y: (Variable) lateral feature map.
    Returns:
      (Variable) added feature map.
    Note in PyTorch, when input size is odd, the upsampled feature map
    with `F.upsample(..., scale_factor=2, mode='nearest')`
    maybe not equal to the lateral feature map size.
    e.g.
    original input size: [N,_,15,15] ->
    conv2d feature map size: [N,_,8,8] ->
    upsampled feature map size: [N,_,16,16]
    So we choose bilinear upsample which supports arbitrary output sizes.
    """
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y


class Encoder4Editing(Module):
    def __init__(self, n_styles=18, inp_ch=3):
        super(Encoder4Editing, self).__init__()

        blocks = get_blocks(num_layers=50)
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

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        # self.progressive_stage = ProgressiveStage.Inference
        # self.cam_block = GradualStyleBlock(512, 25, 16)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        # stage = self.progressive_stage.value
        features = c3
        for i in range(1, self.style_count):  # Infer additional deltas
        # for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i
        # cam = self.cam_block(c3)
        return w


class e4e(nn.Module):

    def __init__(self, 
        n_styles = 14,
        if_load_weights = True,
        generator = None,
        set_restyle_encoder = False,
        **unused
    ):
        super(e4e, self).__init__()
        # Define architecture
        self.n_styles = n_styles
        self.encoder = self.set_encoder(n_styles, inp_ch=3)
        # self.restyle_encoder = self.set_encoder(n_styles, inp_ch=6) if set_restyle_encoder else None
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.generator = generator.train().requires_grad_(False) if generator is not None else None
        self.register_buffer('latent_avg', self.generator.backbone.mapping.w_avg.reshape(1, 512))
        # if if_load_weights:
        #     self.load_weights(G_kwargs)

    def set_encoder(self, n_styles, inp_ch):
        encoder_ws = Encoder4Editing(n_styles, inp_ch)
        if False:
            encoder_texture_ws = Encoder4Editing(n_styles, inp_ch)
            return nn.ModuleList([encoder_ws, encoder_texture_ws])
        else:
            return encoder_ws

    def switch_grad(self, nerf_requires_grad=False):
        for _i in range(self.encoder.middle_ind):
            for p in self.encoder.styles[_i].parameters():
                p.requires_grad = nerf_requires_grad

    # def load_weights(self, path_kwargs):
    #     with open(path_kwargs['path_generator_kwargs'], 'r') as f:
    #         cfg_dict = yaml.load(f, Loader=yaml.SafeLoader)
    #     # self.generator = Generator(**cfg_dict['G_kwargs'])
    #     self.generator = dnnlib.util.construct_class_by_name(**cfg_dict['G_kwargs']).train().requires_grad_(False)
    #     self.generator.register_buffer('dataset_label_std', torch.zeros(25))
    #     self.generator.neural_rendering_resolution = 128
    #     # self.generator = Generator(**G_kwargs)
    #     # ckpt = torch.load(path_kwargs['path_generator'], map_location='cpu')['G_ema']
    #     # self.generator.load_state_dict(ckpt, strict=True)
    #     with dnnlib.util.open_url(path_kwargs['path_generator']) as f:
    #         self.generator = legacy.load_network_pkl(f)['G_ema'] # type: ignore
    #     self.register_buffer('latent_avg', self.generator.backbone.mapping.w_avg.reshape(1, 512))
    #     print(f'Load pre-trained 3D-GAN from {path_kwargs["path_generator"]}')

    def encode(self, x):
        if x.shape[-1] != 256:
            x = self.face_pool(x)
        if type(self.encoder) is nn.ModuleList:
            codes = torch.cat([encoder(x) for encoder in self.encoder], dim=1)
        else:
            codes = self.encoder(x)
        codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        return codes

    def forward(self, x, cam, v, only_w=False):
        ws = self.encode(x)
        # ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        # {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}
        output = self.generator.synthesis(ws, cam, v, noise_mode='const') if not only_w else dict()
        output['w'] = ws
        output['c'] = cam
        return output

    def restyle_forward(self, x, cam, v, init_ws):
        assert self.restyle_encoder is not None
        delta_ws = torch.cat([encoder(x) for encoder in self.restyle_encoder], dim=1)
        ws = delta_ws + init_ws
        output = self.generator.synthesis(ws, cam, v)
        output['w'] = ws
        output['c'] = cam
        return output

    @staticmethod
    def __get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt



class LatentCodesDiscriminator(nn.Module):
    def __init__(self, style_dim, n_mlp):
        super().__init__()

        self.style_dim = style_dim

        layers = []
        for i in range(n_mlp-1):
            layers.append(
                nn.Linear(style_dim, style_dim)
            )
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(512, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, w):
        return self.mlp(w)


class LatentCodesPool:
    """This class implements latent codes buffer that stores previously generated w latent codes.
    This buffer enables us to update discriminators using a history of generated w's
    rather than the ones produced by the latest encoder.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_ws = 0
            self.ws = []

    def query(self, ws):
        """Return w's from the pool.
        Parameters:
            ws: the latest generated w's from the generator
        Returns w's from the buffer.
        By 50/100, the buffer will return input w's.
        By 50/100, the buffer will return w's previously stored in the buffer,
        and insert the current w's to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return ws
        return_ws = []
        for w in ws:  # ws.shape: (batch, 512) or (batch, n_latent, 512)
            # w = torch.unsqueeze(image.data, 0)
            if w.ndim == 2:
                i = random.randint(0, len(w) - 1)  # apply a random latent index as a candidate
                w = w[i]
            self.handle_w(w, return_ws)
        return_ws = torch.stack(return_ws, 0)   # collect all the images and return
        return return_ws

    def handle_w(self, w, return_ws):
        if self.num_ws < self.pool_size:  # if the buffer is not full; keep inserting current codes to the buffer
            self.num_ws = self.num_ws + 1
            self.ws.append(w)
            return_ws.append(w)
        else:
            p = random.uniform(0, 1)
            if p > 0.5:  # by 50% chance, the buffer will return a previously stored latent code, and insert the current code into the buffer
                random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                tmp = self.ws[random_id].clone()
                self.ws[random_id] = w
                return_ws.append(tmp)
            else:  # by another 50% chance, the buffer will return the current image
                return_ws.append(w)
