# import pickle
import torch
import torch.nn.functional as F
from torch import nn
# from torch_utils import persistence
from torch_utils import misc

import dnnlib
import legacy
from encoder_inversion.models.e4e import Encoder4Editing
from encoder_inversion.models.unet_encoders import TriPlaneSFTfeat_Encoder, TriPlanefeat_Encoder
from encoder_inversion.models.networks_styleunet import CondSynthesisNetwork_withGRU


class unet_encoder(nn.Module):
    def __init__(self, encoding_texture=False, encoding_triplane=False):
        super(unet_encoder, self).__init__()
        self.texture_unet = TriPlanefeat_Encoder(inp_ch=7, res=256, use_gru=True) if encoding_texture else None
        self.triplane_unet = TriPlaneSFTfeat_Encoder(inp_ch=6, res=256, use_gru=True) if encoding_triplane else None

    def forward(self, x):
        raise NotImplementedError
        return None


class inversionNet(nn.Module):
    def __init__(self, G_kwargs=None, generator=None, encoding_texture=True, encoding_triplane=False):
        super(inversionNet, self).__init__()
        # Define architecture
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        if generator is not None:
            self.generator = generator
        else:
            self.generator = dnnlib.util.construct_class_by_name(**G_kwargs).train().requires_grad_(False)
        self.register_buffer('latent_avg', self.generator.backbone.mapping.w_avg.reshape(1, 512))
        # self.load_gen(path_kwargs['path_generator'])
        self.n_styles = self.generator.texture_backbone.num_ws
        self.encoder = self.set_encoder(self.n_styles, inp_ch=3)
        self.unet_encoder = unet_encoder(encoding_texture=encoding_texture, encoding_triplane=encoding_triplane)
        self.register_buffer('black_uv_bg', -1 * torch.ones(1, 3, 256, 256, dtype=torch.float32))
        # self.initialize_encoders(path_kwargs['path_irse50'])

    def load_gen(self, gan_path):
        # Gen_func = dnnlib.util.get_obj_by_name(G_kwargs['class_name'])
        # self.generator = Gen_func(**{k:G_kwargs[k] for k in G_kwargs.keys() if k != 'class_name'}).eval().requires_grad_(False)
        print('Loading generator weights from ' + gan_path)
        with dnnlib.util.open_url(gan_path) as f:
            load_G = legacy.load_network_pkl(f)['G_ema']
        # with open(gan_path, 'rb') as f:
        #     load_G = pickle.load(f)['G_ema'].eval()
        # self.generator = Gen_func(*load_G.init_args, **load_G.init_kwargs).eval().requires_grad_(False)
        print(self.generator.neural_rendering_resolution, load_G.neural_rendering_resolution)
        misc.copy_params_and_buffers(load_G, self.generator, require_all=True)
        self.generator.neural_rendering_resolution = load_G.neural_rendering_resolution
        self.generator.rendering_kwargs = load_G.rendering_kwargs
        self.latent_avg = self.generator.backbone.mapping.w_avg.reshape(1, 512)
        del load_G

    def print_parameter_numbers(self):
        num_params = sum(param.numel() for param in self.encoder.parameters())
        print("encoder parmeters number is :    ", num_params)

        num_params = sum(param.numel() for param in self.unet_encoder.triplane_unet.parameters())
        print("triplane_unet parmeters number is :    ", num_params)

        num_params = sum(param.numel() for param in self.unet_encoder.texture_unet.parameters())
        print("texture_unet parmeters number is :    ", num_params)

        num_params = sum(param.numel() for param in self.generator.parameters())
        print("generator parmeters number is :    ", num_params)

    def initialize_encoders(self, ir_se50_path, triplanenet_path=None):
        if self.unet_encoder.texture_unet is not None:
            misc.copy_params_and_buffers(self.generator.texture_backbone, self.unet_encoder.texture_unet, require_all=False)

        print('Loading encoders weights from irse50!')
        encoder_ckpt = torch.load(ir_se50_path, map_location='cpu')
        self.encoder.load_state_dict(encoder_ckpt, strict=False)

        if self.unet_encoder.triplane_unet is not None:
            if triplanenet_path is None:
                # alter cuz triplane encoder works with concatenated inputs
                shape = encoder_ckpt['input_layer.0.weight'].shape
                altered_input_layer = torch.randn(shape[0], 6, shape[2], shape[3], dtype=torch.float32)
                altered_input_layer[:, :3, :, :] = encoder_ckpt['input_layer.0.weight']
                encoder_ckpt['input_layer.0.weight'] = altered_input_layer
                self.unet_encoder.triplane_unet.load_state_dict(encoder_ckpt, strict=False)
            else:
                print('Loading triplane_unet weights from Triplanenet!')
                checkpoint = torch.load(triplanenet_path, map_location='cpu')['state_dict']
                model_dict = self.unet_encoder.triplane_unet.state_dict()
                pretrained_dict = {k[len('triplanenet_encoder') + 1:]: v for k, v in checkpoint.items() if k.split('.')[0] == 'triplanenet_encoder'}
                model_dict.update(pretrained_dict)
                self.unet_encoder.triplane_unet.load_state_dict(model_dict, strict=False)

        del encoder_ckpt

    def set_encoder(self, n_styles, inp_ch):
        encoder_ws = Encoder4Editing(n_styles, inp_ch)
        return encoder_ws

    def switch_grad(self, nerf_requires_grad=False):
        for _i in range(self.encoder.middle_ind):
            for p in self.encoder.styles[_i].parameters():
                p.requires_grad = nerf_requires_grad

    def encode(self, x):
        if x.shape[-1] != 256:
            x = self.face_pool(x)
        if type(self.encoder) is nn.ModuleList:
            codes = torch.cat([encoder(x) for encoder in self.encoder], dim=1)
        else:
            codes = self.encoder(x)
        codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        return codes

    def get_unet_uvinput(self, uv, delta_x):
        uv_gttex, uv_pverts = uv.split(3, dim=1)  # [1, 3, H, W]
        uv_delta = F.grid_sample(delta_x, uv_pverts.permute(0, 2, 3, 1)[..., :2], mode='bilinear', align_corners=False)
        uv_delta = uv_delta * uv_pverts[:, -1:] + self.black_uv_bg * (1 - uv_pverts[:, -1:])
        return torch.cat([uv_gttex, uv_delta, uv_pverts[:, -1:]], dim=1)

    def forward(self, x, cam, v, e4e_results=None, visualize_input=False, return_feats=False):
        with torch.no_grad():
            if e4e_results is None:
                ws = self.encode(x['image'][:, :3])
                e4e_texture_feats = self.generator.texture_backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const')
                e4e_static_feats = self.generator.backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const')
            else:
                ws, e4e_texture_feats, e4e_static_feats = e4e_results['w'], e4e_results['texture'], e4e_results['static']
            y_hat_e4e = self.generator.synthesis_withTexture(ws, e4e_texture_feats, cam, v, static_feats=e4e_static_feats, noise_mode='const')
            if not y_hat_e4e['image'].shape[-1] == x['image'].shape[-1]:
                y_hat_e4e['image'] = torch.nn.functional.interpolate(y_hat_e4e['image'], size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
            delta_x = y_hat_e4e['image'] - x['image'][:, :3]

        assert x['uv'] is not None
        x_input = self.get_unet_uvinput(x['uv'], delta_x)

        texture_offsets = self.unet_encoder.texture_unet(x_input, return_list=True)
        if len(texture_offsets) == len(e4e_texture_feats):
            texture_feats = [feat + offset for feat, offset in zip(e4e_texture_feats, texture_offsets)]
        else:  # len(texture_offsets) < len(e4e_texture_feats)
            texture_feats = [feat + offset for feat, offset in
                             zip(e4e_texture_feats, texture_offsets[:len(texture_offsets)])] + e4e_texture_feats[len(texture_offsets):]

        triplane_feat_offsets = self.unet_encoder.triplane_unet(torch.cat([x['image'][:, :3], delta_x], dim=1))
        static_feats = self.generator.backbone.synthesis(ws, cond_list=None, return_list=True, feat_conditions=triplane_feat_offsets,
                                                         update_emas=False, noise_mode='const')

        output = self.generator.synthesis_withTexture(ws, texture_feats, cam, v, static_feats=static_feats, noise_mode='const')
        if return_feats:
            output['texture'] = texture_feats
            output['static'] = static_feats
        output['w'] = ws
        output['e4e_image'] = y_hat_e4e['image']
        if visualize_input: output['x_input'] = torch.clamp(x_input, min=-1, max=1)
        return output

    @torch.no_grad()
    def AR_eval_forward(self, x, vid_c, vid_v, ws, r_list, e4e_results=None, return_fake=False):
        '''
        ws: [1, ]
        frames: 'image' [T, C, H, W]
        '''
        # B, T = real_vid_c.shape[:2]
        T = vid_c.shape[0]
        # vid_c = real_vid_c.flatten(0, 1)
        # vid_v = real_vid_v.flatten(0, 1)

        if ws is None:
            ws = self.encode(x['image'][0:1])
        if e4e_results is None:
            texture_feats = self.generator.texture_backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const')
            static_feats = self.generator.backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const')
        else:
            texture_feats, static_feats = e4e_results['texture'], e4e_results['static']
        vid_ws = ws.expand(T, -1, -1)

        # if real_vid_uv is None:
        y_hat_e4e = self.generator.synthesis_withTexture(vid_ws, [feat.expand(T, -1, -1, -1) for feat in texture_feats], vid_c, vid_v,
                                                         static_feats=[feat.expand(T, -1, -1, -1) for feat in static_feats], noise_mode='const')
        delta_x = y_hat_e4e['image'] - x['image'][:, :3]
        real_vid_uv = self.get_unet_uvinput(x['uv'], delta_x)
        triplane_input = torch.cat([x['image'][:, :3], delta_x], dim=-3)

        texture_offsets, r_list[0] = self.unet_encoder.texture_unet(real_vid_uv.unsqueeze(0), r_list=r_list[0], return_list=True)
        if len(texture_offsets) == len(texture_feats):
            texture_feats = [feat + offset for feat, offset in zip(texture_feats, texture_offsets)]
        else:  # len(texture_offsets) < len(e4e_texture_feats)
            texture_feats = [feat + offset for feat, offset in zip(texture_feats, texture_offsets[:len(texture_offsets)])] + texture_feats[len(texture_offsets):]

        triplane_feat_offsets, r_list[1] = self.unet_encoder.triplane_unet(triplane_input.unsqueeze(0), r_list=r_list[1])
        static_feats = self.generator.backbone.synthesis(ws, cond_list=None, return_list=True, feat_conditions=triplane_feat_offsets,
                                               update_emas=False, noise_mode='const')

        updated_e4e_results = {'w': ws, 'texture': texture_feats, 'static': static_feats}
        if not return_fake:
            return updated_e4e_results, r_list
        else:
            fake_imgs = self.generator.synthesis_withTexture(vid_ws, [feat.expand(T, -1, -1, -1) for feat in updated_e4e_results['texture']], vid_c, vid_v,
                                                             static_feats=[feat.expand(T, -1, -1, -1) for feat in updated_e4e_results['static']],
                                                             noise_mode='const', evaluation=True)['image']
            return updated_e4e_results, {'e4e': y_hat_e4e['image'], 'image': fake_imgs, 'x_input': real_vid_uv}, r_list

    @staticmethod
    def __get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt