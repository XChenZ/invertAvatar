import cv2
import click
import torch
import os
import copy

import legacy
import numpy as np
from tqdm import tqdm
import dnnlib
import imageio
from torch_utils import misc
from training_avatar_texture.triplane_v20 import TriPlaneGenerator
from encoder_inversion.models.uvnet import inversionNet
from training_avatar_texture.dataset_new import ImageFolderDataset
from data_preprocess.FaceVerse.renderer import Faceverse_manager

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img


def setup_snapshot_image_grid(training_set, random_seed=0, grid_size=(2, 5), specific_name_ls=None):
    rnd = np.random.RandomState(random_seed)
    gh, gw = grid_size
    sample_num = gh * gw - 2
    if True:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(training_set)):
            name = training_set._image_fnames[training_set._raw_idx[idx]]
            if name not in label_groups:
                label_groups[name] = []
            label_groups[name].append(idx)

        # Reorder.
        label_order = list(label_groups.keys())
        rnd.shuffle(label_order)
        for label in label_order:
            rnd.shuffle(label_groups[label])

        grid_indices = []
        # Organize into grid.

        if specific_name_ls is not None:
            for name in specific_name_ls:
                indices = label_groups[name]
                grid_indices += [indices[0]]
            sample_num = max(sample_num - len(specific_name_ls), 0)

        for y in range(sample_num):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[0]]

    # Load data.
    names, images, labels, verts = zip(*[training_set[i][:4] for i in grid_indices])
    return names, images, labels, verts


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def run_video_animation(network_pkl, outdir, reload_modules):

    # outdir = os.path.join(outdir, fname)
    device = torch.device('cuda')

    # 保存当前工作路径
    # current_path = os.getcwd()
    # os.chdir('/media/zxc/10T/Code/animatable_eg3d')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G'].to(device)
    # os.chdir(current_path)

    if reload_modules:
        print("Reloading Modules!")
        generator = TriPlaneGenerator(*G.generator.init_args, **G.generator.init_kwargs).eval().requires_grad_(False)
        G_new = inversionNet(generator=generator, encoding_triplane=True, encoding_texture=True).train().requires_grad_(False).to(device)

        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G = G_new
        G.unet_encoder.triplane_unet.input_layer.eval(); G.unet_encoder.triplane_unet.body.eval()
        G.unet_encoder.texture_unet.input_layer.eval(); G.unet_encoder.texture_unet.body.eval()

    os.makedirs(outdir, exist_ok=True)

    faceverser = Faceverse_manager(device=device, base_coeff=None)
    input_mask = False
    sequential_sampling = False

    root_path = './data/src_data/dataset/images512x512'
    imgFolder_dataset = ImageFolderDataset(path=root_path,
                                           mesh_path=os.path.join(os.path.dirname(root_path), 'orthRender256x256_face_eye'),
                                           label_file='dataset_realcam.json',
                                           load_uv=True, load_bg=input_mask, return_name=True,
                                           fvcoeffs_path=os.path.join(os.path.dirname(root_path), 'coeffs'))
    selected_videos_info = [[
        "Hillary",
        [
            "00000000.png",
            "00000030.png",
            "00000060.png",
            "00000090.png"
        ]
    ]]

    animation_root_path = './data/tgt_data/dataset/images512x512'
    ani_vid_ls = ['Obama']

    animation_imgFolder_dataset = ImageFolderDataset(path=animation_root_path,
                                                     mesh_path=os.path.join(os.path.dirname(animation_root_path), 'orthRender256x256_face_eye'),
                                                     label_file='dataset_realcam.json', load_uv=False, return_name=True,
                                                     fvcoeffs_path=os.path.join(os.path.dirname(animation_root_path), 'coeffs'))


    save_viddir = os.path.join(outdir, 'all_vid')
    os.makedirs(save_viddir, exist_ok=True)

    online_animation = True and ani_vid_ls is not None

    with torch.no_grad():
        for video_info in tqdm(selected_videos_info):
            video_name, ref_image_ls = video_info
            # if not video_name == 'Clip+PBpM7A2LbCc+P0+C1+F10794-11240': continue
            ref_fname_ls = ['%s/%s' % (video_name, f) for f in ref_image_ls]#[::-1]

            if len(ref_fname_ls) > 4 and not len(ref_fname_ls) % 4 == 0:
                ref_fname_ls = ref_fname_ls + ref_fname_ls[:(4-len(ref_fname_ls) % 4)]

            names, images, labels, verts = setup_snapshot_image_grid(training_set=imgFolder_dataset, random_seed=3, grid_size=(len(ref_fname_ls), 1),
                                                                     specific_name_ls=ref_fname_ls)

            if len(ref_fname_ls) < 4:
                assert len(ref_fname_ls) in [1, 2], len(ref_fname_ls)
                phase_real_img = {k: torch.from_numpy(np.stack([image[k] for image in images])).to(device) for k in images[0].keys()}
                phase_real_img = {k: torch.cat([phase_real_img[k] for _ in range(4 // len(ref_fname_ls))], dim=0) for k in phase_real_img.keys()}
                phase_real_img = {k: (phase_real_img[k].to(device).to(torch.float32) / 127.5 - 1) if phase_real_img[k].dtype is torch.uint8 else
                phase_real_img[k].to(device).to(torch.float32) for k in phase_real_img.keys()}
                phase_real_c = torch.from_numpy(np.stack(labels)).to(device)
                phase_real_c = torch.cat([phase_real_c for _ in range(4 // len(ref_fname_ls))], dim=0)
                phase_real_v = {k: torch.from_numpy(np.stack([vert[k] for vert in verts])).to(device) for k in verts[0].keys()}
                phase_real_v = {k: torch.cat([phase_real_v[k] for _ in range(4 // len(ref_fname_ls))], dim=0) for k in phase_real_v.keys()}
            else:
                phase_real_img = {k: torch.from_numpy(np.stack([image[k] for image in images])).to(device) for k in images[0].keys()}
                phase_real_img = {k: (phase_real_img[k].to(device).to(torch.float32) / 127.5 - 1) if phase_real_img[k].dtype is torch.uint8 else
                phase_real_img[k].to(device).to(torch.float32) for k in phase_real_img.keys()}
                phase_real_c = torch.from_numpy(np.stack(labels)).to(device)
                phase_real_v = {k: torch.from_numpy(np.stack([vert[k] for vert in verts])).to(device) for k in verts[0].keys()}
            if phase_real_img['image'].shape[1] > 3:
                phase_real_img['image'] = phase_real_img['image'][:, :3] * phase_real_img['image'][:, 3:] + \
                                          torch.ones_like(phase_real_img['image'][:, :3]) * (1. - phase_real_img['image'][:, 3:])
            cv2.imwrite(os.path.join(outdir, video_name + '_inputs.png'), layout_grid(phase_real_img['image'], grid_w=phase_real_img['image'].shape[0], grid_h=1)[..., ::-1])

            ws = G.encode(phase_real_img['image'][:1])
            texture_feats = G.generator.texture_backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const')
            static_feats = G.generator.backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const')
            e4e_results = {'w': ws, 'texture': texture_feats, 'static': static_feats}

            r_list = [None, None]
            if len(ref_fname_ls) > 4:
                num_iter = len(ref_fname_ls) // 4

                for idx in range(num_iter):
                    if sequential_sampling: # 顺序采样
                        phase_real_img_ = {k: phase_real_img[k][4*idx:4*(idx+1)] for k in phase_real_img.keys()}
                        phase_real_c_ = phase_real_c[4 * idx:4 * (idx + 1)]
                        phase_real_v_ = {k: phase_real_v[k][4 * idx:4 * (idx + 1)] for k in phase_real_v.keys()}
                    else:   # 间隔采样
                        phase_real_img_ = {k: phase_real_img[k][idx::num_iter] for k in phase_real_img.keys()}
                        phase_real_c_ = phase_real_c[idx::num_iter]
                        phase_real_v_ = {k: phase_real_v[k][idx::num_iter] for k in phase_real_v.keys()}
                        assert phase_real_c_.shape[0] == 4, phase_real_c_.shape[0]
                    updated_e4e_results, r_list = G.AR_eval_forward(phase_real_img_, phase_real_c_, phase_real_v_, ws, r_list, e4e_results=e4e_results, return_fake=False)

            else:
                updated_e4e_results, r_list = G.AR_eval_forward(phase_real_img, phase_real_c, phase_real_v, ws, r_list, e4e_results=e4e_results, return_fake=False)

            if online_animation: faceverser.id_coeff = faceverser.recon_model.split_coeffs(phase_real_v['coeff'][:1])[0]
            dst_vid_ls = [video_name] if ani_vid_ls is None else copy.deepcopy(ani_vid_ls)

            for dst_vid in dst_vid_ls:
                save_dir = os.path.join(outdir, video_name+'_drivenby_'+dst_vid) if ani_vid_ls is not None else os.path.join(outdir, video_name)
                os.makedirs(save_dir, exist_ok=True)
                video_dir = os.path.join(animation_root_path, dst_vid)
                eval_fname_ls = ['%s/%s' % (dst_vid, f) for f in os.listdir(video_dir) if f.endswith('.png')]
                eval_fname_ls.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
                video_out = imageio.get_writer(save_dir.replace(outdir, save_viddir) + '.mp4', mode='I', fps=40, codec='libx264', bitrate='10M')

                for image_name in tqdm(eval_fname_ls):
                    gt_img, c, v = animation_imgFolder_dataset.get_by_name(image_name)
                    gt_img = torch.from_numpy(gt_img).unsqueeze(0)
                    gt_img = (gt_img.to(device).to(torch.float32) / 127.5 - 1) if gt_img.dtype is torch.uint8 else gt_img.to(device).to(torch.float32)

                    c = torch.from_numpy(c).to(device).unsqueeze(0)
                    v = {k: torch.from_numpy(v[k]).to(device).unsqueeze(0) for k in v.keys()}
                    if online_animation: v['uvcoords_image'] = faceverser.make_driven_rendering(v['coeff'], res=256)

                    out = G.generator.synthesis_withTexture(ws, updated_e4e_results['texture'], c, v, noise_mode='const', static_feats=updated_e4e_results['static'], evaluation=True)
                    out_imgs = [gt_img[:, :3], out['image']]

                    out_img = layout_grid(torch.cat(out_imgs, dim=0), grid_w=len(out_imgs), grid_h=1)
                    cv2.imwrite(os.path.join(save_dir, '%s' % image_name.split('/')[-1]), out_img[..., ::-1])

                    if video_out is not None: video_out.append_data(out_img)
                video_out.close()


if __name__ == "__main__":
    run_video_animation()