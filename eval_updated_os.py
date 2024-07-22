import cv2
import click
import torch
import os

import legacy
import numpy as np
from tqdm import tqdm
import dnnlib
import json
import imageio
from torch_utils import misc
from camera_utils import LookAtPoseSampler
from training_avatar_texture.triplane_v20 import TriPlaneGenerator
from encoder_inversion.models.uvnet_new import inversionNet
from training_avatar_texture.dataset_new import ImageFolderDataset
from data_preprocess.FaceVerse.renderer import Faceverse_manager


def rotate_by_theta_along_y(theta):
    tform = np.eye(4).astype(np.float32)
    tform[0, 0] = tform[2, 2] = np.cos(theta)
    tform[0, 2] = -np.sin(theta)
    tform[2, 0] = -tform[0, 2]
    return tform


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

    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G'].to(device)
    if reload_modules:
        print("Reloading Modules!")
        generator = TriPlaneGenerator(*G.generator.init_args, **G.generator.init_kwargs).eval().requires_grad_(False)
        G_new = inversionNet(generator=generator, encoding_triplane=True, encoding_texture=True).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G = G_new

    os.makedirs(outdir, exist_ok=True)

    cam_pivot = torch.tensor(G.generator.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.generator.rendering_kwargs.get('avg_camera_radius', 2.7)
    pitch_range, yaw_range = 0.25, 0.35

    input_mask = False
    online_animation = True
    corr_json = None

    root_path = './data/src_data/dataset/images512x512'
    src_cls_names = ['os']
    animation_root_path = './data/tgt_data/dataset/images512x512'
    target_cls_names = ['Obama']
    source_fname_ls = ['demo.png']

    same_id_reenact = False
    save_root_dir = os.path.join(outdir, 'all_vid')
    os.makedirs(save_root_dir, exist_ok=True)

    imgFolder_dataset = ImageFolderDataset(path=root_path,
                                           mesh_path=os.path.join(os.path.dirname(root_path), 'orthRender256x256_face_eye'),
                                           label_file='dataset_realcam.json',
                                           load_uv=True, return_name=True, load_bg=input_mask,
                                           fvcoeffs_path=os.path.join(os.path.dirname(root_path), 'coeffs') if online_animation else None)

    animation_imgFolder_dataset = ImageFolderDataset(path=animation_root_path,
                                           mesh_path=os.path.join(os.path.dirname(animation_root_path), 'orthRender256x256_face_eye'),
                                           label_file='dataset_realcam.json', load_uv=False, return_name=True,
                                           fvcoeffs_path=os.path.join(os.path.dirname(animation_root_path), 'coeffs') if online_animation else None)
    faceverser = Faceverse_manager(device=device, base_coeff=None)

    if same_id_reenact:    # 自监督
        iter_cls_names = []
        for src_cls_name, target_cls_name in zip(src_cls_names, target_cls_names):
            for src_fname in source_fname_ls:
                iter_cls_names.append([src_cls_name, src_fname, target_cls_name])
    else:
        iter_cls_names = []
        if corr_json is None:
            for src_cls_name in src_cls_names:
                for src_fname in source_fname_ls:
                # for src_fname in os.listdir(os.path.join(root_path, src_cls_name)):
                    if not src_fname.endswith('png'): continue
                    for target_cls_name in target_cls_names:
                        iter_cls_names.append([src_cls_name, src_fname, target_cls_name])
        else:
            corr_json = json.loads(open(corr_json).read())
            for case in corr_json:
                src_case_name = case[0].split('/')[-1].split('.')[0]
                tgt_case_name = case[1].split('/')[-1].split('.')[0]
                iter_cls_names.append(['source', src_case_name+'.png', tgt_case_name])


    for src_cls_name, source_fname, target_cls_name in tqdm(iter_cls_names):
        print('Start Process', '%s-%s_drivenby_%s' % (src_cls_name, source_fname.split('.')[0], target_cls_name))
        if not '%s/%s' % (src_cls_name, source_fname) in imgFolder_dataset._image_fnames:
            print('Cannot find %s in Dataset.' % source_fname)
            continue
        names, images, labels, verts = setup_snapshot_image_grid(training_set=imgFolder_dataset, random_seed=3, grid_size=(1, 1),
                                                                 specific_name_ls=['%s/%s' % (src_cls_name, source_fname)])
        phase_real_img = {k: torch.from_numpy(np.stack([image[k] for image in images])).to(device) for k in images[0].keys()}
        phase_real_img = {k: (phase_real_img[k].to(device).to(torch.float32) / 127.5 - 1) if phase_real_img[k].dtype is torch.uint8 else
        phase_real_img[k].to(device).to(torch.float32) for k in phase_real_img.keys()}
        phase_real_c = torch.from_numpy(np.stack(labels)).to(device)
        phase_real_v = {k: torch.from_numpy(np.stack([vert[k] for vert in verts])).to(device) for k in verts[0].keys()}
        if phase_real_img['image'].shape[1] > 3:
            phase_real_img['image'] = phase_real_img['image'][:, :3] * phase_real_img['image'][:, 3:] + \
                                      torch.ones_like(phase_real_img['image'][:, :3]) * (1. - phase_real_img['image'][:, 3:])
        cv2.imwrite(os.path.join(outdir, '%s_drivenby_%s' % (source_fname.split('.')[0], target_cls_name) + '_inputs.png'),
                    layout_grid(phase_real_img['image'], grid_w=phase_real_img['image'].shape[0], grid_h=1)[..., ::-1])

        ws = G.encode(phase_real_img['image'])
        texture_feats = G.generator.texture_backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const')
        static_feats = G.generator.backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const')
        e4e_results = {'w': ws, 'texture': texture_feats, 'static': static_feats}

        os_updated_e4e_results = G({'image': phase_real_img['image'], 'uv': phase_real_img['uv']}, phase_real_c,
                                      {'uvcoords_image': phase_real_v['uvcoords_image']}, e4e_results=e4e_results,
                                      return_feats=True)
        os_updated_e4e_results['static'] = e4e_results['static'][:-1] + os_updated_e4e_results['static'][-1:]
        eval_fname_ls = ['%s/%s' % (target_cls_name, f) for f in os.listdir(os.path.join(animation_root_path, target_cls_name)) if f.endswith('.png')]

        eval_fname_ls.sort(key=lambda x:int(x.split('/')[-1].split('.')[0]))
        if online_animation: faceverser.id_coeff = faceverser.recon_model.split_coeffs(phase_real_v['coeff'])[0]
        video_out = imageio.get_writer(os.path.join(save_root_dir, '%s_drivenby_%s' % (source_fname.split('.')[0], target_cls_name) + '.mp4'),
                                       mode='I', fps=25, codec='libx264', bitrate='10M') if save_root_dir is not None else None

        for image_name in tqdm(eval_fname_ls):
            gt_img, c, v = animation_imgFolder_dataset.get_by_name(image_name)

            gt_img = torch.from_numpy(gt_img).unsqueeze(0)
            gt_img = (gt_img.to(device).to(torch.float32) / 127.5 - 1) if gt_img.dtype is torch.uint8 else gt_img.to(device).to(torch.float32)

            c = torch.from_numpy(c).to(device).unsqueeze(0)
            v = {k: torch.from_numpy(v[k]).to(device).float().unsqueeze(0) for k in v.keys()}
            if online_animation: v['uvcoords_image'] = faceverser.make_driven_rendering(v['coeff'], res=256)
            out = G.generator.synthesis_withTexture(ws, os_updated_e4e_results['texture'], c, v, noise_mode='const', static_feats=os_updated_e4e_results['static'], evaluation=True)
            out_imgs = [gt_img[:, :3], out['image']]

            if same_id_reenact:
                save_dir = os.path.join(outdir, src_cls_name)
            else:
                save_dir = os.path.join(outdir, '%s_drivenby_%s' % (source_fname.split('.')[0], target_cls_name))
            os.makedirs(save_dir, exist_ok=True)
            out_img = layout_grid(torch.cat(out_imgs, dim=0), grid_w=len(out_imgs), grid_h=1)
            cv2.imwrite(os.path.join(save_dir, image_name.split('/')[-1]), out_img[..., ::-1])

            video_out.append_data(out_img)

        video_out.close()

        if True:  # only free view
            vert = {'uvcoords_image': phase_real_v['uvcoords_image'][:1]}
            mv_video_out = imageio.get_writer(os.path.join(save_root_dir, source_fname.split('.')[0]) + '_mv.mp4', mode='I', fps=25 // 5, codec='libx264', bitrate='10M')
            for k in tqdm(range(240)[::10]):
                cam2world_pose = LookAtPoseSampler.sample(3.14 / 2 + yaw_range * np.sin(2 * 3.14 * k / (240)),
                                                          3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * k / (240)),
                                                          cam_pivot, radius=cam_radius, device=device)
                focal_length = 4.2647
                intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                out = G.generator.synthesis_withTexture(ws, os_updated_e4e_results['texture'], camera_params, vert,
                                                        static_feats=os_updated_e4e_results['static'], noise_mode='const')

                mv_video_out.append_data(layout_grid(out['image'], grid_w=1, grid_h=1))
            mv_video_out.close()


if __name__ == "__main__":
    run_video_animation()