import cv2
import click
import torch
import os
import legacy
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, utils
import dnnlib
import json
import imageio
import re
from typing import List, Optional, Tuple, Union
from torch_utils import misc
from training_avatar_texture.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from FaceVerse.renderer import Faceverse_manager
from torch.utils.data import DataLoader
from training_avatar_texture.triplane_v20 import TriPlaneGenerator
from torch.utils.data import Dataset

class ImagesDataset(Dataset):
    def __init__(self, source_root, label_path, mesh_path=None, gtlabel_path=None, source_transform=None, resolution=(512, 512), skip=1,
                 return_vert=False, idx_range=None, fvcoeffs_path=None):
        self.source_transform = source_transform
        self.resolution = resolution
        self.return_vert = return_vert
        with open(label_path, 'rb') as f:
            labels = json.load(f)['labels']
        self.mesh_path = mesh_path
        self._raw_cams = dict(json.loads(open(gtlabel_path).read())['labels'])
        self._coeff_path = fvcoeffs_path
        self.load_coeff = fvcoeffs_path is not None

        labels = [label for label in labels if label[0] in self._raw_cams.keys()]
        labels.sort(key=lambda x: int(x[0].split('.')[0].split('/')[-1]))
        if idx_range is not None:
            if len(idx_range) == 3:
                prefix, min_, max_ = idx_range
                labels = [label for label in labels if (int(label[0].split('/')[-1].split('.')[0]) in range(min_, max_ + 1)) and (label[0].split('/')[0] == prefix)]
            else:
                min_, max_ = idx_range
                labels = [label for label in labels if int(label[0].split('/')[-1].split('.')[0]) in range(min_, max_ + 1)]
        self.labels = dict(labels)

        self.source_paths = [[inst[0].split('.')[0], os.path.join(source_root, inst[0])] for inst in labels]
        self.source_paths = self.source_paths[::skip]

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path = self.source_paths[index]

        from_im = Image.open(from_path).convert('RGB')
        if from_im.size != self.resolution:
            from_im = from_im.resize(self.resolution)

        if self.source_transform:
            from_im = self.source_transform(from_im)

        out = [from_im, self.get_label(fname)[0]]
        if self.return_vert:
            out.append(self.get_geo(fname))

        return out

    def get_label(self, idx):
        label = np.asarray([self.labels[idx + '.png']], dtype=np.float32)
        cam = np.asarray([self._raw_cams[idx + '.png']], dtype=np.float32)
        return torch.tensor(np.concatenate([label, cam], axis=1))

    def get_gtlabel(self, idx):
        return self.get_label(idx)

    def get_geo(self, idx):
        uvcoords_image = np.load(os.path.join(self.mesh_path, idx+'.npy'))[..., :3] # [HW3] 前两维date range(-1, 1)，第三维是face_mask，最后一维是render_mask
        uvcoords_image[..., -1][uvcoords_image[..., -1] < 0.5] = 0; uvcoords_image[..., -1][uvcoords_image[..., -1] >= 0.5] = 1
        out = {'uvcoords_image': torch.tensor(uvcoords_image.copy()).float()}
        if self.load_coeff: out['coeff'] = np.load(os.path.join(self._coeff_path, idx+'.npy')).astype(np.float32)

        return out
# ----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.
    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')


# ----------------------------------------------------------------------------

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


@click.command()
@click.option("--drive_root", type=str, default=None)
@click.option("--fname", type=str, default='reenact.mp4')
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--grid', 'grid_dims', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1, 1))
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=True, show_default=True)
@click.option('--lms_cond', help='If condition 2d landmarks?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--fixed_camera', help='If fix camera poses?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--num_frames', help='number of testing frames', type=int, required=False, metavar='BOOL', default=500, show_default=True)
def run_video_animation(drive_root, fname, network_pkl, grid_dims, seeds, outdir, fov_deg, truncation_psi, truncation_cutoff, reload_modules,
                        lms_cond, fixed_camera, num_frames):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]
    mp4 = os.path.join(outdir, fname+'.mp4')
    device = torch.device('cuda')

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    # total = sum([param.nelement() for param in G.parameters()])
    # print('Number of parameter: % .4fM' % (total / 1024 / 1024))

    os.makedirs(outdir, exist_ok=True)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    video_out = imageio.get_writer(mp4, mode='I', fps=25, codec='libx264', bitrate='12M')

    ws = []
    for seed in seeds:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        w = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        ws.append(w)

    with open(os.path.join(drive_root, 'dataset.json'), 'rb') as f:
        label_list = json.load(f)['labels']

    # img_list = sorted(glob.glob(drive_root + '/*.png'))
    # vert_root = lms_root = drive_root
    # for k, img_path in tqdm(enumerate([img for img in img_list])):
    #     if k > num_frames:
    #         break
    #     if k < 1:
    #         continue
    #
    #     img_path = img_list[k]
    #     img_render = Image.open(img_path)
    #     target_img = np.array(img_render.crop((0, 0, 512, 512)))
    #     target_img = torch.tensor(target_img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    #     target_img = target_img / 127.5 - 1.
    #
    #     img_id = os.path.basename(img_path).split('.')[0]
    #     vert_path = vert_root + f'/{img_id}.obj'
    #     v = []
    #     with open(vert_path, "r") as f:
    #         while True:
    #             line = f.readline()
    #             if line == "":
    #                 break
    #             if line[:2] == "v ":
    #                 v.append([float(x) for x in line.split()[1:]])
    #     v = np.array(v).reshape((-1, 3))
    #     v = torch.from_numpy(v).cuda().float().unsqueeze(0)
    #
    #     if lms_cond:
    #         lms_path = lms_root + f'/{img_id}_kpt2d.txt'
    #         lms = np.loadtxt(lms_path)
    #         lms = torch.from_numpy(lms).cuda().float().unsqueeze(0)
    #         v = torch.cat((v, lms), 1)
    #
    #     imgs = [target_img[0]]
    #     for idx, seed in enumerate(seeds):
    #         w = ws[idx]
    #         camera_params = (np.array(label_list[k - 1][1]) + np.array(label_list[k][1]) + np.array(
    #             label_list[k + 1][1])) / 3  # add camera smoothness
    #         camera_params = torch.tensor(camera_params).unsqueeze(0).float().to(device)
    #         if fixed_camera:
    #             camera_params = conditioning_params
    #         G.rendering_kwargs['return_normal'] = False
    #         img = G.synthesis(w, camera_params, v, noise_mode='const')['image'][0]
    #         imgs.append(img)
    #
    #     video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    #
    # video_out.close()

    label_path = os.path.join(drive_root, 'dataset_realcam.json')

    mesh_path = os.path.join(os.path.dirname(drive_root), 'orthRender256x256_face_eye')
    dataset = ImagesDataset(drive_root, mesh_path=mesh_path, label_path=label_path, gtlabel_path=label_path,
                            source_transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]), return_vert=True,
                            skip=1, idx_range=('zyf_1811', 110, 610),
                            fvcoeffs_path=os.path.join(os.path.dirname(drive_root), 'coeffs_smooth'))

    print(mesh_path)
    faceverser = Faceverse_manager(device=device, base_coeff=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    count = 0
    bar = tqdm(enumerate(dataloader))
    for k, fetch_data in bar:
        image = fetch_data[0]
        label = fetch_data[1]
        vert = fetch_data[2]

        target_img = image.to(device).to(torch.float32)

        camera_params = conditioning_params.expand(label.shape[0], -1) if fixed_camera else label.to(device).to(torch.float32)
        if isinstance(vert, dict):
            for key in vert.keys(): vert[key] = vert[key].to(device).to(torch.float32)
        else:
            vert = vert.to(device).to(torch.float32)

        faceverser.id_coeff = faceverser.recon_model.split_coeffs(vert['coeff'][:1])[0]
        vert['uvcoords_image'] = faceverser.make_driven_rendering(vert['coeff'], res=256)

        imgs = [target_img[0]]
        for idx, seed in enumerate(seeds):
            w = ws[idx]
            img = G.synthesis(w, camera_params, vert, noise_mode='const', evaluation=True)['image'][0]
            imgs.append(img)

        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
        count += 1
    video_out.close()

if __name__ == "__main__":
    run_video_animation()