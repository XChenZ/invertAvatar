from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np
import torch
# from inversion.configs import paths_config
import copy
import cv2
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                fname = fname.split('.')[0]
                images.append((fname, path))
    return images


class ImagesDataset(Dataset):
    def __init__(self, source_root, label_path, mesh_path=None, mask_path=None, gtlabel_path=None, source_transform=None, resolution=(512, 512), skip=1, load_lms=False,
                 load_flame_expr=False, return_vert=False, return_expr=False, return_name=False, idx_range=None):
        # self.source_paths = sorted(make_dataset(source_root))[::skip]
        self.source_transform = source_transform
        self.resolution = resolution
        self.return_vert = return_vert
        self.return_expr = return_expr
        self.return_name = return_name
        self.load_lms = load_lms
        assert not (load_flame_expr and return_expr)
        with open(label_path, 'rb') as f:
            labels = json.load(f)['labels']
        self.mask_path = mask_path
        self.mesh_path = mesh_path
        # self._raw_mouth_masks = dict(json.loads(open(os.path.join(mesh_path, 'mouth_masks.json')).read()))
        self._raw_cams = dict(json.loads(open(gtlabel_path).read())['labels'])

        self.flame_expr_labels = dict(json.loads(open(
            os.path.join(os.path.dirname(label_path), 'dataset_exp_eye.json')).read())['labels']) if load_flame_expr else None
        # for label in self.gt_labels.keys():
        #     if not label == '0/01750.png':
        #         continue
        #     self.gt_labels = {label:self.gt_labels[label]}
        #     break

        labels = [label for label in labels if label[0] in self._raw_cams.keys()]
        labels.sort(key=lambda x: int(x[0].split('.')[0].split('/')[-1]))
        if idx_range is not None:
            min_, max_ = idx_range
            labels = [label for label in labels if int(label[0].split('/')[-1].split('.')[0]) in range(min_, max_ + 1)]
        self.labels = dict(labels)

        self.source_paths = [[inst[0].split('.')[0], os.path.join(source_root, inst[0])] for inst in labels]
        # self.source_paths.sort(key=lambda x:int(x[0].split('/')[-1]))
        self.source_paths = self.source_paths[::skip]
        self.return_realcam = True
        if return_expr:
            exp_path = os.path.join(os.path.dirname(label_path), 'dataset_BFMexp.json')
            with open(exp_path, 'rb') as f:
                exps = json.load(f)['labels']
            self.exps = dict(exps)
            self.expr_idx = np.asarray(
                [6, 7, 8, 9, 14, 15, 16, 17, 18, 40, 45, 46, 49, 50, 55, 56, 75, 80, 135, 147, 169, 170],
                dtype=np.int64)
            self.mouth_idx =np.asarray(
                [8, 9, 40, 45, 46, 49, 50, 55, 56, 73, 74, 80, 135, 146, 147], dtype=np.int64
                # [8, 9, 40, 45, 46, 49, 50, 55, 56, 75, 80, 135, 147, 169, 170], dtype=np.int64
            )

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path = self.source_paths[index]

        from_im = Image.open(from_path).convert('RGB')
        if from_im.size != self.resolution:
            from_im = from_im.resize(self.resolution)

        if self.source_transform:
            from_im = self.source_transform(from_im)

        if self.mask_path is not None:
            mask = np.asarray(cv2.imread(os.path.join(self.mask_path, fname+'.png'))).astype(np.float32) / 255  # [head, eye, mouth]
            from_im = torch.cat([from_im, torch.from_numpy(mask).permute(2, 0, 1)], dim=0)
            # mask = np.asarray(Image.open(os.path.join(self.mask_path, fname+'.png'))).astype(np.float32) / 255  # [head, eye, mouth]
            # from_im = torch.cat([from_im, torch.from_numpy(mask[None, :, :])], dim=0)

        out = [from_im, self.get_label(fname)[0]]
        if self.return_name:
            out.append(fname)
        if self.return_vert:
            out.append(self.get_v14_geo(fname))
        if self.return_expr:
            out.append(self.get_expr(fname)[0])
        if self.return_realcam:
            out.append(self.get_label(fname)[0])

        return out

    def _load_raw_label(self, json_path, sub_key=None):
        with open(json_path, 'rb') as f:
            labels = json.load(f)
        if sub_key is not None: labels = labels[sub_key]
        return dict(labels)
        # labels = dict(labels)
        # return np.array(labels).astype(np.float32)

    def get_by_name(self, fname):
        for f in self.source_paths:
            if f[0] == fname:
                from_path = f[1]
                break
        from_im = Image.open(from_path).convert('RGB')
        if from_im.size != self.resolution:
            from_im = from_im.resize(self.resolution)
        if self.source_transform:
            from_im = self.source_transform(from_im)

        if self.mask_path is not None:
            mask = np.asarray(cv2.imread(os.path.join(self.mask_path, fname + '.png'))).astype(np.float32) / 255  # [head, eye, mouth]
            from_im = torch.cat([from_im, torch.from_numpy(mask).permute(2, 0, 1)], dim=0)
            # mask = np.asarray(Image.open(os.path.join(self.mask_path, fname+'.png')).convert('L')).astype(np.float32) / 255
            # from_im = torch.cat([from_im, torch.from_numpy(mask[None, :, :, :])], dim=0)

        out = [from_im, self.get_label(fname)[0]]
        if self.return_vert:
            out.append(self.get_v14_geo(fname))
        if self.return_expr:
            out.append(self.get_expr(fname)[0])
        if self.return_realcam:
            out.append(self.get_label(fname)[0])
        return out

    def get_expr(self, idx):
        exps = np.asarray(self.exps[idx + '.png'])
        label = exps
        # label = np.concatenate((exps[:-4][ self.expr_idx], exps[-4:]), 0)  # fv3.1，最后四维是eye
        # label = exps[:-4][self.mouth_idx]
        label = label[None, :]
        label = label.astype(np.float32)
        return torch.tensor(label)

    def get_label(self, idx):
        label = np.asarray([self.labels[idx + '.png']], dtype=np.float32)
        if self.flame_expr_labels is not None:
            flame_expr_label = np.asarray([self.flame_expr_labels[idx + '.png']]).astype(np.float32)
            label = np.concatenate([label, flame_expr_label], axis=1)
        cam = np.asarray([self._raw_cams[idx + '.png']], dtype=np.float32)
        return torch.tensor(np.concatenate([label, cam], axis=1))

    def get_gtlabel(self, idx):
        return self.get_label(idx)
    # def get_vert_from_obj(self, obj_path):
    #     v = []
    #     with open(obj_path, "r") as f:
    #         while True:
    #             line = f.readline()
    #
    #             if line == "":
    #                 break
    #
    #             if line[:2] == "v ":
    #                 v.append([float(x) for x in line.split()[1:]])
    #
    #     v = np.array(v).reshape((-1, 3))
    #     return v

    # def get_vert(self, idx):
    #     v = self.get_vert_from_obj(os.path.join(paths_config.input_mesh_path, idx+'.obj'))
    #     if self.load_lms:
    #         lms = np.loadtxt(os.path.join(self.lms_path, idx+'.txt'))
    #         v = np.concatenate([v, lms], axis=0)
    #     return torch.from_numpy(v.astype(np.float32)).unsqueeze(0)
    #
    # def get_ortho_render(self, idx):
    #     render = np.load(os.path.join(paths_config.input_mesh_path, idx.split('/')[0]+'/'+idx.split('/')[1]+'.npy')) # [8, H, W]
    #     return torch.from_numpy(render.astype(np.float32))

    def get_v14_geo(self, idx):
        uvcoords_image = np.load(os.path.join(self.mesh_path, idx+'.npy'))[..., :3] # [HW3] 前两维date range(-1, 1)，第三维是face_mask，最后一维是render_mask
        uvcoords_image[..., -1][uvcoords_image[..., -1] < 0.5] = 0; uvcoords_image[..., -1][uvcoords_image[..., -1] >= 0.5] = 1
        # uvcoords_image = cv2.resize(cv2.resize(uvcoords_image, (512, 512)), (256, 256))
        # uvcoords_image = ((uvcoords_image+1)*127.5).astype(np.uint8)    #
        # uvcoords_image = cv2.medianBlur(uvcoords_image, ksize=3).astype(np.float32)/127.5 - 1.   #
        # import torchvision
        # torchvision.utils.save_image(torch.from_numpy(uvcoords_image).permute(2, 0, 1).unsqueeze(0), os.path.join('/root/autodl-tmp/Code/animatable_eg3d/out', idx[2:] + '_.png'), normalize=True, range=(-1, 1))

        # m_mask = np.asarray(self._raw_mouth_masks[idx+'.png'], dtype=np.int)
        m_mask = np.asarray([0, 0, 1, 1], dtype=np.int32) ####tmp
        # exps = np.asarray(self.exps[idx + '.png'])  ######
        return {'uvcoords_image': torch.tensor(uvcoords_image.copy()), 'mouths_mask': torch.tensor(m_mask)}