from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np
import torch
from inversion.configs import paths_config
import copy
import time
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

    def __init__(self, source_root, mesh_path, label_path, mask_path=None, gtlabel_path=None, source_transform=None, resolution=(512, 512), skip=1, load_lms=False,
                 load_flame_expr=False, return_vert=False, return_expr=False, return_name=False, idx_range=None):
        # self.source_paths = sorted(make_dataset(source_root))[::skip]
        self.source_transform = source_transform
        self.resolution = resolution
        self.return_vert = return_vert
        self.return_expr = return_expr
        self.return_name = return_name
        self.load_lms = load_lms
        self.mesh_path = mesh_path
        assert not (load_flame_expr and return_expr)
        with open(label_path, 'rb') as f:
            labels = json.load(f)['labels']
        self.mask_path = mask_path
        self.gt_labels = dict(json.loads(open(gtlabel_path).read())['labels']) if gtlabel_path is not None else None
        self.flame_expr_labels = dict(json.loads(open(
            os.path.join(os.path.dirname(label_path), 'dataset_exp_eye.json')).read())['labels']) if load_flame_expr else None
        # for label in self.gt_labels.keys():
        #     if not label == '0/01750.png':
        #         continue
        #     self.gt_labels = {label:self.gt_labels[label]}
        #     break
        labels = [label for label in labels if label[0] in self.gt_labels.keys()]
        if idx_range is not None:
            min_, max_ = idx_range
            labels = [label for label in labels if int(label[0].split('/')[-1].split('.')[0]) in range(min_, max_ + 1)]
        self.labels = dict(labels)
        self.return_realcam = gtlabel_path is not None

        self.source_paths = [[inst[0].split('.')[0], os.path.join(source_root, inst[0])] for inst in labels]
        self.source_paths.sort(key=lambda x:int(x[0].split('/')[-1]))
        self.source_paths = self.source_paths[::skip]
        self.lms_path = self.mesh_path.replace('meshes', 'lms')
        self.range = range
        self.all_verts = None
        # self.preload_meshes(os.path.join(paths_config.input_mesh_path, 'dataset_meshes.npy'))
        if return_expr:
            exp_path = label_path.split('.')[0] + '_BFMexp_res.json'
            with open(exp_path, 'rb') as f:
                exps = json.load(f)['labels']
            self.exps = dict(exps)
            self.expr_idx = np.asarray(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 30, 31, 32, 34],
                dtype=np.int64)

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
            mask = np.asarray(Image.open(os.path.join(self.mask_path, fname+'.png')).convert('L')).astype(np.float32) / 255
            from_im = torch.cat([from_im, torch.from_numpy(mask[None, :, :])], dim=0)

        # if not self.return_all:
        #     return fname, from_im
        # else:
        out = [from_im, self.get_label(fname)[0]]
        if self.return_name:
            out.append(fname)
        if self.return_vert:
            if self.all_verts is not None:
                out.append(self.all_verts[fname+'.obj'])
            elif self.load_lms:
                out.append(self.get_vert(fname)[0])
            else:
                out.append(self.get_ortho_render(fname))
        if self.return_expr:
            out.append(self.get_expr(fname)[0])
        if self.return_realcam:
            out.append(self.get_gtlabel(fname)[0])

        return out

    def preload_meshes(self, path):
        # self.all_verts = dict(json.loads(open(path).read()))
        self.all_verts = np.load(path, allow_pickle=True).item()
        for key in self.all_verts.keys():
            self.all_verts[key] = torch.tensor(self.all_verts[key]).float()
        # self.all_verts = torch.load(path)


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
            mask = np.asarray(Image.open(os.path.join(self.mask_path, fname+'.png')).convert('L')).astype(np.float32) / 255
            from_im = torch.cat([from_im, torch.from_numpy(mask[None, :, :])], dim=0)

        out = [from_im, self.get_label(fname)[0]]
        if self.return_vert:
            if self.load_lms:
                out.append(self.get_vert(fname)[0])
            else:
                out.append(self.get_ortho_render(fname))
        if self.return_expr:
            out.append(self.get_expr(fname)[0])
        if self.return_realcam:
            out.append(self.get_gtlabel(fname)[0])
        return out

    def get_expr(self, idx):
        label = np.asarray(self.exps[idx + '.png'])
        if self.expr_idx is not None:
            label = label[self.expr_idx]
        label = label[None, :]
        label = label.astype({1: np.int64, 2: np.float32}[label.ndim])
        return torch.tensor(label)

    def get_label(self, idx):
        label = [self.labels[idx + '.png']]
        label = np.array(label)
        label = label.astype({1: np.int64, 2: np.float32}[label.ndim])

        if self.flame_expr_labels is not None:
            flame_expr_label = np.array([self.flame_expr_labels[idx + '.png']]).astype(np.float32)
            label = np.concatenate([label, flame_expr_label], axis=1)

        return torch.tensor(label)

    def get_gtlabel(self, idx):
        label = [self.gt_labels[idx + '.png']]
        label = np.array(label)
        label = label.astype({1: np.int64, 2: np.float32}[label.ndim])

        return torch.tensor(label)

    def get_vert_from_obj(self, obj_path):
        v = []
        with open(obj_path, "r") as f:
            while True:
                line = f.readline()

                if line == "":
                    break

                if line[:2] == "v ":
                    v.append([float(x) for x in line.split()[1:]])

        v = np.array(v).reshape((-1, 3))
        return v

    def get_vert(self, idx):
        v = self.get_vert_from_obj(os.path.join(self.mesh_path, idx+'.obj'))
        if self.load_lms:
            # lms = []
            # print(os.path.join(self.lms_path, idx+'.txt'))
            # exit(0)
            # with open(os.path.join(self.lms_path, idx+'.txt'), "r") as f:
            #     while True:
            #         line = f.readline()
            #         print(line)
            #         lms.append([float(x) for x in line.split()])
            # lms = np.array(lms).reshape((-1, 3))
            lms = np.loadtxt(os.path.join(self.lms_path, idx+'.txt'))
            v = np.concatenate([v, lms], axis=0)
        return torch.from_numpy(v.astype(np.float32)).unsqueeze(0)

    def get_ortho_render(self, idx):
        render = np.load(os.path.join(self.mesh_path, idx.split('/')[0]+'/'+idx.split('/')[1]+'.npy')) # [8, H, W]
        return torch.from_numpy(render.astype(np.float32))