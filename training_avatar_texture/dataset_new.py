# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
from unittest import skip
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = True,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        load_obj    = True,
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        self.load_obj = load_obj

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_verts_ply(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_geo(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx], resolution=self.resolution)
        assert isinstance(image, np.ndarray)
        # assert list(image.shape) == self.image_shape
        # assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        label_cam = self.get_label(idx)
        mesh_cond = self.get_vert(self._raw_idx[idx])
        return image.copy(), label_cam, mesh_cond

    def load_random_data(self):
        gen_cond_sample_idx = [np.random.randint(self.__len__()) for _ in range(self.random_sample_num)]
        all_gen_c = np.stack([self.get_label(i) for i in gen_cond_sample_idx])
        all_gen_v = [self.get_vert(i) for i in gen_cond_sample_idx]
        all_gt_img = np.stack([self.get_image(i).astype(np.float32) / 127.5 - 1 for i in gen_cond_sample_idx])

        return all_gen_c, all_gen_v, all_gt_img

    def get_by_name(self, name):
        raw_idx = self._image_fnames.index(name)
        image = self._load_raw_image(raw_idx, resolution=self.resolution)
        mesh_cond = self.get_vert(raw_idx)
        label = self._get_raw_labels()[raw_idx]
        cam = self._raw_cams[raw_idx]
        label_cam = np.concatenate([label, cam], axis=-1)
        return image.copy(), label_cam, mesh_cond

    def get_label(self, idx):
        raise NotImplementedError

    def get_vert(self, idx):
        raise NotImplementedError


    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    # @property
    # def gen_label_dim(self):
    #         return 25 # 25 for camera params only

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        mesh_path       = None, # Path to mesh.
        mesh_type       = '.obj',
        resolution      = None, # Ensure specific resolution, None = highest available.
        load_exp        = False,
        load_lms_counter= False,
        load_bg         = False,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._mesh_path = mesh_path
        self.mesh_type = mesh_type
        self._zipfile = None
        self.load_exp = load_exp
        self.load_lms_counter = load_lms_counter
        self.load_bg = load_bg
        self.label_file = 'dataset.json'#'dataset_fv_simple.json'
        self._type = 'dir'
        self._condImg_path = path.replace('images512x512', 'lms_counter512x512_newF') if self.load_lms_counter else None
        self._bg_path = path.replace('images512x512', 'fgmasks512x512') if self.load_bg else None

        PIL.Image.init()
        self._image_fnames = list(dict(json.loads(open(os.path.join(self._path, 'dataset_realcam.json')).read())['labels']).keys())

        self._uv_fnames = [fname.split('.')[0]+'.npy' for fname in self._image_fnames]
        if len(self._image_fnames) == 0 or len(self._uv_fnames) == 0:
            raise IOError('No image files found in the specified path')
        self._raw_cams = self._load_raw_label(os.path.join(self._path, 'dataset_realcam.json'), 'labels')
        self._wcode_path = path.replace('images512x512', 'inversion_wplus')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + [3,] + list(self._load_raw_image(0, resolution).shape[-2:])
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname, path=None):
        if not path:
            path = self._path
        if self._type == 'dir':
            return open(os.path.join(path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_mouth_masks(self):
        json_path = os.path.join(self._mesh_path, 'mouth_masks.json')
        with self._open_file(json_path) as f:
            labels = json.load(f)
        return labels

    def _load_raw_label(self, json_path, sub_key=None):
        with open(json_path, 'rb') as f:
            labels = json.load(f)
        if sub_key is not None: labels = labels[sub_key]
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        return np.array(labels).astype(np.float32)

    def _load_raw_image_core(self, fname, path=None, resolution=None):
        with self._open_file(fname, path) as f:
            image = PIL.Image.open(f)
            if resolution:
                image = image.resize((resolution, resolution))
            image = np.array(image)#.astype(np.float32)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_image(self, raw_idx, resolution=None):
        fname = self._image_fnames[raw_idx]
        image = self._load_raw_image_core(fname, resolution=resolution) # [C, H, W]
        if self._condImg_path is not None:
            cond_image = self._load_raw_image_core(fname, path=self._condImg_path, resolution=resolution)
            image = np.concatenate([image, cond_image], axis=0)
        if self._bg_path is not None:
            cond_image = self._load_raw_image_core(fname, path=self._bg_path, resolution=resolution)[:1]    # TODO:最好把数据处理一遍确保是单通道的0-255
            cond_image[cond_image>127] = 255
            cond_image[cond_image<128] = 127.5
            image = np.concatenate([image, cond_image], axis=0)
        return image

    def _load_raw_labels(self):
        labels = self._load_raw_label(os.path.join(self._path, self.label_file), 'labels')
        return labels

    def get_vert(self, raw_idx):
        fname = self._uv_fnames[raw_idx]
        uvcoords_image = np.load(os.path.join(self._mesh_path, fname))[..., :3] # 前两维date range(-1, 1)，第三维是face_mask，最后一维是render_mask
        uvcoords_image[..., -1][uvcoords_image[..., -1] < 0.5] = 0
        uvcoords_image[..., -1][uvcoords_image[..., -1] >= 0.5] = 1
        # # uvcoords_image = uvcoords_image.transpose(2, 0, 1)  # HW3 => 3HW
        m_mask = np.asarray([0, 0, 1, 1], dtype=np.int32)  ####tmp
        return {'uvcoords_image': uvcoords_image.copy(), 'mouths_mask': m_mask}

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        cam = self._raw_cams[self._raw_idx[idx]]
        return np.concatenate([label, cam], axis=-1)

    def get_image(self, idx, resolution=None):
        # if resolution is None: resolution = self.resolution
        image = self._load_raw_image(self._raw_idx[idx], resolution=resolution)
        return image

    def get_CondImg(self, idx, resolution=None):
        assert self._condImg_path is not None
        # if resolution is None: resolution = self.resolution
        fname = self._image_fnames[self._raw_idx[idx]]
        cond_image = self._load_raw_image_core(fname, path=self._condImg_path, resolution=resolution)
        return cond_image

    def get_Wcode(self, idx):
        fname = self._image_fnames[self._raw_idx[idx]]
        w_code = np.load(os.path.join(self._wcode_path, fname.replace('png', 'npy')))
        return w_code