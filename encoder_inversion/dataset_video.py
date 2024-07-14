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
import copy
import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import cv2

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# exp_idx = [8,  9,  18,  19,  40,  45,  46,  47,  48,  49,  50,  51,  52, 53,  54,  55,  56,  61,  69,  70,  73,  74,  75,  76,  77,  78,
#            79,  80,  81,  82,  83,  84,  89,  90,  91,  92, 115, 116, 117, 118, 127, 128, 131, 132, 135, 146, 147, 149, 150, 151, 156, 157,
#            160, 169, 170]   # for fv_v3.1   # 这55/171的表情分量贡献了大约80%的expr_blendshape
# exp_idx = [  4,   5,   6,   7,   8,   9,  12,  13,  14,  15,  16,  17,  18,
#         19,  20,  21,  34,  35,  40,  45,  46,  47,  48,  49,  50,  51,
#         52,  53,  54,  55,  56,  61,  62,  63,  64,  67,  68,  69,  70,
#         71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
#         84,  89,  90,  91,  92,  93,  94,  95,  96, 115, 116, 117, 118,
#        119, 120, 121, 122, 127, 128, 129, 130, 131, 132, 133, 134, 135,
#        146, 147, 148, 149, 150, 151, 156, 157, 158, 159, 160, 169, 170] # 这91/171的表情分量贡献了大约92%的expr_blendshape
# exp_idx = [6, 7, 8, 9, 14, 15, 16, 17, 18, 40, 45, 46, 49, 50, 55, 56, 75, 80, 135, 147, 169, 170]  # 手动挑选的，粗糙表情
# mouth_idx = [8, 9, 40, 45, 46, 49, 50, 55, 56, 75, 80, 135, 147, 169, 170]

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
        # self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        # if (max_size is not None) and (self._raw_idx.size > max_size):
        #     np.random.RandomState(random_seed).shuffle(self._raw_idx)
        #     self._raw_idx = np.sort(self._raw_idx[:max_size])

        # # Apply xflip.
        # self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        # if xflip:
        #     self._raw_idx = np.tile(self._raw_idx, 2)
        #     self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

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
        raise NotImplementedError

    def __getitem__(self, vidx):
        video_frames, Ts, label_cam, mesh_cond, raw_idxs, m_mask, video_uv, base_inversion, fake_imgs, fnames = self.get_raw_video_frames(
            vidx, sample_num=self.frm_per_vid, resolution=self.resolution)  # [2, C, H, W], [2,], [2]
        # print(fnames)
        # if m_mask is None:
        #     video_frames = video_frames_
        # else:
        #     video_frames = []
        #     for i, m in enumerate(m_mask):
        #         video_frame = PIL.Image.fromarray(video_frames_[i, :, m[0]:m[1], m[2]:m[3]].transpose(1, 2, 0))
        #         video_frames.append(np.asarray(video_frame.resize((64, 64))).transpose(2, 0, 1))
        #     video_frames = np.stack(video_frames, axis=0)

        assert isinstance(video_frames, np.ndarray)
        # assert list(image.shape) == self.image_shape
        assert video_frames.dtype == np.uint8

        # fname = [self._image_fnames[raw_idx] for raw_idx in raw_idxs]  ############
        # label_cam = np.stack([self.get_label(raw_idx) for raw_idx in raw_idxs], axis=0)
        # mesh_cond = {'uvcoords_image': uvcoords}
        out = [video_frames, label_cam, mesh_cond, Ts, video_uv]
        if base_inversion is not None: out.append(base_inversion)
        if fake_imgs is not None: out.append(fake_imgs)
        if m_mask is not None: out.append(m_mask)
        return out

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
        d.raw_idx = int(idx)
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


class VideoFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        mesh_path       = None, # Path to mesh.
        mesh_type       = '.obj',
        resolution      = None, # Ensure specific resolution, None = highest available.
        load_exp        = False,
        load_lms_counter       = False,
        load_inversion         = True,
        load_uv         = False,
        load_bg         = False,
        frm_per_vid   = 1,
        label_file      = 'dataset.json',
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._mesh_path = mesh_path
        self.mesh_type = mesh_type
        self._zipfile = None
        self.load_exp = load_exp
        self.load_lms_counter = load_lms_counter
        self.load_bg = load_bg
        self.load_inversion = load_inversion
        # self.load_uv = load_uv
        self.frm_per_vid = frm_per_vid
        self.label_file = 'dataset_threshXY020_33fpv_skip3_clean.json'
        self._type = 'dir'
        self._condImg_path = path.replace('images512x512', 'lms_counter512x512_newF') if self.load_lms_counter else None
        self._bg_path = path.replace('images512x512', 'headmasks512x512') if self.load_bg else None
        self._uv_path = path.replace('images512x512', 'uvRender256x256') #if self.load_uv else None

        self.uvmask = cv2.imread('/data/zhaoxiaochen/Dataset/dense_uv_expanded_mask_onlyFace.png', 0).astype(np.float32)/255

        PIL.Image.init()
        self._image_fnames = list(dict(json.loads(open(os.path.join(self._path, self.label_file)).read())['labels']).keys())
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        self._uv_fnames, self._video_infos = [], {}
        for raw_idx, fname in enumerate(self._image_fnames):
            self._uv_fnames.append(fname.split('.')[0]+'.npy')
            vid_name = fname.split('/')[0]
            if vid_name in self._video_infos.keys(): self._video_infos[vid_name].append([fname.split('/')[-1], raw_idx])
            else: self._video_infos[vid_name] = [[fname.split('/')[-1], raw_idx]]
        self._video_infos = [sorted(self._video_infos[vid_name], key=lambda x:int(x[0].split('.')[0])) for vid_name in self._video_infos.keys()]

        self._raw_img_mouth_masks = None#self._load_raw_label(os.path.join(self._path, 'img_mouth_masks.json'))
        self._raw_cams = self._load_raw_label(os.path.join(self._path, 'dataset_realcam.json'), 'labels')
        self._inversion_path = path.replace('images512x512', 'inversion')
        # if self.load_exp:
        #     self.exp_file = 'dataset_fv_exp_onlyMouth.json'
        #     self._raw_exps = self._load_raw_label(os.path.join(self._path, self.exp_file), 'labels')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + [3, resolution, resolution]
        # if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #     raise IOError('Image files do not match the specified resolution')
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

    def __len__(self):
        return len(self._video_infos)

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
            image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_uv(self, raw_idx, resolution=None):
        fname = self._image_fnames[raw_idx]
        uvppverts_image = np.load(os.path.join(self._uv_path, fname.replace('.png', '.npy'))).astype(np.float32)
        uvppverts_image[..., -1] *= self.uvmask
        uvgttex_image = np.array(PIL.Image.open(open(os.path.join(self._uv_path, fname.split('.')[0] + '_uvgttex.png'), 'rb'))).astype(np.float32) / 127.5 - 1
        uv_image = np.concatenate([uvgttex_image, uvppverts_image], axis=-1).transpose(2, 0, 1)  # HWC => CHW
        return uv_image

    def _load_raw_image(self, raw_idx, resolution=None):
        fname = self._image_fnames[raw_idx]
        image = self._load_raw_image_core(fname, resolution=resolution) # [C, H, W]
        if self._condImg_path is not None:
            cond_image = self._load_raw_image_core(fname, path=self._condImg_path, resolution=resolution)
            image = np.concatenate([image, cond_image], axis=0)
        if self.load_bg:
            cond_image = self._load_raw_image_core(fname, path=self._bg_path, resolution=resolution)[:1]    # 最后一个通道是head masks，第一个通道是fg masks
            cond_image[cond_image>127] = 255
            cond_image[cond_image<128] = 127.5  # 放缩到[-1, 1]后对应于0
            mask_image = cond_image
            image = np.concatenate([image, mask_image], axis=0)
        return image

    def get_raw_video_frames(self, vidx, sample_num, base_fidx=None, resolution=None):

        v_info = self._video_infos[vidx]
        num_frames = len(v_info)

        if sample_num == 1:
            frames = [np.random.randint(num_frames)]
        else:
            frames = np.random.uniform(low=0., high=num_frames, size=sample_num).astype(np.int32).tolist()
            # frames.sort()

        # frames = [[fidx for fidx in range(num_frames) if v_info[fidx][0] == '00000024.png'][0],
        #             [fidx for fidx in range(num_frames) if v_info[fidx][0] == '00000006.png'][0],
        #             [fidx for fidx in range(num_frames) if v_info[fidx][0] == '00000000.png'][0],
        #             [fidx for fidx in range(num_frames) if v_info[fidx][0] == '00000090.png'][0]]
        raw_idxs = [v_info[fidx][1] for fidx in frames]
        Ts = np.asarray([fidx / (num_frames - 1) for fidx in frames], dtype=np.float32)

        label_cam = np.stack([self.get_label(raw_idx) for raw_idx in raw_idxs], axis=0)
        uvcoords = np.stack([self.get_vert(raw_idx) for raw_idx in raw_idxs], axis=0)
        # uv_image = np.stack([self._load_raw_uv(raw_idx) for raw_idx in raw_idxs], axis=0) if self.load_uv else None
        m_mask = np.stack([self._raw_img_mouth_masks[raw_idx].astype(np.int32) for raw_idx in raw_idxs], axis=0) if self._raw_img_mouth_masks is not None else None
        video_frames = np.stack([self._load_raw_image(raw_idx, resolution=resolution) for raw_idx in raw_idxs], axis=0)  # [C, H, W]
        if base_fidx is None:
            # base_fidx = v_info[np.random.randint(num_frames)][1]  # 随机找一帧作为e4e的base
            base_fidx = v_info[frames[0]][1]    # 找第一张图片作为e4e的base

        if self.load_inversion:
            w_inversion = None #self.get_inversion(base_fidx, return_texture=False)

            if False:   # dataloader时就完成uv_delta的计算
                uv_input, fnames = [], []
                for i, raw_idx in enumerate(raw_idxs):
                    uv_gttex, uv_pverts = torch.from_numpy(self._load_raw_uv(raw_idx)).unsqueeze(0).split(3, dim=1) # [1, 3, H, W]
                    fname = self._image_fnames[raw_idx].split('.')[0] + '_fakeImg/' + self._image_fnames[base_fidx].split('/')[-1]
                    fnames.append(fname)
                    fake_img = self._load_raw_image_core(fname, path=self._inversion_path, resolution=resolution)
                    delta_x = torch.from_numpy(
                        (fake_img.astype(np.float32) / 127.5 - 1) - (video_frames[i, :3].astype(np.float32) / 127.5 - 1)).unsqueeze(0)  # 注意此时值的范围是[-1, 1]
                    uv_delta = torch.nn.functional.grid_sample(delta_x, uv_pverts.permute(0, 2, 3, 1)[..., :2], mode='bilinear', align_corners=False)
                    uv_delta = -1 * torch.ones_like(uv_delta, dtype=torch.float32) * (1 - uv_pverts[:, -1:]) + uv_delta * uv_pverts[:, -1:]
                    uv_input.append(torch.cat([uv_gttex, uv_delta, uv_pverts[:, -1:]], dim=1)[0]) # [C, H, W]
                uv_input = torch.stack(uv_input, dim=0)
                return video_frames, Ts, label_cam, uvcoords.astype(np.float16), raw_idxs, m_mask, uv_input.to( torch.float16), w_inversion, fake_imgs
            elif False:
                fnames = [self._image_fnames[raw_idx].split('.')[0] + '_fakeImg/' + self._image_fnames[base_fidx].split('/')[-1] for raw_idx in raw_idxs]
                fake_imgs = np.stack([self._load_raw_image_core(fname, path=self._inversion_path, resolution=resolution) for fname in fnames], axis=0)
                # fake_imgs = [None for _ in range(sample_num)]
                uv = torch.from_numpy(np.stack([self._load_raw_uv(raw_idx) for raw_idx in raw_idxs], axis=0))  # [T, C, H, W]
                return video_frames, Ts, label_cam, uvcoords.astype(np.float16), raw_idxs, m_mask, uv.to(torch.float16), w_inversion, fake_imgs
            else:   # 不load fake_img
                uv = torch.from_numpy(np.stack([self._load_raw_uv(raw_idx) for raw_idx in raw_idxs], axis=0))  # [T, C, H, W]
                return video_frames, Ts, label_cam, uvcoords.astype(np.float16), raw_idxs, m_mask, uv.to(torch.float16), w_inversion, None, \
                    [self._image_fnames[raw_idx].split('.')[0] for raw_idx in raw_idxs]
        else:
            uv = np.stack([self._load_raw_uv(raw_idx) for raw_idx in raw_idxs], axis=0)  # [T, C, H, W]
            video_frames = {'image': video_frames, 'uv': uv}
            return video_frames, Ts, label_cam, uvcoords.astype(np.float16), raw_idxs, m_mask

    def _load_raw_labels(self):
        labels = self._load_raw_label(os.path.join(self._path, self.label_file), 'labels')
        # if self.load_exp:
        #     exps = self._load_raw_label(os.path.join(self._path, self.exp_file), 'labels')
        #     labels = np.concatenate((labels, exps), 1)
            # labels = np.concatenate((labels, exps[:, :-4][:, exp_idx], exps[:, -4:]), 1)  # fv3.1，最后四维是eye
        return labels

    def get_vert(self, raw_idx):
        fname = self._uv_fnames[raw_idx]
        uvcoords_image = np.load(os.path.join(self._mesh_path, fname))[..., :3] # 前两维date range(-1, 1)，第三维是face_mask，最后一维是render_mask
        uvcoords_image[..., -1][uvcoords_image[..., -1] < 0.5] = 0
        uvcoords_image[..., -1][uvcoords_image[..., -1] >= 0.5] = 1
        return uvcoords_image
        # m_mask = self._raw_mouth_masks[raw_idx].astype(np.int)
        # return {'uvcoords_image': uvcoords_image.copy(), 'mouths_mask': m_mask}

    def get_label(self, raw_idx):
        label = self._get_raw_labels()[raw_idx]
        cam = self._raw_cams[raw_idx]
        return np.concatenate([label, cam], axis=-1)

    def get_image(self, raw_idx, resolution=None):
        if resolution is None: resolution = self.resolution
        image = self._load_raw_image(raw_idx, resolution=resolution)
        return image

    def get_CondImg(self, raw_idx, resolution=None):
        assert self._condImg_path is not None
        if resolution is None: resolution = self.resolution
        fname = self._image_fnames[raw_idx]
        cond_image = self._load_raw_image_core(fname, path=self._condImg_path, resolution=resolution)
        return cond_image

    def get_inversion(self, raw_idx, return_texture=False):
        fname = self._image_fnames[raw_idx]
        w_code = np.load(os.path.join(self._inversion_path, fname.replace('.png', '_code.npy')))
        if not return_texture:
            return w_code
        else:
            texture = np.load(os.path.join(self._inversion_path, fname.replace('.png', '_textureFeat.npy')))
            return {'ws': w_code, 'texture': texture}