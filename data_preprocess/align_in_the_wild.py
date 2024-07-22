# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Includes modifications proposed by Jeremy Fix
# from here: https://github.com/NVlabs/ffhq-dataset/pull/3

"""Download Flickr-Faces-HQ (FFHQ) dataset to current working directory."""
import copy
import os
from PIL import ImageDraw
import PIL.Image
import PIL.ImageFile
import numpy as np
import scipy.ndimage
# import threading
# import queue
import time
import json
# import uuid
# import glob
import argparse
from preprocess import align_img
# import shutil
# from collections import OrderedDict, defaultdict
import cv2
from tqdm import tqdm
import multiprocessing
from scipy.ndimage import gaussian_filter1d
import sys
sys.path.append('Deep3DFaceRecon_pytorch')
from Deep3DFaceRecon_pytorch.util.load_mats import load_lm3d


PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True  # avoid "Decompressed Data Too Large" error

# ----------------------------------------------------------------------------

json_spec = dict(file_url='https://drive.google.com/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA', file_path='ffhq-dataset-v2.json', file_size=267793842,
                 file_md5='425ae20f06a4da1d4dc0f46d40ba5fd6')


# ----------------------------------------------------------------------------

def get_transform_coord(ori_size, quad, transform_size, lm5p, circle_rad=3):
    # color_bank = np.arange(255, 0, -256//lm5p.shape[0])
    color_bank = cv2.applyColorMap(np.arange(255, 0, -256//lm5p.shape[0]).astype(np.uint8), colormap=cv2.COLORMAP_JET).squeeze()
    img_ = PIL.Image.new("RGB", ori_size, color=0)
    draw = ImageDraw.Draw(img_)
    for i, lm in enumerate(lm5p):
        draw.chord((lm[0]-circle_rad, lm[1]-circle_rad, lm[0]+circle_rad, lm[1]+circle_rad), 0, 360, tuple(color_bank[i]), tuple(color_bank[i]))
    # img_.save('/media/zxc/hdd2/Dataset/VFHQ/eg3d_dataset/realign/Clip+BW2rPGtQCNo+P0+C1+F1356-1716/get_transform_coord.png')
    img_ = img_.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    # img_.save('/media/zxc/hdd2/Dataset/VFHQ/eg3d_dataset/realign/Clip+BW2rPGtQCNo+P0+C1+F1356-1716/get_transform_coord_trans.png')
    out_lm5p = lm5p.copy()
    for i in range(lm5p.shape[0]):
        gray = cv2.cvtColor((np.abs(np.float32(img_) - color_bank[i]).astype(np.uint8)), cv2.COLOR_BGR2GRAY)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
        out_lm5p[i] = np.asarray(minLoc, dtype=np.float32)
    return out_lm5p

def transform_image(img, lm5p, transform_size, output_size, enable_padding):
    assert lm5p.shape[0] == 5
    lm5p_ = lm5p.copy()
    eye_left, eye_right, nose, mouth_left, mouth_right = lm5p[0], lm5p[1], lm5p[2], lm5p[3], lm5p[4]
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg


    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]

    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    q_scale = 1.2
    x = q_scale * x
    y = q_scale * y
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])  # [upper left, lower left, lower right, upper right]
    qsize = np.hypot(*x) * 2

    # Shrink.
    start_time = time.time()
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        lm5p_ = lm5p_ * (1. / rsize)
        if False:   # 未验证
            img_ = copy.deepcopy(img)
            draw = ImageDraw.Draw(img_)
            draw.point(
                [(lm5p_[0, 0], lm5p_[0, 1]), (lm5p_[1, 0], lm5p_[1, 1]), (lm5p_[2, 0], lm5p_[2, 1]), (lm5p_[3, 0], lm5p_[3, 1]), (lm5p_[4, 0], lm5p_[4, 1])],
                (0, 255, 0))
            img_.save('/media/zxc/hdd2/Dataset/VFHQ/eg3d_dataset/realign/Clip+BW2rPGtQCNo+P0+C1+F1356-1716/shrink.png')  ##################
        quad /= shrink
        qsize /= shrink
    # print("shrink--- %s seconds ---" % (time.time() - start_time))

    # Crop.
    start_time = time.time()
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        lm5p_[:, 0] = lm5p_[:, 0] - crop[0]; lm5p_[:, 1] = lm5p_[:, 1] - crop[1]
        if False:   # 未验证
            img_ = copy.deepcopy(img)
            img_ = PIL.Image.new(img.mode, img.size, color=0)
            draw = ImageDraw.Draw(img_)
            draw.point(
                [(lm5p_[0, 0], lm5p_[0, 1]), (lm5p_[1, 0], lm5p_[1, 1]), (lm5p_[2, 0], lm5p_[2, 1]), (lm5p_[3, 0], lm5p_[3, 1]), (lm5p_[4, 0], lm5p_[4, 1])],
                (0, 255, 0))
            img_.save('/media/zxc/hdd2/Dataset/VFHQ/eg3d_dataset/realign/Clip+BW2rPGtQCNo+P0+C1+F1356-1716/crop.png')  ##################
        quad -= crop[0:2]
    # print("crop--- %s seconds ---" % (time.time() - start_time))

    # Pad.
    start_time = time.time()
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding is not None and max(pad) > border - 4:    # pad with blur
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        if enable_padding == 'zero_padding':
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'constant')
            lm5p_[:, 0] = lm5p_[:, 0] + pad[0]; lm5p_[:, 1] = lm5p_[:, 1] + pad[1]
        else:
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            lm5p_[:, 0] = lm5p_[:, 0] + pad[0]; lm5p_[:, 1] = lm5p_[:, 1] + pad[1]
            if enable_padding == 'blur_padding':
                h, w, _ = img.shape
                y, x, _ = np.ogrid[:h, :w, :1]
                mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                                  1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
                low_res = cv2.resize(img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
                blur = qsize * 0.02 * 0.1
                low_res = scipy.ndimage.gaussian_filter(low_res, [blur, blur, 0])
                low_res = cv2.resize(low_res, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                img += (low_res - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
                median = cv2.resize(img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
                median = np.median(median, axis=(0, 1))
                img += (median - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        # if False:
        #     img_ = copy.deepcopy(img)
        #     draw = ImageDraw.Draw(img_)
        #     for lm in lm5p_:
        #         draw.chord((lm[0]-2, lm[1]-2, lm[0]+2, lm[1]+2), 0, 360, (255, 0, 0), (255, 0, 0))
        #     img_.save('/media/zxc/hdd2/Dataset/VFHQ/eg3d_dataset/realign/Clip+BW2rPGtQCNo+P0+C1+F1356-1716/pad.png')  ##################

        quad += pad[:2]
    # print("pad--- %s seconds ---" % (time.time() - start_time))

    # Transform.
    start_time = time.time()
    transformed_lm5p = get_transform_coord(img.size, (quad + 0.5).flatten(), transform_size, lm5p_.copy())
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    ##  四点透视变换： Maps a quadrilateral (a region defined by four corners) from the image to a rectangle of the given size.
    ##  quda=[upper left, lower left, lower right, upper right corner]

    if False:
        img_ = copy.deepcopy(img)
        draw = ImageDraw.Draw(img_)
        for lm in transformed_lm5p:
            draw.chord((lm[0] - 2, lm[1] - 2, lm[0] + 2, lm[1] + 2), 0, 360, (255, 0, 0), (255, 0, 0))
        img_.save('/media/zxc/hdd2/Dataset/VFHQ/eg3d_dataset/realign/Clip+BW2rPGtQCNo+P0+C1+F1356-1716/transform.png')  ##################

    if output_size < transform_size:
        assert NotImplementedError
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # print("transform--- %s seconds ---" % (time.time() - start_time))

    return img, (quad + 0.5).flatten(), transformed_lm5p

def save_detection_as_txt(dst, lm5p):
    outLand = open(dst, "w")
    for i in range(lm5p.shape[0]):
        outLand.write(str(float(lm5p[i][0])) + " " + str(float(lm5p[i][1])) + "\n")
    outLand.close()

def process_image(kwargs):  # item_idx, item, dst_dir="realign1500", output_size=1500, transform_size=4096, enable_padding=True):

    src_dir = kwargs['src_dir']
    dst_dir = kwargs['dst_dir']
    # dst_param_dir = kwargs['dst_param_dir']
    lm5p = kwargs['lm5p']
    im_name = kwargs['im_name']
    output_size = kwargs['output_size']
    transform_size = kwargs['transform_size']
    enable_padding = kwargs['enable_padding']
    enable_warping = kwargs['enable_warping']
    save_realign_dir = kwargs['save_realign_dir']
    save_detection_dir = kwargs['save_detection_dir']
    os.makedirs(dst_dir, exist_ok=True)

    src_file = os.path.join(src_dir, im_name)
    assert os.path.isfile(src_file), src_file
    img = PIL.Image.open(src_file)
    _, H = img.size

    params = {'name': src_file, 'lm': lm5p.tolist()}
    if enable_warping:
        img, quad, aligned_lm5p = transform_image(img, lm5p.copy(), transform_size, output_size, enable_padding)
        params['quda'] = quad.tolist()
        aligned_lm5p[:, -1] = output_size - 1 - aligned_lm5p[:, -1]
    else:
        aligned_lm5p = lm5p.copy()
        aligned_lm5p[:, -1] = H - 1 - aligned_lm5p[:, -1]

    # Save aligned image.
    im_name = im_name.split('.')[0] + '.png'
    dst_file = os.path.join(dst_dir, im_name)
    if save_realign_dir is not None: img.save(os.path.join(save_realign_dir, im_name))
    if save_detection_dir is not None: save_detection_as_txt(os.path.join(save_detection_dir, im_name.replace('png', 'txt')), aligned_lm5p)

    img_cropped, crop_param = crop_image(img, aligned_lm5p.copy(), output_size=512)
    params['crop'] = crop_param
    img_cropped.save(dst_file)


def crop_image(im, lm, center_crop_size=700, rescale_factor=300, target_size=1024., output_size=512):
    _, H = im.size
    lm3d_std = load_lm3d("Deep3DFaceRecon_pytorch/BFM/")
    _, im_high, _, _, crop_left, crop_up, s = align_img(im, lm, lm3d_std, target_size=target_size, rescale_factor=rescale_factor)

    left = int(im_high.size[0] / 2 - center_crop_size / 2)
    upper = int(im_high.size[1] / 2 - center_crop_size / 2)
    right = left + center_crop_size
    lower = upper + center_crop_size
    im_cropped = im_high.crop((left, upper, right, lower))
    im_cropped = im_cropped.resize((output_size, output_size), resample=PIL.Image.LANCZOS)
    # out_path = os.path.join(out_dir, img_file.split(".")[0] + ".png")
    # im_cropped.save(out_path)
    crop_param = [int(left), int(upper), int(center_crop_size), int(crop_left), int(crop_up), float(H * s), int(target_size)]
    return im_cropped, crop_param


def process_video(kwargs):  # item_idx, item, dst_dir="realign1500", output_size=1500, transform_size=4096, enable_padding=True):
    video_dir = kwargs['src_dir']
    dst_dir = kwargs['dst_dir']
    lm5p_dict = kwargs['lm5p']
    im_names = kwargs['im_names']
    output_size = kwargs['output_size']
    transform_size = kwargs['transform_size']
    enable_padding = kwargs['enable_padding']
    enable_warping = kwargs['enable_warping']
    save_realign_dir = kwargs['save_realign_dir']
    save_detection_dir = kwargs['save_detection_dir']
    apply_GF = kwargs['apply_GF']

    im_names = list(lm5p_dict.keys())   # 因为可能存在有的帧没有检测出lm5p
    if apply_GF > 0: im_names.sort(key=lambda x:int(x.split('.')[0]))
    kps_sequence = [lm5p_dict[key] for key in im_names]
    kps_sequence = np.asarray(kps_sequence, dtype=np.float32)
    if apply_GF > 0: kps_sequence = gaussian_filter1d(kps_sequence, sigma=apply_GF, axis=0)

    assert len(im_names)==kps_sequence.shape[0]
    if save_realign_dir is not None: os.makedirs(save_realign_dir, exist_ok=True)
    if save_detection_dir is not None: os.makedirs(save_detection_dir, exist_ok=True)
    for idx, im_name in enumerate(im_names):

        lm5p = kps_sequence[idx].reshape([-1, 2])
        input = {'src_dir': video_dir, 'dst_dir': dst_dir, 'im_name': im_name, 'lm5p': lm5p, 'save_realign_dir': save_realign_dir,
                 'save_detection_dir': save_detection_dir, 'output_size': output_size, 'transform_size': transform_size,
                 'enable_padding': enable_padding, 'enable_warping': enable_warping}
        process_image(input)
    open(os.path.join(dst_dir, 'finish'), "w")



def recreate_aligned_images_fast(root_dir, lms_root_dir, dst_dir, save_realign_dir, valid_video_json, output_size=1024, transform_size=4096, enable_padding=True,
                                 n_threads=12):
    print('Recreating aligned images...')
    valid_idx = json.loads(open(valid_video_json).read())
    inputs = []
    for video_name, img_names in valid_idx:
        video_dir = os.path.join(root_dir, video_name)
        dst_save_dir = os.path.join(dst_dir, video_name)

        if not os.path.isdir(video_dir): continue
        lm5p_dict = json.loads(open(os.path.join(lms_root_dir, video_name+'.json')).read())
        for im_name in img_names:
            lm5p = np.asarray(lm5p_dict[im_name]).astype(np.float32)
            lm5p = lm5p.reshape([-1, 2])
            input = {'src_dir': video_dir, 'dst_dir': dst_save_dir, 'im_name': im_name, 'lm5p': lm5p, 'save_realign_dir': save_realign_dir,
                      'output_size': output_size, 'transform_size': transform_size, 'enable_padding': enable_padding}
            inputs.append(input)
        break
    # with multiprocessing.Pool(n_threads) as p:
    #     results = list(tqdm(p.imap(process_image, inputs), total=len(inputs), smoothing=0.1))
    for input in tqdm(inputs):
        process_image(input)


def recreate_aligned_videos_fast(root_dir, lms_root_dir, dst_dir, valid_video_json, save_realign=True, skip=False, enable_warping=True,
                                 output_size=1024, transform_size=4096, enable_padding=None, n_threads=12, apply_GF=0):
    print('Recreating aligned images...')
    assert enable_padding in [None, 'zero_padding', 'blur_padding', 'reflect_padding'], enable_padding
    valid_idx = json.loads(open(valid_video_json).read())
    inputs = []
    save_realign_dir = save_detection_dir = None
    for video_name, im_names in valid_idx:
        video_dir = os.path.join(root_dir, video_name)
        dst_save_dir = os.path.join(dst_dir, video_name)
        if save_realign:
            save_realign_dir = os.path.join(os.path.dirname(os.path.dirname(dst_dir)), 'realign', video_name)
            save_detection_dir = os.path.join(os.path.dirname(os.path.dirname(dst_dir)), 'realign_detections', video_name)
        if not os.path.isdir(video_dir): continue
        if not os.path.exists(os.path.join(lms_root_dir, video_name+'.json')): continue
        if skip and os.path.exists(os.path.join(dst_save_dir, 'finish')): continue   #### skip

        lm5p_dict = json.loads(open(os.path.join(lms_root_dir, video_name+'.json')).read())
        input = {'src_dir': video_dir, 'dst_dir': dst_save_dir, 'lm5p': lm5p_dict, 'im_names': im_names, 'save_realign_dir': save_realign_dir,
                 'save_detection_dir': save_detection_dir, 'output_size': output_size, 'transform_size': transform_size,
                 'enable_padding': enable_padding, 'apply_GF':apply_GF, 'enable_warping':enable_warping}
        inputs.append(input)

    with multiprocessing.Pool(n_threads) as p:
        results = list(tqdm(p.imap(process_video, inputs), total=len(inputs), smoothing=0.1))
    # for input in tqdm(inputs):
    #     process_video(input)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='.')
    parser.add_argument('--lm_source', type=str, default='')
    parser.add_argument('--dest', type=str, default='realign1500')
    parser.add_argument('--valid_video_json', type=str, default=None)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--output_size', type=int, default=768)
    parser.add_argument('--transform_size', type=int, default=768)
    parser.add_argument('--apply_GF', type=float, default=0)
    parser.add_argument('--save_realign', action='store_true')
    parser.add_argument('--skip', action='store_true')
    parser.add_argument('--disable_warping', action='store_true')
    parser.add_argument('--padding_mode', type=str, default=None)
    args = parser.parse_args()

    # recreate_aligned_images_fast(args.source, args.lm_source, args.dest, args.save_realign_dir, args.valid_video_json,
    #                              output_size=args.output_size, transform_size=args.transform_size, n_threads=args.threads)
    recreate_aligned_videos_fast(args.source, args.lm_source, args.dest, args.valid_video_json, save_realign=args.save_realign, skip=args.skip,
                                 output_size=args.output_size, transform_size=args.transform_size, n_threads=args.threads, apply_GF=args.apply_GF,
                                 enable_padding=args.padding_mode, enable_warping=not args.disable_warping)

    # run_cmdline(sys.argv)

# ----------------------------------------------------------------------------
