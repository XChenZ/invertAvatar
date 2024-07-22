import shutil
import os
import json

from tqdm import tqdm
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from preprocess_faceverse import make_cam_dataset_FFHQ, render_orth_mp


def smooth_cam():
    if True:
        img_dir = '/media/zxc/hdd2/Dataset/MV-video/singleView/obama/obama2_animation_dataset/dataset/images512x512'
        ori_json_path = os.path.join(img_dir, 'dataset_realcam.json')
        # ori_json_path = '/media/zxc/hdd2/Dataset/HDTF_test/mergedVFHQ_smooth_crop_dataset/dataset/images512x512/dataset_realcam.json'
        with open(ori_json_path, 'rb') as f:
            ori_json = json.load(f)['labels']
        new_json = []
        for sub_name in os.listdir(img_dir):
            if not os.path.isdir(os.path.join(img_dir, sub_name)): continue
            sub_json = [case for case in ori_json if case[0].split('/')[0] == sub_name]
            sub_json.sort(key=lambda x:int(x[0].split('/')[1].split('.')[0]))

            coeff_seq = np.asarray([x[1] for x in sub_json], dtype=np.float32)
            coeff_seq = gaussian_filter1d(coeff_seq, sigma=1.5, axis=0)

            new_json.extend([[x[0], coeff_seq[idx].tolist()] for idx, x in enumerate(sub_json)])

        with open(ori_json_path.replace('.json', '_smooth.json'), "w") as f:
            json.dump({"labels": new_json}, f, indent=4)


def make_faceverse_labels_FFHQ(tracking_dir, root_dir, fv2fl_T_path, focal, need_render=False, save_uv=True, save_mesh=False, save_name=None,
                               render_normal_uv=False, num_thread=1, use_smooth=False, test_data=False, skip=False):

    save_dir = os.path.join(root_dir, 'dataset')

    fv2fl_T = np.load(fv2fl_T_path).astype(np.float32)
    orth_scale, orth_shift, box_warp = 5.00, np.asarray([0, 0.005, 0.], dtype=np.float32), 2.

    face_model_dir = '/media/zxc/10T/Code/PROJECT/metaface_fitting/metamodel/v3'
    save_render_dir = os.path.join(save_dir, 'orthRender256x256_face_eye' if save_name is None else save_name)
    save_mesh_dir = None if not save_mesh else os.path.join(save_dir, 'FVmeshes512x512')
    save_uv_dir = None if not save_uv else os.path.join(save_dir, 'uvRender256x256')

    render_orth_mp(tracking_dir, save_render_dir, face_model_dir, fv2fl_T, {'scale': orth_scale, 'shift': orth_shift}, focal, render_vis=need_render,
                   save_mesh_dir=save_mesh_dir, save_uv_dir=save_uv_dir, render_normal_uv=render_normal_uv, skip=skip,
                   num_thread=num_thread, crop_param=[128, 114, 256, 256], save_coeff=True)

    normalizeFL_T = np.eye(4, dtype=np.float32)
    scale_T = (orth_scale / box_warp) * np.eye(3, dtype=np.float32)
    shift_T = scale_T.dot(orth_shift.reshape(3, 1))
    normalizeFL_T[:3, :3], normalizeFL_T[:3, 3:] = scale_T, shift_T

    fv2fl_T = np.dot(normalizeFL_T, fv2fl_T)

    cam_params, cond_cam_params, fv_exp_eye_params = make_cam_dataset_FFHQ(tracking_dir, fv2fl_T, focal, test_data=test_data)
    print(len(cam_params))
    if test_data:
        for prefix in cam_params.keys():
            save_json_name = 'dataset_%s_realcam.json' % prefix
            with open(os.path.join(save_dir, 'images512x512', save_json_name), "w") as f:
                json.dump({"labels": cam_params[prefix]}, f, indent=4)
    else:
        save_json_name = 'dataset_realcam.json'
        if use_smooth:
            new_json = []
            for sub_name in os.listdir(os.path.join(save_dir, 'images512x512')):
                if not os.path.isdir(os.path.join(save_dir, 'images512x512', sub_name)): continue
                sub_json = [case for case in cam_params if case[0].split('/')[0] == sub_name]
                sub_json.sort(key=lambda x: int(x[0].split('/')[1].split('.')[0]))

                coeff_seq = np.asarray([x[1] for x in sub_json], dtype=np.float32)
                coeff_seq = gaussian_filter1d(coeff_seq, sigma=1.5, axis=0)

                new_json.extend([[x[0], coeff_seq[idx].tolist()] for idx, x in enumerate(sub_json)])
            cam_params = new_json
        with open(os.path.join(save_dir, 'images512x512', save_json_name), "w") as f:
            json.dump({"labels": cam_params}, f, indent=4)

    make_coeff_dataset_FFHQ(tracking_dir, os.path.join(save_dir, 'coeffs'), smooth=use_smooth)


def make_BFMexp_dataset(ori_dataset_path, ori_dataset_json_path):
    ori_dataset_json = json.loads(open(ori_dataset_json_path).read())['labels']

    dataset_BFM_exp_json = []
    for sub in tqdm(ori_dataset_json):
        subdir, img_name = sub[0].split('/')
        name = img_name.split('.')[0]
        if False:#int(name.split('.')[0][3:]) % 2 == 1: # for FFHQ
            bfm_path = os.path.join(ori_dataset_path, subdir, 'img%08d.mat' % (int(name.split('.')[0][3:]) - 1))
        else:
            bfm_path = os.path.join(ori_dataset_path, subdir, name + '.mat')
        if not os.path.exists(bfm_path):
            print(bfm_path)
            continue
        bfm_expr = loadmat(bfm_path)['exp']
        dataset_BFM_exp_json.append([sub[0], bfm_expr[0, :32].tolist()])

    with open(os.path.join(os.path.dirname(ori_dataset_json_path), 'dataset_BFMexp.json'), 'w') as f:
        f.write(json.dumps({'labels': dataset_BFM_exp_json}, indent=4))


def make_coeff_dataset_FFHQ(tracking_dir, save_dir, smooth=False):
    for prefix in tqdm(os.listdir(tracking_dir)):
        if not os.path.isdir(os.path.join(tracking_dir, prefix)):
            continue
        os.makedirs(os.path.join(save_dir, prefix), exist_ok=True)
        sub_dir = os.path.join(tracking_dir, prefix)
        fname_ls = [name for name in os.listdir(sub_dir) if os.path.exists(os.path.join(sub_dir, name, 'finish'))]

        fname_ls.sort(key=lambda x: int(x))
        coeff_seq = np.stack([np.load(os.path.join(sub_dir, fname, 'coeffs.npy')) for fname in fname_ls], axis=0)
        if smooth: coeff_seq = gaussian_filter1d(coeff_seq, sigma=0.5, axis=0)
        for idx, fname in enumerate(fname_ls):
            dst_path = os.path.join(save_dir, prefix, fname + '.npy')
            np.save(dst_path, coeff_seq[idx])



def smooth_coeffs(coeff_dir):
    save_dir = coeff_dir
    for sub_name in os.listdir(coeff_dir):
        if not os.path.isdir(os.path.join(coeff_dir, sub_name)): continue
        sub_coeff_dir = os.path.join(coeff_dir, sub_name)
        fname_ls = [fname for fname in os.listdir(sub_coeff_dir) if fname.endswith('npy')]
        fname_ls.sort(key=lambda x: int(x.split('.')[0]))

        coeff_seq = np.stack([np.load(os.path.join(sub_coeff_dir, fname)) for fname in fname_ls], axis=0)
        coeff_seq = gaussian_filter1d(coeff_seq, sigma=0.5, axis=0)
        for idx, fname in enumerate(fname_ls):
            np.save(os.path.join(save_dir, sub_name, fname), coeff_seq[idx])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking_dir', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--test_data', action='store_true', default=False)
    parser.add_argument('--skip', action='store_true', default=False)
    args = parser.parse_args()
    make_faceverse_labels_FFHQ(tracking_dir=args.tracking_dir, root_dir=args.root_dir,
                               fv2fl_T_path='FaceVerse/v3/fv2fl_30.npy', need_render=False,
                               save_mesh=False, focal=4.2647, num_thread=1 if args.test_data else 8, test_data=args.test_data, skip=args.skip)
