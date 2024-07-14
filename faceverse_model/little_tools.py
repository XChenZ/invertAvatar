import numpy as np
from faceverse_model.FaceVerseModel_v3 import get_recon_model
import os
import torch
import faceverse_model.utils as utils
from pytorch3d.renderer import look_at_view_transform
import json


def check_depth():
    inst_dir = '/media/zxc/hdd2/Dataset/MV-video/singleView/wlz_0318/video_track_singleView_v30/55'
    metaFace_coeffs = torch.from_numpy(np.load(os.path.join(inst_dir, 'coeffs.npy'))).unsqueeze(0).to('cuda:0')
    calib = json.loads(open('/media/zxc/hdd2/Dataset/MV-video/singleView/wlz_0318/calib_1024.json').read())
    # view_ls = []
    metaFace_extr = np.load(os.path.join(inst_dir, 'metaFace_extr_new.npz'))
    valid_view_name = ['0']  # 确保首个是正面视角
    cam_dist = 5.
    fix_cam_R, fix_cam_t = look_at_view_transform(dist=cam_dist, elev=0, azim=0)
    view_T = np.eye(4, dtype=np.float32)
    view_T[:3, :3] = fix_cam_R[0].cpu().detach().numpy()
    view_T[:3, -1] = fix_cam_t[0].cpu().detach().numpy()

    pcl = []
    for idx, view_name in enumerate(valid_view_name):
        cam_T = np.asarray(calib["intrinsics"][view_name]['cam_T'], dtype=np.float32).reshape(4, 4)
        if idx == 0:
            cam_T0 = cam_T.copy()
        camT_tensor = torch.from_numpy(np.dot(np.linalg.inv(view_T), np.dot(cam_T, np.dot(np.linalg.inv(cam_T0), view_T)))).cuda()
        camK = np.asarray(calib["intrinsics"][view_name]['cam_K'], dtype=np.float32).reshape(3, 3)
        recon_model = get_recon_model(device='cuda', batch_size=1, img_size=calib['img_res'], cam_dist=cam_dist, render_depth=True, intr=camK)
        rgb, depth = recon_model(metaFace_coeffs, render=True, camT=camT_tensor)['rendered_img']
        depth = depth.cpu().numpy().squeeze()
        np.save(os.path.join(inst_dir, 'test_depth.npy'), depth)
        camT_mesh2glo = np.dot(np.linalg.inv(cam_T0), metaFace_extr['transformation']).astype(np.float32)
        camT_cam2mesh = np.linalg.inv(np.dot(cam_T, camT_mesh2glo)) # 对于单目tracking来说就是inv(metaFace_extr['transformation'])
        print(camT_cam2mesh-np.linalg.inv(metaFace_extr['transformation']))
        pcl.append(utils.map_depth_to_3D(depth, (depth>0).astype(np.float32), np.linalg.inv(camK), camT_cam2mesh))
    pcl = np.concatenate(pcl, axis=0)
    utils.save_obj(os.path.join(inst_dir, 'unproject_depth.obj'), pcl)

    id_coeff, exp_coeff, _, angles = recon_model.split_coeffs(metaFace_coeffs)[:4]
    vs = recon_model.get_vs(id_coeff, exp_coeff).cpu().numpy()[0]
    # vs = recon_model.rigid_transform(vs, recon_model.compute_rotation_matrix(angles), torch.zeros(1, 3).to('cuda'), 1.0)
    utils.save_obj(os.path.join(inst_dir, 'rot_mesh.obj'), vs, recon_model.tri + 1)

    if True:
        fv2fl_T = np.load(os.path.join('/media/zxc/hdd2/Dataset/MV-video/singleView/IMG_2519', 'fv2fl_30.npy')).astype(np.float32)
        vs = np.dot(fv2fl_T[:3, :3], vs.transpose()) + fv2fl_T[:3, 3:]
        utils.save_obj(os.path.join(inst_dir, 'rot_mesh_fl.obj'), vs.transpose(), recon_model.tri + 1)
