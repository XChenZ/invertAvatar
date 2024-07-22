import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import torchvision
from render_utils.renderer import batch_orth_proj, angle2matrix, face_vertices, render_after_rasterize
from render_utils.ortho_renderer import get_renderer
from FaceVerse.FaceVerseModel_v3 import ModelRenderer
import torchvision.utils as utils
from tqdm import tqdm
from FaceVerse import get_recon_model
import time
from pytorch3d.structures import Meshes
import json
import multiprocessing
import shutil
count, total = multiprocessing.Value('i', 0), multiprocessing.Value('i', 0)


def load_obj_data(filename):
    """load model data from .obj file"""
    v_list = []  # vertex coordinate
    vt_list = []  # vertex texture coordinate
    vc_list = []  # vertex color
    vn_list = []  # vertex normal
    f_list = []  # face vertex indices
    fn_list = []  # face normal indices
    ft_list = []  # face texture indices

    # read data
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()

    for line in lines:
        if len(line) < 2:
            continue
        line_data = line.strip().split(' ')
        # parse vertex cocordinate
        if line_data[0] == 'v':
            v_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))
            if len(line_data) == 7:
                vc_list.append((float(line_data[4]), float(line_data[5]), float(line_data[6])))
            else:
                vc_list.append((0.5, 0.5, 0.5))

        # parse vertex texture coordinate
        if line_data[0] == 'vt':
            vt_list.append((float(line_data[1]), float(line_data[2])))

        # parse vertex normal
        if line_data[0] == 'vn':
            vn_list.append((float(line_data[1]), float(line_data[2]), float(line_data[3])))

        # parse face
        if line_data[0] == 'f':
            # used for parsing face element data
            def segElementData(ele_str):
                fv = None
                ft = None
                fn = None
                eles = ele_str.strip().split('/')
                if len(eles) == 1:
                    fv = int(eles[0]) - 1
                elif len(eles) == 2:
                    fv = int(eles[0]) - 1
                    ft = int(eles[1]) - 1
                elif len(eles) == 3:
                    fv = int(eles[0]) - 1
                    fn = int(eles[2]) - 1
                    ft = None if eles[1] == '' else int(eles[1]) - 1
                return fv, ft, fn

            fv0, ft0, fn0 = segElementData(line_data[1])
            fv1, ft1, fn1 = segElementData(line_data[2])
            fv2, ft2, fn2 = segElementData(line_data[3])
            f_list.append((fv0, fv1, fv2))
            if ft0 is not None and ft1 is not None and ft2 is not None:
                ft_list.append((ft0, ft1, ft2))
            if fn0 is not None and fn1 is not None and fn2 is not None:
                fn_list.append((fn0, fn1, fn2))

    v_list = np.asarray(v_list)
    vn_list = np.asarray(vn_list)
    vt_list = np.asarray(vt_list)
    vc_list = np.asarray(vc_list)
    f_list = np.asarray(f_list)
    ft_list = np.asarray(ft_list)
    fn_list = np.asarray(fn_list)

    model = {'v': v_list, 'vt': vt_list, 'vc': vc_list, 'vn': vn_list,
             'f': f_list, 'ft': ft_list, 'fn': fn_list}
    return model


def save_obj_data(model, filename, log=True):
    import numpy as np
    assert 'v' in model and model['v'].size != 0

    with open(filename, 'w') as fp:
        if 'v' in model and model['v'].size != 0:
            if 'vc' in model and model['vc'].size != 0:
                assert model['vc'].size == model['v'].size
                for v, vc in zip(model['v'], model['vc']):
                    fp.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], vc[2], vc[1], vc[0]))
            else:
                for v in model['v']:
                    fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

        if 'vn' in model and model['vn'].size != 0:
            for vn in model['vn']:
                fp.write('vn %f %f %f\n' % (vn[0], vn[1], vn[2]))

        if 'vt' in model and model['vt'].size != 0:
            for vt in model['vt']:
                fp.write('vt %f %f\n' % (vt[0], vt[1]))

        if 'f' in model and model['f'].size != 0:
            if 'fn' in model and model['fn'].size != 0 and 'ft' in model and model['ft'].size != 0:
                assert model['f'].size == model['fn'].size
                assert model['f'].size == model['ft'].size
                for f_, ft_, fn_ in zip(model['f'], model['ft'], model['fn']):
                    f = np.copy(f_) + 1
                    ft = np.copy(ft_) + 1
                    fn = np.copy(fn_) + 1
                    fp.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' %
                             (f[0], ft[0], fn[0], f[1], ft[1], fn[1], f[2], ft[2], fn[2]))
            elif 'fn' in model and model['fn'].size != 0:
                assert model['f'].size == model['fn'].size
                for f_, fn_ in zip(model['f'], model['fn']):
                    f = np.copy(f_) + 1
                    fn = np.copy(fn_) + 1
                    fp.write('f %d//%d %d//%d %d//%d\n' % (f[0], fn[0], f[1], fn[1], f[2], fn[2]))
            elif 'ft' in model and model['ft'].size != 0:
                assert model['f'].size == model['ft'].size
                for f_, ft_ in zip(model['f'], model['ft']):
                    f = np.copy(f_) + 1
                    ft = np.copy(ft_) + 1
                    fp.write('f %d/%d %d/%d %d/%d\n' % (f[0], ft[0], f[1], ft[1], f[2], ft[2]))
            else:
                for f_ in model['f']:
                    f = np.copy(f_) + 1
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if log:
        print("Saved mesh as " + filename)


def gen_mouth_mask(lms_2d, new_crop=True):
    lm = lms_2d[np.newaxis, ...]
    # # lm: (B, 68, 2) [-1, 1]
    if new_crop:
        lm_mouth_outer = lm[:, [164, 18, 57, 287]]  # up, bottom, left, right corners and others
        # lm_mouth_outer = lm[:, [2, 200, 212, 432]]  # up, bottom, left, right corners and others
        mouth_mask = np.concatenate([np.min(lm_mouth_outer[..., 1], axis=1, keepdims=True),
                                     np.max(lm_mouth_outer[..., 1], axis=1, keepdims=True),
                                     np.min(lm_mouth_outer[..., 0], axis=1, keepdims=True),
                                     np.max(lm_mouth_outer[..., 0], axis=1, keepdims=True)], 1)  # (B, 4)
    else:
        lm_mouth_outer = lm[:, [0, 17, 61, 291, 39, 269, 405, 181]]  # up, bottom, left, right corners and others
        mouth_avg = np.mean(lm_mouth_outer, axis=1, keepdims=False)  # (B, 2)
        ups, bottoms = np.max(lm_mouth_outer[..., 0], axis=1, keepdims=True), np.min(lm_mouth_outer[..., 0], axis=1, keepdims=True)
        lefts, rights = np.min(lm_mouth_outer[..., 1], axis=1, keepdims=True), np.max(lm_mouth_outer[..., 1], axis=1, keepdims=True)
        mask_res = np.max(np.concatenate((ups - bottoms, rights - lefts), axis=1), axis=1, keepdims=True) * 1.2
        mask_res = mask_res.astype(int)
        mouth_mask = np.concatenate([(mouth_avg[:, 1:] - mask_res // 2).astype(int),
                                     (mouth_avg[:, 1:] + mask_res // 2).astype(int),
                                     (mouth_avg[:, 0:1] - mask_res // 2).astype(int),
                                     (mouth_avg[:, 0:1] + mask_res // 2).astype(int)], 1)  # (B, 4)
    return mouth_mask[0]


def render_orth(tracking_dir, save_dir, face_model_dir, fv2fl_T, orth_transforms, render_vis=True, save_mesh_dir=None):
    debug = False
    save_mesh_flag = save_mesh_dir is not None
    res = 256
    ortho_renderer = get_renderer(img_size=res, device='cuda:0', T=torch.tensor([[0, 0, 10.]], dtype=torch.float32, device='cuda:0'),
                                  K=[-1.0, -1.0, 0., 0.], orthoCam=True, rasterize_blur_radius=1e-6)
    orth_scale, orth_shift = orth_transforms['scale'], torch.from_numpy(orth_transforms['shift']).cuda().unsqueeze(0)

    # model_topology_path = os.path.join(face_model_dir, 'base_mesh.obj')
    face_model_path = os.path.join(face_model_dir, 'faceverse_v3_1.npy')

    recon_model, model_dict = get_recon_model(model_path=face_model_path, return_dict=True, device='cuda:0')
    vert_uvcoords = model_dict['uv_per_ver']
    if True: # 扩大face部分在UV图中占据的面积
        vert_idx = (vert_uvcoords[:, 1] > 0.273) * (vert_uvcoords[:, 1] < 0.727) * (vert_uvcoords[:, 0] > 0.195) * (vert_uvcoords[:, 0] < 0.805)
        vert_uvcoords[vert_idx] = (vert_uvcoords[vert_idx] - 0.5) * 1.4 + 0.5
    vert_uvcoords = torch.from_numpy(vert_uvcoords).unsqueeze(0).cuda()
    faces = uvfaces = torch.from_numpy(model_dict['tri']).unsqueeze(0).cuda()
    # vert_mask = model_dict['face_mask']
    vert_mask = np.load('metamodel/v3/v31_face_mask_new.npy')
    vert_mask[model_dict['ver_inds'][0]:model_dict['ver_inds'][2]] = 1
    vert_mask = torch.from_numpy(vert_mask).view(1, -1, 1).cuda()
    vert_uvcoords = vert_uvcoords * 2 - 1
    vert_uvcoords = torch.cat([vert_uvcoords, vert_mask], -1)  # [bz, ntv, 3]
    face_uvcoords = face_vertices(vert_uvcoords, uvfaces).cuda()  # 面片不反向

    if save_mesh_flag:
        tri = recon_model.tri.cpu().numpy().squeeze()
        uv = recon_model.uv.cpu().numpy().squeeze()
        tri_uv = recon_model.tri_uv.cpu().numpy().squeeze()
    trans_init = torch.from_numpy(fv2fl_T).cuda()
    R_ = trans_init[:3, :3]
    t_ = trans_init[:3, 3:]

    tform = angle2matrix(torch.tensor([0, 0, 0]).reshape(1, -1)).cuda()
    cam = torch.tensor([1., 0, 0]).cuda()
    count = 0
    mouth_masks = []
    total_num = len(os.listdir(tracking_dir))
    # bar = tqdm(os.listdir(tracking_dir)[(total_num // 3 + 1) * 2:(total_num // 3 + 1) * 3])
    bar = tqdm(os.listdir(tracking_dir))
    t0 = time.time()
    for name in bar:
        # ind = (int(name.split('.')[0][3:]) - 1)   # FFHQ
        # prefix = '%05i' % int(ind/1000)
        prefix = '0'
        # if not name =='6150':
        #     continue

        dst_sub_dir = os.path.join(save_dir, prefix)
        os.makedirs(dst_sub_dir, exist_ok=True)

        coeff = torch.from_numpy(np.load(os.path.join(tracking_dir, name, 'coeffs.npy'))).unsqueeze(0).cuda()
        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, eye_coeff, scale = recon_model.split_coeffs(coeff)
        vs = recon_model.get_vs(id_coeff, exp_coeff)
        vert = torch.matmul(vs[0], R_.T) + t_.T

        v = vert.unsqueeze(0)
        transformed_vertices = (torch.bmm(v, tform) + orth_shift) * orth_scale  # 此处只验证了正面视角渲染时结果与之前相同，由于是先乘tform再trans&shift，不确保其它视角渲染与原有方法相同
        transformed_vertices = batch_orth_proj(transformed_vertices, cam)  ##无变化操作
        transformed_vertices = torch.bmm(transformed_vertices, angle2matrix(torch.tensor([0, 180, 0]).reshape(1, -1)).cuda())
        if save_mesh_flag:
            mesh = {'v': transformed_vertices.squeeze().cpu().numpy(), 'vt': uv, 'f': tri, 'ft': tri_uv}
            os.makedirs(os.path.join(save_mesh_dir, prefix), exist_ok=True)
            # print(os.path.join(save_mesh_dir, prefix, name.split('.')[0] + '.obj'))
            save_obj_data(mesh, os.path.join(save_mesh_dir, prefix, name.split('.')[0] + '.obj'), log=False)

        mesh = Meshes(transformed_vertices, faces.long())
        fragment = ortho_renderer.rasterizer(mesh)

        rendering = render_after_rasterize(attributes=face_uvcoords, pix_to_face=fragment.pix_to_face, bary_coords=fragment.bary_coords)    # [1, 4, H, W]
        uvcoords_images, render_mask = rendering[:, :-1, :, :], rendering[:, -1:, :, :]
        render_mask *= uvcoords_images[:, -1:]
        # uvcoords_images[:, -1:] *= render_mask
        uvcoords_images *= render_mask
        np.save(os.path.join(dst_sub_dir, name.split('.')[0] + '.npy'), rendering[0].permute(1, 2, 0).cpu().numpy())
        if render_vis:
            utils.save_image(uvcoords_images, os.path.join(dst_sub_dir, name.split('.')[0] + '.png'), normalize=True, range=(-1, 1))

        lms_3d = recon_model.get_lms(transformed_vertices).cpu().squeeze().numpy()
        lms_2d = np.round((lms_3d[:, :2] + 1) * 0.5 * res).astype(np.uint8)
        mouth_mask = gen_mouth_mask(lms_2d)
        mouth_masks.append([prefix+'/'+name.split('.')[0] + '.png', mouth_mask.tolist()])
        # np.save(os.path.join(dst_sub_dir, name.split('.')[0] + '.npy'), {'bary_coords': fragment.bary_coords[0].cpu().numpy(),
        #                                                                  'pix_to_face': fragment.pix_to_face[0].cpu().numpy(),
        #                                                                  'mouth_mask': mouth_mask})
        if debug:  # debug
            face_mask_path = os.path.join(face_model_dir, 'FV_face_mask.npy')
            vert_color = recon_model.get_color(tex_coeff)
            face_color = face_vertices(vert_color, faces).cuda()  # 面片反向
            face_mask = torch.from_numpy(np.load(face_mask_path).astype(np.float32)).unsqueeze(0).unsqueeze(-1).cuda()
            print(face_color.shape, face_mask.shape)
            face_color *= face_mask.unsqueeze(-1)
            face_color = (face_color / 255 * 2) - 1
            # rendering = render_after_rasterize(attributes=face_uvcoords, pix_to_face=fragment.pix_to_face, bary_coords=fragment.bary_coords)
            rendering = render_after_rasterize(attributes=face_color, pix_to_face=fragment.pix_to_face, bary_coords=fragment.bary_coords)
            print(rendering.shape)
            vis_uvcoords_images, render_mask = rendering[:, :-1, :, :], rendering[:, -1:, :, :]
            # vis_uvcoords_images = torch.cat([uvcoords_image, facemask_image], dim=1)
            vis_uvcoords_images[:, :, mouth_mask[0]:mouth_mask[1], mouth_mask[2]:mouth_mask[3]] *= 0  #####
            utils.save_image(vis_uvcoords_images, os.path.join(dst_sub_dir, name.split('.')[0] + '.png'), normalize=True, range=(-1, 1))
        count += 1
        # print(count, os.path.join(dst_sub_dir, name.split('.')[0] + '.npy'))
        bar.set_description('%s %03i' % (name.split('.')[0], int(1000 * (time.time() - t0) / count)))

    with open(os.path.join(save_dir, 'mouth_masks.json'), "w") as f:
        json.dump(mouth_masks, f, indent=4)


def render_orth_mp(tracking_dir, save_dir, face_model_dir, fv2fl_T, orth_transforms, focal_ratio, render_vis=False, save_mesh_dir=None, save_uv_dir=None,
                   num_thread=1, render_normal_uv=False, prefix_ls=None, crop_param=None, use_smooth=False, save_coeff=False, skip=False):
    print('Num Threads: ', num_thread)
    if num_thread > 1:
        data_ls = [{'tracking_dir':os.path.join(tracking_dir, prefix), 'save_dir':save_dir, 'face_model_dir':face_model_dir, 'fv2fl_T':fv2fl_T,
                    'orth_transforms':orth_transforms, 'render_vis':render_vis, 'save_mesh_dir':save_mesh_dir, 'save_uv_dir':save_uv_dir,
                    'prefix':prefix, 'render_normal_uv':render_normal_uv, 'crop_param':crop_param, 'use_smooth':use_smooth, 'focal_ratio': focal_ratio,
                    'save_coeff': save_coeff}
                   for prefix in os.listdir(tracking_dir)
                   if (os.path.isdir(os.path.join(tracking_dir, prefix))) and (not os.path.exists(os.path.join(save_dir, prefix))if skip else True)]
        num_thread = min(num_thread, len(data_ls))
        p = multiprocessing.Pool(num_thread)
        p.map(perform_render, data_ls)
        p.close()
        p.join()
    else:
        if prefix_ls is None:
            for prefix in os.listdir(tracking_dir):
                if os.path.isdir(os.path.join(tracking_dir, prefix)):
                    # if not int(prefix) in range(0, 70): continue
                    perform_render({'tracking_dir':os.path.join(tracking_dir, prefix), 'save_dir':save_dir, 'face_model_dir':face_model_dir,
                                    'fv2fl_T':fv2fl_T, 'orth_transforms':orth_transforms, 'render_vis':render_vis, 'save_mesh_dir':save_mesh_dir,
                                    'save_uv_dir':save_uv_dir, 'prefix':prefix, 'render_normal_uv':render_normal_uv, 'crop_param':crop_param,
                                    'use_smooth':use_smooth, 'focal_ratio':focal_ratio, 'save_coeff': save_coeff})
        else:
            for prefix in prefix_ls:
                if prefix == '': prefix = '0'
                perform_render(
                    {'tracking_dir': tracking_dir, 'save_dir': save_dir, 'face_model_dir': face_model_dir, 'fv2fl_T': fv2fl_T, 'focal_ratio': focal_ratio,
                     'orth_transforms': orth_transforms, 'render_vis': render_vis, 'save_mesh_dir': save_mesh_dir, 'save_uv_dir':save_uv_dir,
                     'prefix': prefix, 'render_normal_uv': render_normal_uv, 'crop_param':crop_param, 'use_smooth':use_smooth, 'save_coeff': save_coeff})

    # ## merge mouth_mask
    # mouth_masks = []
    # for name in os.listdir(save_dir):
    #     if len(name.split('_'))>2 and name.startswith('mouth_masks'):
    #         mouth_masks += json.loads(open(os.path.join(save_dir, name)).read())
    #
    # with open(os.path.join(save_dir, 'mouth_masks.json'), "w") as f:
    #     json.dump(mouth_masks, f, indent=4)

    # mouth_masks = []
    # for name in os.listdir(save_dir):
    #     if len(name.split('_')) > 2 and name.startswith('old_mouth_masks'):
    #         mouth_masks += json.loads(open(os.path.join(save_dir, name)).read())
    #
    # with open(os.path.join(save_dir, 'old_mouth_masks.json'), "w") as f:
    #     json.dump(mouth_masks, f, indent=4)

def perform_render(data):
    render_orth_(data)
    if data['save_uv_dir'] is not None: save_uv_(data)

def save_uv_(data):
    tracking_dir, save_uv_dir, face_model_dir, prefix, focal_ratio, render_normal_uv = \
        data['tracking_dir'], data['save_uv_dir'], data['face_model_dir'], data['prefix'],  data['focal_ratio'], data['render_normal_uv']

    img_res, render_res = 512, 256  # 此处默认真实图片分辨率为512
    uv_renderer = get_renderer(img_size=render_res, device='cuda:0', T=torch.tensor([[0, 0, 10.]], dtype=torch.float32, device='cuda:0'),
                                  K=[-1.0, -1.0, 0., 0.], orthoCam=True, rasterize_blur_radius=1e-6)
    cam_K = np.eye(3, dtype=np.float32); cam_K[0, 0] = cam_K[1, 1] = focal_ratio * img_res; cam_K[0, 2] = cam_K[1, 2] = img_res // 2
    renderer = ModelRenderer(img_size=img_res, device='cuda:0', intr=cam_K, cam_dist=5.)
    face_model_path = os.path.join(face_model_dir, 'faceverse_v3_1.npy')

    recon_model, model_dict = get_recon_model(model_path=face_model_path, return_dict=True, device='cuda:0', img_size=img_res, intr=cam_K, cam_dist=5)
    vert_uvcoords = model_dict['uv_per_ver']
    if True:  # 扩大face部分在UV图中占据的面积
        vert_idx = (vert_uvcoords[:, 1] > 0.273) * (vert_uvcoords[:, 1] < 0.727) * (vert_uvcoords[:, 0] > 0.195) * (vert_uvcoords[:, 0] < 0.805)
        vert_uvcoords[vert_idx] = (vert_uvcoords[vert_idx] - 0.5) * 1.4 + 0.5
    vert_uvcoords = torch.from_numpy(vert_uvcoords).unsqueeze(0).cuda()
    faces = torch.from_numpy(model_dict['tri']).unsqueeze(0).cuda()
    vert_mask = np.load(os.path.join(face_model_dir, 'v31_face_mask_new.npy'))
    vert_mask[model_dict['ver_inds'][0]:model_dict['ver_inds'][2]] = 1

    vert_mask = torch.from_numpy(vert_mask).view(1, -1, 1).cuda()
    vert_uvcoords = vert_uvcoords * 2 - 1
    vert_mask[0, ~vert_idx] *= 0  # for UV rendering
    vert_uvcoords = torch.cat([vert_uvcoords, (1 - vert_mask)], -1)
    uv_fragment = uv_renderer.rasterizer(Meshes(vert_uvcoords, faces.long()))

    uv_face_eye_mask = cv2.imread(os.path.join(face_model_dir, 'dense_uv_expanded_mask_onlyFace.png'))[..., 0]
    uv_face_eye_mask = torch.from_numpy(uv_face_eye_mask.astype(np.float32) / 255).view(1, 256, 256, 1).permute(0, 3, 1, 2)
    os.makedirs(os.path.join(save_uv_dir, prefix), exist_ok=True)

    print('Rendering ', tracking_dir)
    for name in os.listdir(tracking_dir):
        if not os.path.exists(os.path.join(tracking_dir, name, 'finish')):
            print('Not exist ' + os.path.join(tracking_dir, name, 'finish'))
            continue

        coeff = torch.from_numpy(np.load(os.path.join(tracking_dir, name, 'coeffs.npy'))).unsqueeze( 0).cuda()
        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, eye_coeff, scale = recon_model.split_coeffs(coeff)

        l_eye_mat = recon_model.compute_eye_rotation_matrix(eye_coeff[:, :2])
        r_eye_mat = recon_model.compute_eye_rotation_matrix(eye_coeff[:, 2:])
        l_eye_mean = recon_model.get_l_eye_center(id_coeff)
        r_eye_mean = recon_model.get_r_eye_center(id_coeff)

        vs = recon_model.get_vs(id_coeff, exp_coeff, l_eye_mat, r_eye_mat, l_eye_mean, r_eye_mean)

        # save cano vert normal map in UV
        if render_normal_uv:
            vert_norm = recon_model.compute_norm(vs, recon_model.tri, recon_model.point_buf)
            vert_norm = torch.clip((vert_norm + 1) * 127.5, 0, 255)
            vert_norm = torch.cat([vert_norm, vert_mask], dim=-1)
            rendered_normal = render_after_rasterize(attributes=face_vertices(vert_norm, faces), pix_to_face=uv_fragment.pix_to_face,
                                                     bary_coords=uv_fragment.bary_coords).cpu()  # [1, 4, H, W]
            rendered_normal = rendered_normal[:, :3] * (rendered_normal[:, -1:].clone() * rendered_normal[:, -2:-1]) * uv_face_eye_mask
            normal_img = torch.clamp(rendered_normal[0, :3, :, :], 0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(save_uv_dir, prefix, name + '_uvnormal.png'), normal_img[:, :, ::-1])

        # save proj_position map in UV
        rotation = recon_model.compute_rotation_matrix(angles)
        vs_t = recon_model.rigid_transform(vs, rotation, translation, torch.abs(scale))
        vs_norm = recon_model.compute_norm(vs_t, recon_model.tri, recon_model.point_buf)
        vs_proj = renderer.project_vs(vs_t) / img_res * 2 - 1  # [bz, V, 2]
        vert_attr = torch.cat([vs_proj, vert_mask * (vs_norm[..., 2:] > 0.1).float()], dim=-1)

        uv_pverts = render_after_rasterize(attributes=face_vertices(vert_attr, faces), pix_to_face=uv_fragment.pix_to_face,
                                           bary_coords=uv_fragment.bary_coords).cpu()
        uv_pverts = (uv_pverts[:, :-1] * uv_pverts[:, -1:])  # [1, C, H, W]  # proj_position map in UV
        uv_pverts[:, -1:] *= uv_face_eye_mask
        np.save(os.path.join(save_uv_dir, prefix, name + '.npy'), uv_pverts[0].permute(1, 2, 0).numpy().astype(np.float16))

        # images = cv2.imread(os.path.join('/media/zxc/hdd2/Dataset/HDTF_test/HDTF_test/raw_videos', prefix, name + '.png'))
        images = cv2.imread(os.path.join(os.path.dirname(save_uv_dir), 'images512x512', prefix, name+'.png'))
        images = torch.from_numpy(images.astype(np.float32) / 255).view(1, 512, 512, 3).permute(0, 3, 1, 2)
        uv_gt = F.grid_sample(images, uv_pverts.permute(0, 2, 3, 1)[..., :2], mode='bilinear', align_corners=False)
        uv_texture_gt = (uv_gt * uv_pverts[:, -1:] + torch.ones_like(uv_gt) * (1 - uv_pverts[:, -1:]))# * uv_face_eye_mask
        cv2.imwrite(os.path.join(save_uv_dir, prefix, name + '_uvgttex.png'), (uv_texture_gt[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def render_orth_(data):
    tracking_dir, save_dir, face_model_dir, fv2fl_T, orth_transforms, prefix, render_vis, save_mesh_dir, crop_param, use_smooth, save_coeff = \
            data['tracking_dir'], data['save_dir'], data['face_model_dir'], data['fv2fl_T'], data['orth_transforms'], data['prefix'], \
            data['render_vis'], data['save_mesh_dir'], data['crop_param'], data['use_smooth'], data['save_coeff']
    save_mesh_flag = save_mesh_dir is not None
    res, render_res = 256, 512  #render_res512搭配crop_param最終裁剪结果正好256
    ortho_renderer = get_renderer(img_size=render_res, device='cuda:0', T=torch.tensor([[0, 0, 10.]], dtype=torch.float32, device='cuda:0'),
                                  K=[-1.0, -1.0, 0., 0.], orthoCam=True, rasterize_blur_radius=1e-6)
    orth_scale, orth_shift = orth_transforms['scale'], torch.from_numpy(orth_transforms['shift']).cuda().unsqueeze(0)

    face_model_path = os.path.join(face_model_dir, 'faceverse_v3_1.npy')

    recon_model, model_dict = get_recon_model(model_path=face_model_path, return_dict=True, device='cuda:0')
    vert_uvcoords = model_dict['uv_per_ver']
    if True: # 扩大face部分在UV图中占据的面积
        vert_idx = (vert_uvcoords[:, 1] > 0.273) * (vert_uvcoords[:, 1] < 0.727) * (vert_uvcoords[:, 0] > 0.195) * (vert_uvcoords[:, 0] < 0.805)
        vert_uvcoords[vert_idx] = (vert_uvcoords[vert_idx] - 0.5) * 1.4 + 0.5
    vert_uvcoords = torch.from_numpy(vert_uvcoords).unsqueeze(0).cuda()
    faces = uvfaces = torch.from_numpy(model_dict['tri']).unsqueeze(0).cuda()
    # vert_mask = model_dict['face_mask']
    vert_mask = np.load(os.path.join(face_model_dir, 'v31_face_mask_new.npy'))
    vert_mask[model_dict['ver_inds'][0]:model_dict['ver_inds'][2]] = 1

    vert_mask = torch.from_numpy(vert_mask).view(1, -1, 1).cuda()
    vert_uvcoords = vert_uvcoords * 2 - 1
    vert_uvcoords = torch.cat([vert_uvcoords, vert_mask.clone()], -1)  # [bz, ntv, 3]
    face_uvcoords = face_vertices(vert_uvcoords, uvfaces)  # 面片不反向
    vert_mask[0, ~vert_idx] *= 0    # for UV rendering

    if save_mesh_flag:
        tri = recon_model.tri.cpu().numpy().squeeze()
        uv = recon_model.uv.cpu().numpy().squeeze()
        tri_uv = recon_model.tri_uv.cpu().numpy().squeeze()
        os.makedirs(os.path.join(save_mesh_dir, prefix), exist_ok=True)

    trans_init = torch.from_numpy(fv2fl_T).cuda()
    R_ = trans_init[:3, :3]
    t_ = trans_init[:3, 3:]

    tform = angle2matrix(torch.tensor([0, 0, 0]).reshape(1, -1)).cuda()
    cam = torch.tensor([1., 0, 0]).cuda()
    mouth_masks, old_mouth_masks = [], []

    print('Rendering ', tracking_dir)
    for name in os.listdir(tracking_dir):
        if not os.path.exists(os.path.join(tracking_dir, name, 'finish')):
            print('Not exist ' + os.path.join(tracking_dir, name, 'finish'))
            continue
        # if os.path.exists(os.path.join('/media/zxc/hdd2/Dataset/VFHQ/eg3d_dataset/dataset/orthRender256x256_face', prefix, name.split('.')[0]+'.npy')): continue
        # if not os.path.exists(os.path.join('/media/zxc/hdd2/Dataset/VFHQ/eg3d_dataset/dataset/images512x512_2', os.path.basename(tracking_dir), name+'.png')): continue
        # if not int(name) in range(2745, 2746): continue #
        dst_sub_dir = os.path.join(save_dir, prefix)
        os.makedirs(dst_sub_dir, exist_ok=True)

        coeff_path = os.path.join(tracking_dir, name, 'smooth_coeffs.npy' if use_smooth else 'coeffs.npy') #'smooth_coeffs.npy'
        if save_coeff: shutil.copy(coeff_path, os.path.join(dst_sub_dir, name + '_coeff.npy'))
        coeff = torch.from_numpy(np.load(coeff_path)).unsqueeze(0).cuda()
        id_coeff, exp_coeff, tex_coeff, angles, gamma, translation, eye_coeff, scale = recon_model.split_coeffs(coeff)
        # vs = recon_model.get_vs(id_coeff, exp_coeff)

        l_eye_mat = recon_model.compute_eye_rotation_matrix(eye_coeff[:, :2])
        r_eye_mat = recon_model.compute_eye_rotation_matrix(eye_coeff[:, 2:])
        l_eye_mean = recon_model.get_l_eye_center(id_coeff)
        r_eye_mean = recon_model.get_r_eye_center(id_coeff)

        vs = recon_model.get_vs(id_coeff, exp_coeff, l_eye_mat, r_eye_mat, l_eye_mean, r_eye_mean)
        vert = torch.matmul(vs[0], R_.T) + t_.T

        v = vert.unsqueeze(0)
        transformed_vertices = (torch.bmm(v, tform) + orth_shift) * orth_scale  # 此处只验证了正面视角渲染时结果与之前相同，由于是先乘tform再trans&shift，不确保其它视角渲染与原有方法相同
        transformed_vertices = batch_orth_proj(transformed_vertices, cam)  ##无变化操作

        if save_mesh_flag:
            mesh = {'v': transformed_vertices.squeeze().cpu().numpy(), 'vt': uv, 'f': tri, 'ft': tri_uv}
            save_obj_data(mesh, os.path.join(save_mesh_dir, prefix, name + '.obj'), log=False)

        ## 绕Y轴旋转180°：因为orthoRender默认相机朝向 + z，所以必须转个头后才能渲染面部正视图
        ## 但绕Y轴180°会导致射线方向与原版EG3D中相反，阻碍优化速度。在此处正交渲染之前将模型沿Z轴反向，使得按照现有外参（不考虑Z反向）投影到模型表面的点，在正视图投影图上对应的像素点正好是，Z反向后模型渲染结果。
        transformed_vertices[..., -1] *= -1

        mesh = Meshes(transformed_vertices, faces.long())
        fragment = ortho_renderer.rasterizer(mesh)
        rendering = render_after_rasterize(attributes=face_uvcoords, pix_to_face=fragment.pix_to_face,
                                           bary_coords=fragment.bary_coords)  # [1, 4, H, W]
        render_mask = rendering[:, -1:, :, :].clone()
        render_mask *= rendering[:, -2:-1]  # face_mask
        rendering *= render_mask

        if crop_param is not None:  # [left, top, width, height]
            rendering = rendering[:, :, crop_param[1]:crop_param[1]+crop_param[3], crop_param[0]:crop_param[0]+crop_param[2]]
        if not res == rendering.shape[2]:
            rendering = torch.nn.functional.interpolate(rendering, size=(res, res), mode='bilinear', align_corners=False)

        # rendering = torch.flip(rendering, dims=[3]) # 经过v14的check_camera验证，不flip是正确的
        np.save(os.path.join(dst_sub_dir, name + '.npy'), rendering[0].permute(1, 2, 0).cpu().numpy().astype(np.float16))

        # mouth mask
        lms_3d = recon_model.get_lms(transformed_vertices).cpu().squeeze().numpy()
        lms_2d = np.round((lms_3d[:, :2] + 1) * 0.5 * res).astype(np.uint8)
        mouth_mask = gen_mouth_mask(lms_2d, new_crop=False)
        mouth_masks.append([prefix+'/'+name.split('.')[0] + '.png', mouth_mask.tolist()])

        if render_vis:
            boxes = torch.tensor([[mouth_mask[2], mouth_mask[0], mouth_mask[3], mouth_mask[1]]])
            vis_uvcoords = utils.draw_bounding_boxes(((rendering[0, :-1, :, :] + 1) * 127.5).to(dtype=torch.uint8).cpu(), boxes,
                                                                 colors=(0, 255, 0), width=1)
            vis_image = torchvision.transforms.ToPILImage()(vis_uvcoords)
            vis_image.save(os.path.join(dst_sub_dir, name.split('.')[0] + '.png'))



def fill_mouth(images):
    #Input: images: [batch, 1, h, w]
    device = images.device
    mouth_masks = []
    for image in images:
        image = image[0].cpu().numpy()
        image = image * 255.
        copyImg = image.copy()
        h, w = image.shape[:2]
        mask = np.zeros([h+2, w+2],np.uint8)
        cv2.floodFill(copyImg, mask, (0, 0), (255, 255, 255), (0, 0, 0), (254, 254, 254), cv2.FLOODFILL_FIXED_RANGE)
        # cv2.imwrite("debug.png", copyImg)
        copyImg = torch.tensor(copyImg).to(device).to(torch.float32) / 127.5 - 1
        mouth_masks.append(copyImg.unsqueeze(0))
    mouth_masks = torch.stack(mouth_masks, 0)
    mouth_masks = ((mouth_masks * 2. - 1.) * -1. + 1.) / 2.
    # images = (images.bool() | mouth_masks.bool()).float()
    res = (images + mouth_masks).clip(0, 1)

    return res


def rasterize(verts, faces, face_attr, rasterizer, cam_dist=10):
    verts[:, :, 2] = verts[:, :, 2] + cam_dist
    rendering = rasterizer(verts, faces, face_attr, 256, 256)
    return rendering


def ortho_render(verts, faces, face_attr, renderer):
    mesh = Meshes(verts, faces.long())
    rendering = renderer(mesh, face_attr, need_rgb=False)[-1]
    return rendering


def calculate_new_intrinsic(intr, mode, param):
    '''
    mode:   resize,     crop,   padding
    param:  (fx, fy),     (left, top)
    '''
    cam_K = intr.copy()
    if mode == 'resize':
        cam_K[0] *= param[0]
        cam_K[1] *= param[1]
    elif mode == 'crop':
        cam_K[0, 2] = cam_K[0, 2] - param[0]  # -left
        cam_K[1, 2] = cam_K[1, 2] - param[1]  # -top
    elif mode == 'padding':
        cam_K[0, 2] = cam_K[0, 2] + param[2]  # + padding left
        cam_K[1, 2] = cam_K[1, 2] + param[0]  # + padding top
    else:
        assert False
    return cam_K


def make_cam_dataset_FFHQ(tracking_dir, fv2fl_T, focal_ratio=2.568, use_smooth=False, test_data=False):
    # focal_ratio = 2.568
    cam_K = np.eye(3, dtype=np.float32)
    cam_K[0, 0] = cam_K[1, 1] = focal_ratio
    cam_K[0, 2] = cam_K[1, 2] = 0.5

    if test_data:
        cam_params, cond_cam_params, fv_exp_eye_params = {}, {}, {}
    else:
        cam_params, cond_cam_params, fv_exp_eye_params = [], [], []
    for prefix in tqdm(os.listdir(tracking_dir)):
        if not os.path.isdir(os.path.join(tracking_dir, prefix)):
            continue
        if test_data:
            cam_params[prefix], cond_cam_params[prefix], fv_exp_eye_params[prefix] = [], [], []
        for name in os.listdir(os.path.join(tracking_dir, prefix)):
            if not os.path.exists(os.path.join(tracking_dir, prefix, name, 'finish')):
                continue
            metaFace_extr = np.load(os.path.join(tracking_dir, prefix, name, 'metaFace_extr_smooth.npz' if use_smooth else 'metaFace_extr.npz'))
            camT_mesh2cam = metaFace_extr['transformation']
            camT_cam2mesh = np.linalg.inv(camT_mesh2cam)
            camT_cam2mesh = np.dot(fv2fl_T, camT_cam2mesh)

            angle = metaFace_extr['self_angle']
            trans = metaFace_extr['self_translation']

            coeff = np.load(os.path.join(tracking_dir, prefix, name, 'coeffs.npy'))
            exp_coeff = coeff[150:150+171]      # [id_dims:id_dims + exp_dims]
            eye_coeff = coeff[572+33:572+37]    # [all_dims + 33:all_dims + 37]

            if test_data:
                cam_params[prefix].append(["%s/%s" % (prefix, name + ".png"), np.concatenate([camT_cam2mesh.reshape(-1), cam_K.reshape(-1)]).tolist()])
                cond_cam_params[prefix].append(["%s/%s" % (prefix, name + ".png"), np.concatenate([angle, trans]).tolist()])
                fv_exp_eye_params[prefix].append(["%s/%s" % (prefix, name + ".png"), np.concatenate([exp_coeff, eye_coeff]).tolist()])
            else:
                cam_params.append(["%s/%s" % (prefix, name + ".png"), np.concatenate([camT_cam2mesh.reshape(-1), cam_K.reshape(-1)]).tolist()])
                cond_cam_params.append(["%s/%s" % (prefix, name + ".png"), np.concatenate([angle, trans]).tolist()])
                fv_exp_eye_params.append(["%s/%s" % (prefix, name + ".png"), np.concatenate([exp_coeff, eye_coeff]).tolist()])
                # fv_exp_eye_params.append(["%s/%s" % (prefix, "img%08d.png" % (int(name[3:]) + 1)), np.concatenate([exp_coeff, eye_coeff]).tolist()])

    return cam_params, cond_cam_params, fv_exp_eye_params
