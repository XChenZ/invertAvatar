import numpy as np


def save_obj(path, v, f=None, c=None):
    with open(path, 'w') as file:
        for i in range(len(v)):
            if c is not None:
                file.write('v %f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))
            else:
                file.write('v %f %f %f %d %d %d\n' % (v[i, 0], v[i, 1], v[i, 2], 1, 1, 1))

        file.write('\n')
        if f is not None:
            for i in range(len(f)):
                file.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))

    file.close()


def map_depth_to_3D(depth, mask, K_inv, T_inv=np.eye(4), mode='k4a'):
    colidx = np.arange(depth.shape[1])
    rowidx = np.arange(depth.shape[0])
    colidx_map, rowidx_map = np.meshgrid(colidx, rowidx)
    colidx_map = colidx_map.astype(np.float)  # + 0.5
    rowidx_map = rowidx_map.astype(np.float)  # + 0.5
    col_indices = colidx_map[mask > 0]
    # row_indices = (depth.shape[0] - rowidx_map)[mask > 0]  ####
    row_indices = rowidx_map[mask > 0]  # if mode == 'k4a' else (depth.shape[0] - rowidx_map)[mask > 0]
    homo_padding = np.ones((col_indices.shape[0], 1), dtype=np.float32)
    homo_indices = np.concatenate((col_indices[..., None], row_indices[..., None], homo_padding), axis=1)

    normalized_points = K_inv[None, ...] @ homo_indices[..., None]

    # z_values = (depth / 1000)[mask > 0]
    z_values = depth[mask > 0]

    valid_points = normalized_points.squeeze() * z_values[..., None]
    # print('cam_K', valid_points[:, 1].max() - valid_points[:, 1].min())
    # if mode == 'opengl':
    #     valid_points[:, 2] = -valid_points[:, 2]  ###
    R = T_inv[:3, :3]
    t = T_inv[:3, 3]
    points = R[None, ...] @ valid_points[..., None] + t[None, ..., None]
    points = points.squeeze()
    # print('cam_T', points[:, 1].max() - points[:, 1].min())
    return points