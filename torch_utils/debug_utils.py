import numpy as np
import torch


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

def save_obj_torch(path, v, f=None, c=None):
    v_ = v.cpu().numpy().astype(np.float32)
    f_ = None if f is None else f.cpu().numpy().astype(np.int32)
    c_ = None if c is None else c.cpu().numpy().astype(np.float32)
    save_obj(path, v_, f_, c_)