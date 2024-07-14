import pickle
import torch
from inversion.configs import paths_config, global_config
from torch_utils import misc
import numpy as np
import legacy
import dnnlib

def create_code_snapshot(root, dst_path, extensions=(".py", ".h", ".cpp", ".cu", ".cc", ".cuh", ".json", ".sh", ".bat"), exclude=()):
    """Creates tarball with the source code"""
    import tarfile
    from pathlib import Path
    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in Path(root).rglob("*"):
            if '.git' in path.parts:
                continue
            exclude_flag = False
            if len(exclude) > 0:
                for k in exclude:
                    if k in path.parts:
                        exclude_flag = True
            if exclude_flag:
                continue
            if path.suffix.lower() in extensions:
                tar.add(path.as_posix(), arcname=path.relative_to(
                    root).as_posix(), recursive=True)

def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_tuned_G(new_G_path, finetune=False, Gen_func=None):
    # new_G_path = f'{paths_config.checkpoints_dir}/{run_id}/model_{run_id}.pt' if not finetune else f'{paths_config.checkpoints_dir}/{run_id}/finetuned_model_{run_id}.pt'
    with open(new_G_path, 'rb') as f:
        G = torch.load(f).to(global_config.device).eval()
    if Gen_func is not None:
        G_new = reload_module(G, Gen_func)
    else:
        G_new = G
    G_new = G_new.float()
    toogle_grad(G_new, False)
    return G_new


def load_D():
    with open(paths_config.ide3d_ffhq, 'rb') as f:
    # with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        D = pickle.load(f)['D'].to(global_config.device).eval()
        D = D.float()
    return D


def load_old_G(Gen_func=None):
    with open(paths_config.ide3d_ffhq, 'rb') as f:
    # # with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].to(global_config.device).eval()
        # old_G = old_G.float()
    # with open('/media/zxc/10T/Code/animatable_eg3d/pretrained_models/v10/network-snapshot-001362.pkl', 'rb') as f:
    #     old_G = torch.load(f).to(global_config.device).eval()
    if Gen_func is not None:
        old_G = reload_module(old_G, Gen_func)
    old_G = old_G.float()
    return old_G

def load_old_D(Net_func=None):
    with open(paths_config.ide3d_ffhq, 'rb') as f:
        old_D = pickle.load(f)['D'].to(global_config.device).eval()
    if Net_func is not None:
        old_D = reload_module(old_D, Net_func, gen=False)
    return old_D.float()

def reload_module(G, Gen_func, gen=True):
    G_new = Gen_func(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(global_config.device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)
    if gen:
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
    return G_new


def load_G(network_pkl, Gen_func=None):
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(global_config.device)

    if Gen_func is not None:
        return reload_module(G, Gen_func)


def create_samples(N=512, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size


def sample_generator_ide3d(generator, ws, c, v, max_batch=100000, voxel_resolution=512, voxel_origin=[0, 0, 0], cube_length=0.3, psi=0.7, **kwargs):
    head = 0
    samples, voxel_origin, voxel_size = create_samples(voxel_resolution, voxel_origin, cube_length)
    samples = 0.9 * (samples).to(c.device)
    # debug:
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=c.device)

    with torch.no_grad():
        # mapping latent codes into W space
        # TODO: compute nerf input
        voxel_block_ws = []
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, generator.synthesis.num_ws, generator.synthesis.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in generator.synthesis.voxel_block_resolutions:
                block = getattr(generator.synthesis, f'vb{res}')
                voxel_block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv
            for res in generator.synthesis.block_resolutions:
                block = getattr(generator.synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x_v = img_v = seg_v = None
        for res, cur_ws in zip(generator.synthesis.voxel_block_resolutions, voxel_block_ws):
            block = getattr(generator.synthesis, f'vb{res}')
            x_v, img_v, seg_v = block(x_v, img_v, cur_ws, condition_img=seg_v)

        render_kwargs = {}
        render_kwargs["img_size"] = generator.synthesis.render_size
        render_kwargs["nerf_noise"] = 0
        render_kwargs["ray_start"] = 2.25
        render_kwargs["ray_end"] = 3.3
        render_kwargs["fov"] = 18
        render_kwargs.update(kwargs)

        P = c[:, :16].reshape(-1, 4, 4)
        P = P.detach()
        render_kwargs["camera"] = P

        # Sequentially evaluate siren with max_batch_size to avoid OOM
        while head < samples.shape[1]:
            tail = head + max_batch
            # coarse_output = generator.synthesis.nerf_forward(samples[:, head:tail], w[:, :14]).reshape(samples.shape[0], -1, 33) # 33
            coarse_output = generator.synthesis.renderer.sample_voxel(img_v, seg_v, samples[:, head:tail]).reshape(samples.size(0), -1, 52)
            sigmas[:, head:head + max_batch] = coarse_output[:, :, -1:]
            head += max_batch
    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
    return sigmas