# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
# from metrics import metric_main
# from camera_utils import LookAtPoseSampler
# from training.crosssection_utils import sample_cross_section
from encoder_inversion.models.e4e import LatentCodesDiscriminator

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = list(label_groups.keys())
        rnd.shuffle(label_order)
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels, verts = zip(*[training_set[i][:3] for i in grid_indices])
    return (gw, gh), images, labels, verts

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    img = img[:, :3]
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def split_real(real, batch_gpu, batch_size, device):
    assert type(real) in [torch.Tensor, dict], type(real)
    if type(real) is torch.Tensor:
        all_real = (real.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu) if real.dtype is torch.uint8 else \
            real.to(device).to(torch.float32).split(batch_gpu)
    else:
        all_real = [{} for _ in range(batch_size // batch_gpu)]
        for key in real.keys():
            key_value = (real[key].to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu) if real[key].dtype is torch.uint8 else \
                real[key].to(device).to(torch.float32).split(batch_gpu)
            for i in range(len(key_value)): all_real[i][key] = key_value[i]
    return all_real

def split_gen(gen, batch_gpu, batch_size, device):
    assert type(gen) == list
    if type(gen[0]) == np.ndarray:
        all_gen = torch.from_numpy(np.stack(gen)).pin_memory().to(device)
        all_gen = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen.split(batch_size)]
    elif type(gen[0]) == dict:
        all_gen = [[{} for _ in range(batch_size//batch_gpu)] for _ in range(len(gen)//batch_size)]
        for key in gen[0].keys():
            key_value = torch.from_numpy(np.stack([sub[key] for sub in gen])).pin_memory().to(device)
            key_value_split = [phase_gen_c.split(batch_gpu) for phase_gen_c in key_value.split(batch_size)]
            for i in range(len(key_value_split)):
                for j in range(len(key_value_split[i])):
                    all_gen[i][j][key] = key_value_split[i][j]
    else: raise NotImplementedError
    return all_gen


#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    I_kwargs                = {},
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                  # Avoids errors with the augmentation pipe.
    use_D = loss_kwargs['loss_weight']['adv'] > 0.


    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')

    if False:  # G_kwargs.rendering_kwargs.get('gen_exp_cond'):
        g_c_dim = 6
        d_c_dim = training_set.label_dim
    else:
        g_c_dim = d_c_dim = training_set.label_dim
    G_kwargs['c_dim'] = g_c_dim
    G_kwargs['img_resolution'] = training_set.resolution
    G_kwargs['img_channels'] = training_set.num_channels
    G = dnnlib.util.construct_class_by_name(G_kwargs=G_kwargs, encoding_texture=I_kwargs['encoding_texture'], use_msfeat=I_kwargs['use_msfeat'],
                                            encoding_triplane=I_kwargs['encoding_triplane'], use_gru=I_kwargs['use_gru'],
                                            class_name=I_kwargs['class_name']).train().requires_grad_(False).to(device)
    if rank == 0:
        G.load_gen(I_kwargs['path_kwargs']['path_generator'])
        G.initialize_encoders(I_kwargs['path_kwargs']['path_irse50'], I_kwargs['path_kwargs']['path_triplanenet'])
        if resume_pkl is not None:
            print(f'Resuming from "{resume_pkl}"')
            with dnnlib.util.open_url(resume_pkl) as f:
                resume_data = legacy.load_network_pkl(f)
            misc.copy_params_and_buffers(resume_data['G'].encoder, G.encoder, require_all=True)
            out_str = "encoder"
            if hasattr(resume_data['G'].unet_encoder, 'texture_unet') and isinstance(G.unet_encoder.texture_unet, torch.nn.Module):
                misc.copy_params_and_buffers(resume_data['G'].unet_encoder.texture_unet, G.unet_encoder.texture_unet, require_all=True, print_=False)
                out_str += ", texture_unet"
            if hasattr(resume_data['G'].unet_encoder, 'triplane_unet') and isinstance(G.unet_encoder.triplane_unet, torch.nn.Module):
                misc.copy_params_and_buffers(resume_data['G'].unet_encoder.triplane_unet, G.unet_encoder.triplane_unet, require_all=True, print_=False)
                out_str += ", triplane_unet"
            # with dnnlib.util.open_url('training-runs/encoder_inversion/v20_128/sftTriNet_TriplaneTexNet_uvInp-000240.pkl') as f:
            #     resume_data_ = legacy.load_network_pkl(f)
            # if hasattr(resume_data_['G'].unet_encoder, 'triplane_unet') and isinstance(G.unet_encoder.triplane_unet, torch.nn.Module):
            #     misc.copy_params_and_buffers(resume_data_['G'].unet_encoder.triplane_unet, G.unet_encoder.triplane_unet, require_all=True, print_=True)
            #     out_str += ", triplane_unet"
            if hasattr(resume_data['G'].unet_encoder, 'cca') and isinstance(G.unet_encoder.cca, torch.nn.Module):
                misc.copy_params_and_buffers(resume_data['G'].unet_encoder.cca, G.unet_encoder.cca, require_all=True)
                out_str += ", cca"
            print('Load %s!' % (out_str))
            # del resume_data
        G.print_parameter_numbers()

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):  # For discriminator use
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, augment_pipe]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    
    phases = []
    # # Igen phase
    # # Igen_module = G
    # # for module_ in G_opt_kwargs['module'].split('.'):
    # #     Igen_module = getattr(Igen_module, module_)# 确保只有encoder的参数被优化
    # # Igen_modules = [Igen_module]
    # Igen_module, Igen_module_ = G.unet_encoder.triplane_unet, G.unet_encoder.texture_unet
    # Igen_modules = [Igen_module]
    # # Igen_modules = [Igen_module.up1, Igen_module.up2, Igen_module.up3, Igen_module.up4, Igen_module.head, Igen_module.final_head]# + \
    #                # [Igen_module.outconv0, Igen_module.outconv1, Igen_module.outconv2, Igen_module.outconv3] #,
    # # Igen_modules += [Igen_module_.fusion] + [getattr(Igen_module_, f'b{res}') for res in Igen_module_.block_resolutions]
    #
    # # Igen_modules = [Igen_module.fusion] + [getattr(Igen_module, f'b{res}') for res in Igen_module.block_resolutions]    ###fix_encoder
    # G_opt_kwargs.pop('module')
    # Igen_opt = dnnlib.util.construct_class_by_name(params=[{"params": gm.parameters()} for gm in Igen_modules], **G_opt_kwargs) # subclass of torch.optim.Optimizer
    # # phases += [dnnlib.EasyDict(name='Ireal', module=Igen_module, opt=Igen_opt, interval=1)]
    # # Igen_opt = torch.optim.Adam([{"params": m.parameters()} for m in Igen_modules], betas=G_opt_kwargs['betas'], eps=G_opt_kwargs['eps'])
    #
    # # Igen_opt = torch.optim.Adam(
    # #     [{"params": gm.parameters(), 'lr': G_opt_kwargs['lr']} for gm in Igen_modules] +
    # #     [{"params": gm.parameters(), 'lr': 0.005} for gm in [Igen_module.outconv0, Igen_module.outconv1, Igen_module.outconv2, Igen_module.outconv3]],
    # #     betas=G_opt_kwargs['betas'], eps=G_opt_kwargs['eps'])

    # phases += [dnnlib.EasyDict(name='Ireal', module=Igen_modules, opt=Igen_opt, interval=1)]
    # phases += [dnnlib.EasyDict(name='Igen', module=Igen_modules, opt=Igen_opt, interval=1)]

    G_opt_kwargs.pop('module')
    Igen_tri_modules = [G.unet_encoder.triplane_unet.up1, G.unet_encoder.triplane_unet.up2, G.unet_encoder.triplane_unet.up3,
                        G.unet_encoder.triplane_unet.up4, G.unet_encoder.triplane_unet.head, G.unet_encoder.triplane_unet.final_head] + \
                       [getattr(G.unet_encoder.triplane_unet, f'condition_scale{res}') for res in G.unet_encoder.triplane_unet.block_resolutions] + \
                       [getattr(G.unet_encoder.triplane_unet, f'condition_shift{res}') for res in G.unet_encoder.triplane_unet.block_resolutions]
    # Igen_tri_modules = [G.unet_encoder.triplane_unet]
    Igen_tri_opt = dnnlib.util.construct_class_by_name(params=[{"params": gm.parameters()} for gm in Igen_tri_modules], **G_opt_kwargs)
    G.unet_encoder.triplane_unet.eval()
    for pmodule in Igen_tri_modules: pmodule.train()
    # Igen_tex_modules = [G.unet_encoder.texture_unet.fusion] + \
    #                    [getattr(G.unet_encoder.texture_unet, f'b{res}') for res in G.unet_encoder.texture_unet.block_resolutions]
    Igen_tex_modules = [G.unet_encoder.texture_unet.up1, G.unet_encoder.texture_unet.up2, G.unet_encoder.texture_unet.up3,
                        G.unet_encoder.texture_unet.up4, G.unet_encoder.texture_unet.outconv0, G.unet_encoder.texture_unet.outconv1,
                        G.unet_encoder.texture_unet.outconv2, G.unet_encoder.texture_unet.outconv3]
    G.unet_encoder.texture_unet.eval()
    for pmodule in Igen_tex_modules: pmodule.train()
    # Igen_tex_modules = [G.unet_encoder.texture_unet]
    Igen_tex_opt = dnnlib.util.construct_class_by_name(params=[{"params": gm.parameters()} for gm in Igen_tex_modules], **G_opt_kwargs)
    phases += [dnnlib.EasyDict(name='Ireal', module=None, opt=None, interval=1)]
    phases += [dnnlib.EasyDict(name='Igen', module=None, opt=None, interval=1)]

    if use_D:
        D = dnnlib.util.construct_class_by_name(c_dim=d_c_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels * 2,
                                                **D_kwargs).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
        if rank == 0:
            resume_pkl = '/data/zhaoxiaochen/Code/animatable_eg3d/training-runs/encoder_inversion/VFHQ/gru/both_unetNetSFT/onlyDec_adv/00005-ffhq-images512x512-gpus5-batch5/network-snapshot-000040.pkl'
            with dnnlib.util.open_url(resume_pkl) as f: resume_data = legacy.load_network_pkl(f)
            if resume_pkl is not None and 'D' in list(resume_data.keys()):
                print('Load D from ' + resume_pkl)
                misc.copy_params_and_buffers(resume_data['D'], D, require_all=True, print_=False)
            else:
                print('Load D from ' + I_kwargs['path_kwargs']['path_generator'])
                with dnnlib.util.open_url(I_kwargs['path_kwargs']['path_generator']) as f:
                    resume_data = legacy.load_network_pkl(f)
                misc.copy_params_and_buffers(resume_data['D'], D, require_all=True, print_=False)

        for param in misc.params_and_buffers(D):
            if param.numel() > 0 and num_gpus > 1:
                torch.distributed.broadcast(param, src=0)

        D_opt = dnnlib.util.construct_class_by_name(params=D.parameters(), **D_opt_kwargs)
        phases += [dnnlib.EasyDict(name='Dmain', module=[D], opt=D_opt, interval=1)]
    else: D = None
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, frm_per_vid=min(6, training_set_kwargs['frm_per_vid']),
                                               **loss_kwargs) # subclass of training.loss.Loss

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    batch_num = batch_size // num_gpus

    ffhq_interval = 0
    if ffhq_interval > 0:   # 结合FFHQ数据集
        image_training_set_kwargs = training_set_kwargs.copy()
        image_training_set_kwargs['label_file'] = 'dataset.json'
        image_training_set_kwargs['class_name'] = 'training_avatar_texture.dataset_new.ImageFolderDataset'
        image_training_set_kwargs['path'] = '/data/zhaoxiaochen/Dataset/FFHQ/images512x512'
        image_training_set_kwargs['mesh_path'] = '/data/zhaoxiaochen/Dataset/FFHQ/orthRender256x256_face_eye'
        image_training_set_kwargs.pop('frm_per_vid')
        image_training_set = dnnlib.util.construct_class_by_name(**image_training_set_kwargs)
        image_training_set_sampler = misc.InfiniteSampler(dataset=image_training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
        image_training_set_iterator = iter(torch.utils.data.DataLoader(dataset=image_training_set, sampler=image_training_set_sampler, batch_size=batch_size // num_gpus, **data_loader_kwargs))
    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            all_real_packed = []
            for _ in range(len(phases) - int(use_D)):  # 一次为了Ireal，一次为了Igen&D
                if ffhq_interval > 0 and batch_idx % ffhq_interval == 0:
                    phase_real_img, phase_real_c, phase_real_v = next(image_training_set_iterator)
                    phase_real_vid, phase_real_vid_uv, phase_real_vid_v, phase_real_vid_c = \
                        phase_real_img['image'].unsqueeze(1), phase_real_img['uv'].unsqueeze(1), phase_real_v['uvcoords_image'].unsqueeze(1), phase_real_c.unsqueeze(1)
                    phase_real_vid_ws, phase_real_vid_gens, phase_real_vid_mm = [[None for _ in range(num_gpus)] for _ in range(3)]
                else:
                    sample_batch = next(training_set_iterator)
                    phase_real_vid, phase_real_vid_c, phase_real_vid_v, _, phase_real_vid_uv = sample_batch[:5]
                    phase_real_vid_ws = split_real(phase_real_vid_ws, batch_gpu, batch_size, device) if len(sample_batch)>5 else [None for _ in range(num_gpus)]
                    phase_real_vid_gens = split_real(sample_batch[6], batch_gpu, batch_size, device) if len(sample_batch)>6 else [None for _ in range(num_gpus)]
                    phase_real_vid_mm = sample_batch[7].to(device).split(batch_gpu) if len(sample_batch)>7 else [None for _ in range(num_gpus)]

                phase_real_vid_c = phase_real_vid_c.to(device).split(batch_gpu)
                phase_real_vid = split_real(phase_real_vid, batch_gpu, batch_size, device)
                phase_real_vid_v = split_real(phase_real_vid_v, batch_gpu, batch_size, device)
                phase_real_vid_uv = split_real(phase_real_vid_uv, batch_gpu, batch_size, device)

                all_real_packed.append([phase_real_vid, phase_real_vid_c, phase_real_vid_v, phase_real_vid_uv, phase_real_vid_ws, phase_real_vid_mm, phase_real_vid_gens])

        # Execute training phases.
        for phase in phases:
            gain = phase.interval
            if phase.name.startswith('I'):
                phase.module = Igen_tri_modules if batch_idx % 2 == 0 else Igen_tex_modules
                phase.opt = Igen_tri_opt if batch_idx % 2 == 0 else Igen_tex_opt
            if (batch_idx % D_reg_interval == 0) and phase.name.startswith('D'):
                phase.name = 'Dreg'
                gain = D_reg_interval
            if len(all_real_packed) == 1: phase_real_packed = all_real_packed[0]
            else: phase_real_packed = all_real_packed[0] if phase.name == 'Ireal' else all_real_packed[1]
            # if batch_idx % phase.interval != 0:
            #     continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            # phase.module.requires_grad_(True)
            for pmodule in phase.module: pmodule.requires_grad_(True)
            for real_vid, real_vid_c, real_vid_v, real_vid_uv, real_vid_ws, real_vid_mm, real_vid_gens in zip(*phase_real_packed):
                if False:
                    loss.accumulate_gradients(phase.name, real_vid, real_vid_c, real_vid_v, real_vid_uv, None, real_vid_mm, real_vid_gens, gain)
                else:   # multi_T
                    if batch_idx % 5 < 1:
                        loss.accumulate_gradients(phase.name, real_vid[:, :6], real_vid_c[:, :6], real_vid_v[:, :6], real_vid_uv[:, :6], None, real_vid_mm, real_vid_gens, gain)
                    elif batch_idx % 5 < 3:
                        loss.accumulate_gradients(phase.name, real_vid[:, :12], real_vid_c[:, :12], real_vid_v[:, :12], real_vid_uv[:, :12], None, real_vid_mm, real_vid_gens, gain)
                    else:
                        loss.accumulate_gradients(phase.name, real_vid[:, :18], real_vid_c[:, :18], real_vid_v[:, :18], real_vid_uv[:, :18], None, real_vid_mm, real_vid_gens, gain)
            # phase.module.requires_grad_(False)
            for pmodule in phase.module: pmodule.requires_grad_(False)
            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                if isinstance(phase.module, list):
                    params = []
                    for pm in phase.module:
                        params.extend([param for param in pm.parameters() if param.numel() > 0 and param.grad is not None])
                else:
                    params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))
            # loss.run_G_seq(phase_real_vid_c[0], phase_real_vid_v[0], phase_real_vid_uv[0], phase_real_vid_ws[0], phase_real_vid[0],
            #                phase_real_vid_gens[0], flag=True)

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')
        # Save image snapshot.
        if (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            grid_c_num = 1
            if True:
                bs, Ts = phase_real_vid[0].shape[:2]
                # G.eval()    # 对于VFHQ上的batch_num=1的训练，似乎用train()模式更符合训练时的情况
                with torch.no_grad():
                    if True:
                        # if loss_kwargs['multiT_training']:
                        #     out_image = loss.run_G_seq_multiT(phase_real_vid_c[0], phase_real_vid_v[0], phase_real_vid_uv[0], phase_real_vid_ws[0],
                        #                                phase_real_vid[0][:, :, :3], phase_real_vid_gens[0])
                        # else:
                        out_image = loss.run_G(phase_real_vid_c[0], phase_real_vid_v[0], phase_real_vid_uv[0], phase_real_vid_ws[0],
                                                       phase_real_vid[0][:, :, :3], phase_real_vid_gens[0])
                        images_real = [phase_real_vid[0][:, :, :3].flatten(0, 1), out_image['e4e_image'].flatten(0, 1), out_image['image'].flatten(0, 1)]
                        # if phase_real_vid_gens[0] is not None: images_real.insert(1, phase_real_vid_gens[0].flatten(0, 1))
                        if phase_real_vid[0].shape[2] > 3: images_real.insert(1, (phase_real_vid[0][:, :, 3:].flatten(0, 1).expand_as(images_real[0]) * 2) - 1)
                        save_image_grid(torch.cat(images_real, dim=0).cpu().numpy(), os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_{rank}.png'),
                                        drange=[-1, 1], grid_size=(bs * Ts, len(images_real)))
                    else:
                        texture_feats = G.generator.texture_backbone.synthesis(phase_real_vid_ws[0], cond_list=None, return_list=True, update_emas=False,
                                                                                    noise_mode='const')
                        static_feats = G.generator.backbone.synthesis(phase_real_vid_ws[0], cond_list=None, return_list=True, update_emas=False, noise_mode='const')
                        # updated_texture_feats, _ = G.AR_forward_step(phase_real_vid_ws[0], phase_real_vid_c[0][:, 0], {'uvcoords_image': phase_real_vid_v[0][:, 0]},
                        #                                                   texture_feats, static_feats, uv_input=phase_real_vid_uv[0][:, 0])

                        texture_offsets, _ = G.unet_encoder.texture_unet(phase_real_vid_uv[0], return_list=True)
                        updated_texture_feats = [(feat + offset).unsqueeze(1).expand(-1, Ts, -1, -1, -1).flatten(0, 1) for feat, offset in zip(texture_feats, texture_offsets)]

                        out = G.generator.synthesis_withTexture(phase_real_vid_ws[0].unsqueeze(1).expand(-1, Ts, -1, -1).flatten(0, 1), updated_texture_feats, phase_real_vid_c[0].flatten(0, 1),
                                                                {'uvcoords_image': phase_real_vid_v[0].flatten(0, 1)},
                                                                static_feats=[feat.unsqueeze(1).expand(-1, Ts, -1, -1, -1).flatten(0, 1) for feat in static_feats],
                                                                noise_mode='const')
                        out['x_input'] = torch.clamp(phase_real_vid_uv[0].flatten(0, 1), min=-1, max=1)
                        images_raw = out['image_raw'].cpu().numpy()
                        images_depth = -out['image_depth'].cpu().numpy()
                        images_real = torch.cat([phase_real_vid[0][:, :, :3].flatten(0, 1), out['image']], dim=0).cpu().numpy()
                        vis_input = torch.cat([out['x_input'][:, :3], out['x_input'][:, 3:6], out['x_input'][:, 6:].expand(-1, 3, -1, -1)], dim=0).cpu().numpy()
                        save_image_grid(vis_input, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_visInput_{rank}.png'), drange=[-1, 1], grid_size=(bs*Ts, 3))
                        save_image_grid(images_real, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_{rank}.png'), drange=[-1, 1], grid_size=(bs*Ts, grid_c_num + 1))
                        save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw_{rank}.png'), drange=[-1, 1], grid_size=(bs*Ts, grid_c_num))
                        save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth_{rank}.png'), drange=[images_depth.min(), images_depth.max()], grid_size=(bs*Ts, grid_c_num))
                # G.train()

            else:
                bs = len(phase_real_img[0])
                iter_num=3
                with torch.no_grad():
                    out = G.generator.synthesis(phase_real_w[0], phase_real_c[0], phase_real_v[0])
                    out['w'] = phase_real_w[0]
                    images_real = torch.cat([phase_real_img[0][:, :3], out['image'].clone()], dim=0)
                    for i in range(iter_num):
                        inp_img = torch.nn.functional.interpolate(torch.cat([phase_real_img[0][:, :3], out['image']], dim=1), size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
                        init_ws = out['w'].clone()
                        out = G.restyle_forward(inp_img, real_c, real_v, init_ws)
                        images_real = torch.cat([images_real, out['image'].clone()], dim=0)

                images_real = images_real.cpu().numpy()
                save_image_grid(images_real, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}.png'), drange=[-1, 1], grid_size=(bs, grid_c_num + 1 + iter_num))

            #--------------------
            # # Log forward-conditioned images

            # forward_cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
            # intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
            # forward_label = torch.cat([forward_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            # grid_ws = [G_ema.mapping(z, forward_label.expand(z.shape[0], -1)) for z, c in zip(grid_z, grid_c)]
            # out = [G_ema.synthesis(ws, c=c, noise_mode='const') for ws, c in zip(grid_ws, grid_c)]

            # images = torch.cat([o['image'].cpu() for o in out]).numpy()
            # images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
            # images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
            # save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_f.png'), drange=[-1,1], grid_size=grid_size)
            # save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw_f.png'), drange=[-1,1], grid_size=grid_size)
            # save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth_f.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)

            #--------------------
            # # Log Cross sections

            # grid_ws = [G_ema.mapping(z, c.expand(z.shape[0], -1)) for z, c in zip(grid_z, grid_c)]
            # out = [sample_cross_section(G_ema, ws, w=G.rendering_kwargs['box_warp']) for ws, c in zip(grid_ws, grid_c)]
            # crossections = torch.cat([o.cpu() for o in out]).numpy()
            # save_image_grid(crossections, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_crossection.png'), drange=[-50,100], grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            save_modules = [('G', G)]
            if use_D: save_modules.append(('D', D))
            for name, module in save_modules:  # , ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema|mean|var)')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics. TODO
        # if (snapshot_data is not None) and (len(metrics) > 0):
        #     if rank == 0:
        #         print(run_dir)
        #         print('Evaluating metrics...')
        #     for metric in metrics:
        #         result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
        #             dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
        #         if rank == 0:
        #             metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
        #         stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
