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
from metrics import metric_main
from camera_utils import LookAtPoseSampler
from training.crosssection_utils import sample_cross_section

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

def split_gen(gen, batch_gpu, batch_size, device):
    assert type(gen) == list
    if type(gen[0]) == np.ndarray:
        all_gen = torch.from_numpy(np.stack(gen)).pin_memory().to(device).float()
        all_gen = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen.split(batch_size)]
    elif type(gen[0]) == dict:
        all_gen = [[{} for _ in range(batch_size//batch_gpu)] for _ in range(len(gen)//batch_size)]
        for key in gen[0].keys():
            key_value = torch.from_numpy(np.stack([sub[key] for sub in gen])).pin_memory().to(device).float()
            key_value_split = [phase_gen_c.split(batch_gpu) for phase_gen_c in key_value.split(batch_size)]
            for i in range(len(key_value_split)):
                for j in range(len(key_value_split[i])):
                    all_gen[i][j][key] = key_value_split[i][j]
    else: raise NotImplementedError
    return all_gen

def split_gen_new(gen, batch_gpu, batch_size, device):
    if type(gen) == torch.Tensor:
        all_gen = gen.view((gen.shape[0]*gen.shape[1], ) + (gen.shape[2:])).pin_memory().to(device)
        all_gen = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen.split(batch_size)]
    elif type(gen[0]) == dict:
        all_gen = [[{} for _ in range(batch_size//batch_gpu)] for _ in range(int(len(gen)*list(gen[0].values())[0].shape[0]//batch_size))]
        for key in gen[0].keys():
            key_value = torch.cat([sub[key] for sub in gen], dim=0).pin_memory().to(device)
            key_value_split = [phase_gen_c.split(batch_gpu) for phase_gen_c in key_value.split(batch_size)]
            for i in range(len(key_value_split)):
                for j in range(len(key_value_split[i])):
                    all_gen[i][j][key] = key_value_split[i][j]
    else: raise NotImplementedError
    return all_gen

def resume_model(G, G_ema, D, resume_pkl):
    print(f'Resuming from "{resume_pkl}"')
    with dnnlib.util.open_url(resume_pkl) as f:
        resume_data = legacy.load_network_pkl(f)
    for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
        misc.copy_params_and_buffers(resume_data[name], module, require_all=False, print_=False)

    # debug: copy backbone parameters to texture backbone
    if hasattr(G, 'texture_backbone'):
        if hasattr(resume_data['G'], 'texture_backbone'):
            misc.copy_params_and_buffers(resume_data['G'].texture_backbone, G.texture_backbone, require_all=True)
            misc.copy_params_and_buffers(resume_data['G_ema'].texture_backbone, G_ema.texture_backbone, require_all=True)
        else:
            print('texture_backbone')
            misc.copy_params_and_buffers(resume_data['G'].backbone, G.texture_backbone, require_all=False)
            misc.copy_params_and_buffers(resume_data['G_ema'].backbone, G_ema.texture_backbone, require_all=False)

    if hasattr(G, 'face_backbone'):
        if hasattr(resume_data['G'], 'face_backbone'):
            misc.copy_params_and_buffers(resume_data['G'].face_backbone, G.face_backbone, require_all=True)
            misc.copy_params_and_buffers(resume_data['G_ema'].face_backbone, G_ema.face_backbone, require_all=True)
        else:
            print('face_backbone')
            misc.copy_params_and_buffers(resume_data['G'].backbone, G.face_backbone, require_all=False)
            misc.copy_params_and_buffers(resume_data['G_ema'].backbone, G_ema.face_backbone, require_all=False)
    if hasattr(G, 'bg_backbone'):
        if hasattr(resume_data['G'], 'bg_backbone'):
            misc.copy_params_and_buffers(resume_data['G'].bg_backbone, G.bg_backbone, require_all=True)
            misc.copy_params_and_buffers(resume_data['G_ema'].bg_backbone, G_ema.bg_backbone, require_all=True)
        else:
            print('bg_backbone')
            misc.copy_params_and_buffers(resume_data['G'].backbone, G.bg_backbone, require_all=False)
            misc.copy_params_and_buffers(resume_data['G_ema'].backbone, G_ema.bg_backbone, require_all=False)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
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

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')

    g_c_dim = d_c_dim = training_set.label_dim

    D_img_channel = training_set.num_channels * 3 if G_kwargs.rendering_kwargs.get('gen_lms_cond') else training_set.num_channels * 2
    if G_kwargs.rendering_kwargs.get('gen_mask_cond'): D_img_channel += 1
    # common_kwargs = dict(img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(c_dim=g_c_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels, **G_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(c_dim=d_c_dim, img_resolution=training_set.resolution, img_channels=D_img_channel, **D_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if rank == 0:
        resume_model(G, G_ema, D, 'pretrained_models/eg3d/ffhqrebalanced512-128.pkl')
        if (resume_pkl is not None):
            resume_model(G, G_ema, D, resume_pkl)


    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None): # For discriminator use
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    # training_set.random_sample_num = len(phases)

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    grid_v = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels, verts = setup_snapshot_image_grid(training_set=training_set)
        images, labels = np.stack(images), np.stack(labels)

        save_image_grid(np.stack(images), os.path.join(run_dir, 'reals.png'), drange=[0, 255], grid_size=grid_size)
        grid_c = torch.from_numpy(np.stack(labels)).to(device).split(batch_gpu)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        if type(verts[0]) == dict:
            grid_v = [{} for _ in range(len(grid_c))]
            for k in verts[0].keys():
                tuple_ = torch.from_numpy(np.stack([vert[k] for vert in verts])).to(device).split(batch_gpu)
                for i in range(len(tuple_)):
                    grid_v[i][k] = tuple_[i]
            grid_v = tuple(grid_v)
        else:
            grid_v = torch.from_numpy(np.stack(verts)).to(device).split(batch_gpu)
        vis_uv = torch.cat([G_ema.visualize_mesh_condition(v).cpu() for v in grid_v]).numpy()
        save_image_grid(vis_uv, os.path.join(run_dir, 'reals_visUV.png'), drange=[-1, 1], grid_size=grid_size)

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
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(
        torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size // num_gpus, **data_loader_kwargs))
    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c, phase_real_v = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_num, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_num)]
            gen_cond_sample_idx = [np.random.randint(len(training_set)) for _ in range(len(phases) * batch_num)]
            all_gen_c = [training_set.get_label(i) for i in gen_cond_sample_idx]
            all_gen_v = [training_set.get_vert(i) for i in gen_cond_sample_idx]
            all_gen_c = split_gen(all_gen_c, batch_gpu, batch_num, device)
            all_gen_v = split_gen(all_gen_v, batch_gpu, batch_num, device)
            if G_kwargs.rendering_kwargs.get('gen_lms_cond'):
                all_gt_img = torch.from_numpy(np.stack([training_set.get_CondImg(i) for i in gen_cond_sample_idx])).pin_memory().to(device).to(torch.float32) / 127.5 - 1
                all_gt_img = [phase_gen_gt_img.split(batch_gpu) for phase_gen_gt_img in all_gt_img.split(batch_num)]
            else:
                all_gt_img = all_gen_z

        # Execute training phases.
        for phase, phase_gt_img, phase_gen_z, phase_gen_c, phase_gen_v in zip(phases, all_gt_img, all_gen_z, all_gen_c, all_gen_v):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for gt_img, real_img, real_c, gen_z, gen_c, gen_v in zip(phase_gt_img, phase_real_img, phase_real_c, phase_gen_z, phase_gen_c, phase_gen_v):
                if not G_kwargs.rendering_kwargs.get('gen_lms_cond'): gt_img = None
                loss.accumulate_gradients(phase=phase.name, gt_img=gt_img, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c,
                                              gen_v=gen_v, gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
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
            # print(device, phase.name, time.time() - data_t0)

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
            G_ema.neural_rendering_resolution = G.neural_rendering_resolution
            G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Release reserved cuda memory
        if loss.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (loss.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(loss.neural_rendering_resolution_initial * (1 - alpha) + loss.neural_rendering_resolution_final * alpha))
            if neural_rendering_resolution > G.neural_rendering_resolution:
                torch.cuda.empty_cache()
                if rank == 0: print(f'{(cur_nimg / 1e3):<8.1f}kimg Empty cache!')

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
        fields += [f"raw_res {G_ema.neural_rendering_resolution}"]
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            with torch.no_grad():
                out = [G_ema(z=z, c=c, v=v, noise_mode='const') for z, c, v in zip(grid_z, grid_c, grid_v)]
                images = torch.cat([o['image'].cpu() for o in out]).numpy()

                images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
                images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
                save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
                save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw.png'), drange=[-1,1], grid_size=grid_size)
                save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)
                if 'mask_raw' in out[0].keys():
                    mask_raw = torch.cat([o['mask_raw'].cpu() for o in out]).numpy()
                    save_image_grid(mask_raw, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}_mask.png'), drange=[0, 1], grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        training_set_kwargs_tmp = training_set_kwargs.copy()
        training_set_kwargs_tmp['load_obj'] = False
        if (snapshot_data is not None) and (len(metrics) > 0) and (cur_tick > 0):
            if rank == 0:
                print(run_dir)
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                    dataset_kwargs=training_set_kwargs_tmp, num_gpus=num_gpus, rank=rank, device=device, cond_vert=True)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
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
