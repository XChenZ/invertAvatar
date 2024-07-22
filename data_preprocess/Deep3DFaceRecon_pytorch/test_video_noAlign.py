"""This script is the test script for Deep3DFaceRecon_pytorch
"""
import copy
import os
from options.test_options import TestOptions
import multiprocessing
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch 
import json

def get_data_path(root='examples', lm_dir=None):
    # im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if (i.endswith('png') or i.endswith('jpg')) and (int(i.split('.')[0][3:]) % 2 == 0)] # FFHQ x_flip
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    if lm_dir is None:
        lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1], ''), 'detections', i.split(os.path.sep)[-1]) for i in lm_path]
    else:
        lm_path = [os.path.join(lm_dir, i.split(os.path.sep)[-1]) for i in lm_path]
    return im_path, lm_path

def read_data(im_path, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    im = im.resize((224, 224), resample=Image.BICUBIC)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return im

def main(opt):
    video_names = json.loads(open(opt.valid_video_json).read())
    video_root = opt.img_folder
    lm_root = opt.lm_folder
    save_root = opt.save_folder

    device = torch.device(0)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    for idx, video_name in enumerate(video_names):
        video_dir = os.path.join(video_root, video_name)
        dst_save_dir = os.path.join(save_root, video_name)
        lm_dir = None#os.path.join(lm_root, video_name)

        im_path, lm_path = get_data_path(video_dir, lm_dir)
        print('(%d/%d) Processing %s' % (idx, len(video_names), video_name))
        for i in range(len(im_path)):
            img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
            # if not os.path.isfile(lm_path[i]):
            #     continue
            im_tensor = read_data(im_path[i])
            data = {
                'imgs': im_tensor,
            }
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            visualizer.display_current_results(visuals, 0, opt.epoch, dataset=video_dir.split(os.path.sep)[-1],
                save_results=True, count=i, name=img_name, add_image=False, save_dir=dst_save_dir)
            model.save_mesh(os.path.join(dst_save_dir, img_name + '.obj'))  # save reconstruction meshes
            model.save_coeff(os.path.join(dst_save_dir, img_name + '.mat'))  # save predicted coefficients

            # model.save_mesh(os.path.join(visualizer.img_dir, 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.obj')) # save reconstruction meshes
            # model.save_coeff(os.path.join(visualizer.img_dir, 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.mat')) # save predicted coefficients

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    main(opt)
    # video_root = opt.img_folder
    # lm_root = opt.lm_folder
    # save_root = opt.save_folder
    # num_thread = 10
    #
    # file_list = []
    # video_names = []
    # for video_name in os.listdir(lm_root):
    #     video_dir = os.path.join(video_root, video_name)
    #     dst_save_dir = os.path.join(save_root, video_name)
    #     lm_dir = os.path.join(lm_root, video_name)
    #     if not os.path.exists(lm_dir): continue
    #     video_names.append(video_name)
    #
    # p = multiprocessing.Pool(num_thread)
    # all_video_names = [video_names[i * (len(video_names) // num_thread): (i + 1) * (len(video_names) // num_thread)] for i in range(num_thread)] + \
    #            [video_names[num_thread * len(video_names) // num_thread:]]
    # all_list = [{'opt': copy.deepcopy(opt), 'video_names': sub_video_names} for sub_video_names in all_video_names]
    #
    # p.map(process, all_list)
    # p.close()
    # p.join()

    
