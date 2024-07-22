# python preprocess_person_video_dataset.py --vid data_preprocess/Obama.mp4 --savedir data/tgt_data
import glob
import os
import argparse
import torch
from tqdm import tqdm
import json
import cv2

fv_preprocess_path = '/media/zxc/10T/Code/Havatar_clean/data_preprocessing'
current_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('--vid', type=str, required=True)
parser.add_argument('--savedir', type=str, required=True)
parser.add_argument('--valid_video_json', type=str, default=None)
parser.add_argument('--test_data', action='store_true', default=False)
args = parser.parse_args()

correct_path = lambda path:path[:-1] if path.endswith('/') else path

args.savedir = correct_path(args.savedir)

raw_detection_dir = os.path.join(args.savedir, "raw_detections")
save_dir = os.path.join(args.savedir, 'dataset', "images512x512")
save_tracking_dir = os.path.join(args.savedir, 'crop_fv_tracking')
disable_warping = True
smooth_cropping_mode = 2
padding_mode = 'zero_padding' #'reflect_padding' #'blur_padding' 'zero_padding'


def extract_frames(video_path, dst_save_dir, skip=1, center_crop=True, res=512):
    videoCapture = cv2.VideoCapture(video_path)
    os.makedirs(dst_save_dir, exist_ok=True)
    # 获得码率及尺寸
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if center_crop:
        length = min(size) // 2
        top, bottom, left, right = max(0, size[1]//2 - length), min(size[1], size[1]//2 + length), max(0, size[0]//2 - length), min(size[0], size[0]//2 + length)
    else:
        length = max(size) // 2
        top, bottom, left, right = max(0, length - size[1] // 2), max(0, length - size[1] // 2), max(0, length - size[0] // 2), max(0, length - size[0] // 2)


    count = -1
    while True:
        flag, frame = videoCapture.read()
        count += 1
        if not flag:
            break
        if skip > 1 and not (count % skip == 0):
            continue

        if center_crop:
            crop_frame = frame[top: bottom, left: right]
        else:
            crop_frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        if not res == crop_frame.shape[0]:
           crop_frame = cv2.resize(crop_frame, dsize=(res, res), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(dst_save_dir, str(count) + '.png'), crop_frame)

    videoCapture.release()


def generate_valid_videos_idx_default(video_path, save_dir, skip=1):
    frames_dir = os.path.join(save_dir, os.path.basename(video_path).split('.')[0])
    extract_frames(video_path, frames_dir)
    save_path = os.path.join(os.path.dirname(frames_dir), 'valid_videos.json')
    valid_videos = []
    img_nums = [0]
    count = 0
    for video_name in tqdm(os.listdir(save_dir)):
        video_dir = os.path.join(save_dir, video_name)
        if not os.path.isdir(video_dir): continue
        # valid_videos.append([video_name, len(os.listdir(video_dir))])
        img_names = [x for x in os.listdir(os.path.join(save_dir, video_name)) if x.endswith(".jpg") or x.endswith(".png")]
        img_names.sort(key=lambda x: int(x.split('.')[0]))
        valid_videos.append([video_name, img_names[::skip]])
        count += len(valid_videos[-1][1])
        img_nums.append(count)

    with open(save_path, 'w') as f:
        f.write(json.dumps(valid_videos, indent=4))

    return save_path


def remove_labels(tracking_dir, labels_path):
    from tqdm import tqdm
    import json
    no_face_log = []
    for f in os.listdir(tracking_dir):
        if not f.endswith("_total_no_face_log.json"): continue
        no_face_log.extend(json.loads(open(os.path.join(tracking_dir, f)).read()))

    labels = json.loads(open(labels_path).read())['labels']
    names = [label[0] for label in labels]
    skip_vid = []
    for log in no_face_log:
        if '%s/%s' % (log[1].split('/')[-2], log[1].split('/')[-1]) in names:
            skip_vid.append(log[1].split('/')[-2])
    clean_labels = []
    for label in tqdm(labels):
        if (label[0].split('/')[0] not in skip_vid):
            clean_labels.append(label)
    print(len(labels), len(clean_labels))
    with open(labels_path, "w") as f:
        json.dump({'labels': clean_labels}, f, indent=4)


if True:
    args.indir = os.path.join(args.savedir, 'raw_frames')
    print('Extracting frames')
    args.valid_video_json = generate_valid_videos_idx_default(args.vid, args.indir, skip=1)

    # run mtcnn
    command = "python batch_mtcnn_video.py  --in_root=%s --save_dir=%s --skip" % (args.indir, raw_detection_dir)
    if args.valid_video_json is not None:
        command += " --valid_video_json=%s" % args.valid_video_json
    print(command)
    os.system(command)

    # Quad transform the image and Crop the image
    command = "python align_in_the_wild.py --source=%s --lm_source=%s --dest=%s --save_realign --threads=16 --skip --padding_mode %s" % \
              (args.indir, raw_detection_dir, save_dir, padding_mode)
    if args.valid_video_json is not None:
        command += " --valid_video_json=%s" % args.valid_video_json
    if disable_warping:
        command += " --disable_warping"
    if smooth_cropping_mode > 0:
        command += " --apply_GF=%f" % ([1.5, 3.0][smooth_cropping_mode - 1])
    print(command)
    os.system(command)

# run Faceverse Tracking
if True:
    os.chdir(fv_preprocess_path)
    command = "conda run -n torch181 "
    command += "python fit_videos_mp.py --base_dir %s --save_dir %s" % (save_dir, save_tracking_dir)
    print(command)
    os.system(command)

if True:
    os.chdir(current_path)
    command = "python make_dataset_pipe.py --tracking_dir=%s --root_dir=%s" % (save_tracking_dir, args.savedir)# + ("--test_data" if args.test_data else "")
    print(command)
    os.system(command)

    # remove_labels(save_tracking_dir, os.path.join(savedir, 'dataset_realcam.json'))