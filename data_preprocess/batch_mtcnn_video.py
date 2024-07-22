# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import argparse
import cv2
import os
from datetime import datetime
import numpy as np
from mtcnn import MTCNN
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import json
detector = MTCNN()

# see how to visualize the bounding box and the landmarks at : https://github.com/ipazc/mtcnn/blob/master/example.py 

parser = argparse.ArgumentParser()
parser.add_argument('--in_root', type=str, default="", help='process folder')
parser.add_argument('--save_dir', type=str, default="", help='process folder')
parser.add_argument('--enable_no_face_detection', action='store_true')
parser.add_argument('--skip', action='store_true')
parser.add_argument('--valid_video_json', type=str, default=None)
args = parser.parse_args()
in_root = args.in_root

# out_detection = os.path.join(args.save_dir, "video_raw_detections" + ("_smooth" if args.apply_GF else ""))
out_detection = args.save_dir

os.makedirs(out_detection, exist_ok=True)

valid_idx = json.loads(open(args.valid_video_json).read())
no_face_log = []
for vidx, (video_name, imgs) in enumerate(valid_idx):
    # imgs = sorted([x for x in os.listdir(os.path.join(in_root, video_name)) if x.endswith(".jpg") or x.endswith(".png")])
    kps_sequence = []
    # print('Processing ' + video_name)
    if args.skip and os.path.exists(os.path.join(out_detection, video_name + '.json')):
        continue
    bar = tqdm(imgs)
    save_kps = dict()
    for img in bar:
        bar.set_description('                                                     %d/%d: %s' % (vidx, len(valid_idx), video_name))
        src = os.path.join(in_root, video_name, img)
        image = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
        result = detector.detect_faces(image)

        if not len(result)>0:
            no_face_log.append([video_name, img])
            if args.enable_no_face_detection: continue
            else: break
        index = 0
        if len(result)>1: # if multiple faces, take the biggest face
            size = -100000
            for r in range(len(result)):
                size_ = result[r]["box"][2] + result[r]["box"][3]
                if size < size_:
                    size = size_
                    index = r

        bounding_box = result[index]['box']
        keypoints = result[index]['keypoints']
        kps = [[float(keypoints['left_eye'][0]), float(keypoints['left_eye'][1])],
               [float(keypoints['right_eye'][0]), float(keypoints['right_eye'][1])],
               [float(keypoints['nose'][0]), float(keypoints['nose'][1])],
               [float(keypoints['mouth_left'][0]), float(keypoints['mouth_left'][1])],
               [float(keypoints['mouth_right'][0]), float(keypoints['mouth_right'][1])]]
        kps_sequence.append(kps)
        save_kps[img] = kps

    if (not args.enable_no_face_detection) and len(kps_sequence) < len(imgs):
        continue

    # kps_sequence = np.asarray(kps_sequence, dtype=np.float32)
    # save_kps = dict()
    # for i, img in enumerate(imgs):
    #     save_kps[img] = kps_sequence[i].tolist()
    print(os.path.join(out_detection, video_name+'.json'))
    with open(os.path.join(out_detection, video_name+'.json'), 'w') as f:
        f.write(json.dumps(save_kps, indent=4))

if len(no_face_log) > 0:
    jstr = json.dumps(no_face_log, indent=4)
    with open(os.path.join(out_detection, str(datetime.now()) + '_total_no_face_log.json'), 'w') as f:
        f.write(jstr)