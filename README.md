## InvertAvatar: Incremental GAN Inversion for Generalized Head Avatars

![Teaser image](./assets/teaser.png)

**InvertAvatar: Incremental GAN Inversion for Generalized Head Avatars**<br>
[Xiaochen Zhao*](https://xiaochen-zhao.netlify.app/), [Jingxiang Sun*](https://mrtornado24.github.io/), [Lizhen Wang](https://lizhenwangt.github.io/), Jinli Suo, [Yebin Liu](http://www.liuyebin.com/)<br><br>


[**Project**](https://xchenz.github.io/invertavatar_page/) | [**Paper**](https://arxiv.org/abs/2312.02222)

Abstract: *While high fidelity and efficiency are central to the creation of digital head avatars, recent methods relying on 2D or 3D generative models often experience limitations such as shape distortion, expression inaccuracy, and identity flickering. Additionally, existing one-shot inversion techniques fail to fully leverage multiple input images for detailed feature extraction. We propose a novel framework, \textbf{Incremental 3D GAN Inversion}, that enhances avatar reconstruction performance using an algorithm designed to increase the fidelity from multiple frames, resulting in improved reconstruction quality proportional to frame count. Our method introduces a unique animatable 3D GAN prior with two crucial modifications for enhanced expression controllability alongside an innovative neural texture encoder that categorizes texture feature spaces based on UV parameterization. Differentiating from traditional techniques, our architecture emphasizes pixel-aligned image-to-image translation, mitigating the need to learn correspondences between observation and canonical spaces. Furthermore, we incorporate ConvGRU-based recurrent networks for temporal data aggregation from multiple frames, boosting geometry and texture detail reconstruction. The proposed paradigm demonstrates state-of-the-art performance on one-shot and few-shot avatar animation tasks.*

### Prepare Dataset

We provide a processed [demo dataset](https://drive.google.com/file/d/1U_T5uCiUBiJXyPuxtZU_QncbBOtB25Ju/view?usp=drive_link). Please download and unzip it into `data`.

Please download [faceverse file](https://drive.google.com/file/d/1lv2lGiTZet1pMIX4gUVy_wD4VYDOq1Tz/view?usp=drive_link) (`data_preprocessing/FaceVerse/v3/faceverse_v3_1.npy`).

We also provide preprocessing code in `data_preprocessing`. If you want to generate dataset from the video, please correctly prepare [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21?tab=readme-ov-file#prepare-prerequisite-models) and [faceverse fitting](https://github.com/XChenZ/havatar#prepare-dataset). Please specify the correct fv_preprocess_path in `data_preprocessing/preprocess_person_video_dataset.py`.


```.bash
cd data_preprocessing
python preprocess_person_video_dataset.py --vid data_preprocess/Obama.mp4 --savedir data/tgt_data
```

## Animate Avatar

We provide our pretrained animatable 3D GAN (Next3D++) and inversion networks (InvertAvatar) [here](https://drive.google.com/drive/folders/1AvqyvMzMwskI4rCMDwnlQh6heT842G5K?usp=sharing), please download and put them into `pretrained_model/`.


### Reenactment

```.bash
# Reenact Next3D

CUDA_VISIBLE_DEVICES=0 python reenact_avatar_next3d.py \
--drive_root ./data/tgt_data/dataset/images512x512 \
--grid 5x2 \
--seeds 100-108 \
--outdir out/reenact_gan \
--fname obama_reenact_gan \
--trunc 0.7 \
--fixed_camera False \
--network pretrained_model/ani3dgan512.pkl
```

```.bash
# Avatar Reconstruction and Reenact 
CUDA_VISIBLE_DEVICES=0 python eval_seq.py \
--outdir out/fs \
--reload_modules True \
--network pretrained_model/FSInvertAvatar.pkl
```

```.bash
# Better one-shot Avatar Reconstruction and Reenact 
CUDA_VISIBLE_DEVICES=0 python eval_updated_os.py \
--outdir out/os \
--reload_modules True \
--network pretrained_model/updatedOSInvertAvatar.pkl
```

## Citation

```
@inproceedings{10.1145/3641519.3657478,
author = {Zhao, Xiaochen and Sun, Jingxiang and Wang, Lizhen and Suo, Jinli and Liu, Yebin},
title = {InvertAvatar: Incremental GAN Inversion for Generalized Head Avatars},
year = {2024},
isbn = {9798400705250},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3641519.3657478},
doi = {10.1145/3641519.3657478},
booktitle = {ACM SIGGRAPH 2024 Conference Papers},
articleno = {59},
numpages = {10},
keywords = {3D head avatar, GAN inversion, few-shot reconstruction, one-shot reconstruction, recurrent neural network},
location = {Denver, CO, USA},
series = {SIGGRAPH '24}
}


```
