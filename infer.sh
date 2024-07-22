CUDA_VISIBLE_DEVICES=0 python eval_updated_os.py \
--outdir out/os \
--reload_modules True \
--network pretrained_model/updatedOSInvertAvatar.pkl

CUDA_VISIBLE_DEVICES=0 python eval_seq.py \
--outdir out/fs \
--reload_modules True \
--network pretrained_model/FSInvertAvatar.pkl

CUDA_VISIBLE_DEVICES=0 python reenact_avatar_next3d.py \
--drive_root ./data/tgt_data/dataset/images512x512 \
--grid 5x2 \
--seeds 100-108 \
--outdir out/reenact_gan \
--fname obama_reenact_gan \
--trunc 0.7 \
--fixed_camera False \
--network pretrained_model/ani3dgan512.pkl