################E4E training############################################
python encoder_inversion/train.py \
--outdir=training-runs/encoder_inversion/e4e \
--cfg=ffhq \
--data /data/zhaoxiaochen/Dataset/FFHQ/images512x512 \
--rdata /data/zhaoxiaochen/Dataset/FFHQ/orthRender256x256_face \
--gpus=8 \
--batch=32 \
--mbstd-group=4 \
--gamma=1 \
--snap=10 \
--gen_pose_cond=1 \
--model_version v20 \
--configs_path=encoder_inversion/config/train_e4e_real.yaml \
--gen_lms_cond 0 \
--training_state e4e \
--gen_mask_cond 0

################Few-shot Training###################################

python encoder_inversion/train.py \
--outdir=training-runs/encoder_inversion/few-shot \  #VFHQ/gru/both_TriNetSFT/multiT_onlyIreal \
--cfg=ffhq \
--data /data/zhaoxiaochen/Dataset/VFHQ/dataset/images512x512 \
--rdata /data/zhaoxiaochen/Dataset/VFHQ/dataset/orthRender256x256_face_eye \
--dataset_class_name encoder_inversion.dataset_video.VideoFolderDataset \
--gpus=8 \
--batch=8 \
--mbstd-group=1 \
--gamma=1 \
--tick=1 \
--snap=4 \
--gen_pose_cond=1 \
--model_version v20 \
--configs_path=encoder_inversion/config/train_textureUnet_video.yaml \
--gen_lms_cond 0 \
--gen_mask_cond 1 \
--gen_uv_cond 1 \
--training_state fewshot \
--resume training-runs/encoder_inversion/e4e/path-to-your-pkl.pkl


################Improved One-shot Training###################################
python encoder_inversion/train.py \
--outdir=training-runs/encoder_inversion/v20_128/both_TriSFT_TriplaneTexNet_uvFInp/BothSegFormerDecoder \
--cfg=ffhq \
--data /data/zhaoxiaochen/Dataset/FFHQ/images512x512 \
--rdata /data/zhaoxiaochen/Dataset/FFHQ/orthRender256x256_face_eye \
--gpus=8 \
--batch=16 \
--mbstd-group=2 \
--gamma=8 \
--snap=10 \
--gen_pose_cond=1 \
--model_version v20 \
--configs_path=encoder_inversion/config/train_textureUnet_real.yaml \
--gen_lms_cond 0 \
--gen_mask_cond 0 \
--gen_uv_cond 1 \
--training_state oneshot \
--resume training-runs/encoder_inversion/e4e/path-to-your-pkl.pkl
#--resume training-runs/encoder_inversion/v20_128/both_TriSFT_TriplaneTexNet_uvFInp/BothSegFormerDecoder-000961.pkl
#--resume /data/zhaoxiaochen/Code/animatable_eg3d/training-runs/encoder_inversion/v20_128/sftTriNet_TriplaneTexNet_uvInp-000240.pkl
#--resume /data/zhaoxiaochen/Code/animatable_eg3d/training-runs/encoder_inversion/v20_128/both_TriSFT_TexOri/bothData/00001-ffhq-images512x512-gpus4-batch16/network-snapshot-000800.pkl
#--resume /data/zhaoxiaochen/Code/animatable_eg3d/training-runs/encoder_inversion/onlyTriplanet/onlyGenData_L3P/withADV/00001-ffhq-images512x512-gpus2-batch8/network-snapshot-000080.pkl
#--resume training-runs/encoder_inversion/v20_128/lowLR_higherWD_constantNoise/00000-ffhq-images512x512-gpus6-batch24/network-snapshot-000721.pkl
