# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
import torch.nn.functional as F
from dnnlib.util import EasyDict
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
from encoder_inversion.criteria.id_loss import IDLoss
from encoder_inversion.criteria.lpips.lpips import LPIPS
from encoder_inversion.models.e4e import LatentCodesDiscriminator, LatentCodesPool
from einops import rearrange
#----------------------------------------------------------------------------

class WLoss:
	def __init__(
		self, 
		device, 
		G,D,
		augment_pipe,
		r1_gamma,
		loss_weight,
		loss_path,
		frm_per_vid,
		multiT_training,
		**unused_kwargs
	):
		
		super().__init__()
		self.device             = device
		self.I                  = G
		self.D                  = D
		self.augment_pipe = augment_pipe
		self.r1_gamma = r1_gamma
		self.neural_rendering_resolution = self.I.generator.neural_rendering_resolution
		self.loss_weight = EasyDict(loss_weight)
		self.loss_path = EasyDict(loss_path)
		self.__init_loss()
		self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
		self.filter_mode = 'antialiased'
		self.optimize_e4e = False
		self.frm_per_vid = frm_per_vid
		self.multiT_training = multiT_training
		# self.texture_feats = [torch.zeros(4, 32, 32, 32), torch.zeros(4, 512, 32, 32), torch.zeros(4, 512, 64, 64), torch.zeros(4, 256, 128, 128),
		# 				 torch.zeros(4, 128, 256, 256), torch.zeros(4, 32, 256, 256)]
		# self.static_feats = [torch.zeros(4, 96, 32, 32), torch.zeros(4, 512, 32, 32), torch.zeros(4, 512, 64, 64), torch.zeros(4, 256, 128, 128),
		# 				 torch.zeros(4, 128, 256, 256), torch.zeros(4, 96, 256, 256)]
		# for i in range(len(self.texture_feats)):
		# 	self.texture_feats[i] = self.texture_feats[i].to(torch.float32).to(device)
		# 	self.static_feats[i] = self.static_feats[i].to(torch.float32).to(device)

	def __init_loss(self):
		self.l1_loss = torch.nn.L1Loss(reduction='mean').eval()
		self.l2_loss = torch.nn.MSELoss(reduction='mean').eval()
		if self.loss_weight.lpips > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.loss_weight.id > 0:
			self.id_loss = IDLoss(self.loss_path.path_ir_se50).to(self.device).eval()
		# if self.loss_weight.w_l1 > 0:
		# 	self.w_l1_loss = torch.nn.SmoothL1Loss(reduction='mean').eval()
		# if self.loss_weight.w_regular > 0:
		# 	self.batch_latent_avg = self.I.latent_avg.unsqueeze(0).repeat(1, 14*2, 1)
		# if self.loss_weight.w_discriminator > 0:
		# 	# self.w_D = LatentCodesDiscriminator(512, 4).to(self.device)
		# 	# self.w_D_opt = torch.optim.Adam(list(self.w_D.parameters()), lr=2e-5)
		# 	self.real_w_pool = LatentCodesPool(pool_size=50)
		# 	self.fake_w_pool = LatentCodesPool(pool_size=50)

	def run_I_gen(self, z, c, v, neural_rendering_resolution, update_emas=False, truncation_psi=1):
		with torch.no_grad():  # 此处是在生成虚拟数据，没有梯度流过
			ws = self.I.generator.mapping(z, c, truncation_psi=truncation_psi, update_emas=update_emas)
			gen_output = self.I.generator.synthesis(ws, c, v, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
		I_output = self.I(gen_output['image'], c, v)	# 原版用的是生成的c
		return I_output, ws, gen_output

	def run_G(self, real_vid_c, real_vid_v, real_vid_uv, ws, real_vid_frm, real_vid_e4e, gt_texture_feats=None, gt_static_feats=None, only_image=True, return_idx=None):	# 并行计算
		if self.multiT_training:
			return self.run_G_seq_multiT(real_vid_c, real_vid_v, real_vid_uv, ws, real_vid_frm, real_vid_e4e, gt_texture_feats, gt_static_feats, only_image, return_idx)
		else:
			return self.run_G_seq(real_vid_c, real_vid_v, real_vid_uv, ws, real_vid_frm, real_vid_e4e, gt_texture_feats, gt_static_feats, only_image)

	def run_G_seq_old(self, real_vid_c, real_vid_v, real_vid_uv, ws, real_vid_frm, real_vid_e4e, gt_texture_feats=None, gt_static_feats=None, only_image=True):	# 并行计算
		'''
		real_vid_c, real_vid_v, real_vid_uv, real_vid_frm: [B, T, ]
		ws: [B,]
		'''
		B, T = real_vid_c.shape[:2]
		with torch.no_grad():
			if ws is None: ws = self.I.encode(real_vid_frm[:, 0])
			vid_ws = ws.unsqueeze(1).expand(-1, real_vid_c.shape[1], -1, -1).flatten(0, 1)
			vid_c = real_vid_c.flatten(0, 1)
			vid_v = real_vid_v.flatten(0, 1)

			texture_feats = self.I.generator.texture_backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const') if gt_texture_feats is None else gt_texture_feats
			static_feats = self.I.generator.backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const') if gt_static_feats is None else gt_static_feats
			if real_vid_e4e is None:
				real_vid_e4e = self.I.generator.synthesis_withTexture(vid_ws,
									[feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in texture_feats],
									vid_c, {'uvcoords_image': vid_v}, noise_mode='const',
									static_feats=[feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in static_feats])['image'].unflatten(0, (-1, T))

			real_vid_deltax = real_vid_e4e - real_vid_frm
			# if gt_static_feats is None and self.I.unet_encoder.triplane_unet is not None:	#目前这种策略几乎相当于随意在视频中挑一帧进行triplane offset的预测并固定
			# 	triplane_offset = self.I.unet_encoder.triplane_unet(torch.cat([real_vid_frm[:, 0], real_vid_deltax[:, 0]], dim=-3))
			# 	static_feats = static_feats[:-1] + [static_feats[-1] + triplane_offset]
			triplane_input = torch.cat([real_vid_frm, real_vid_deltax], dim=-3)
			real_vid_uv_input = self.I.get_unet_uvinput(real_vid_uv.flatten(0, 1), real_vid_deltax.flatten(0, 1)).unflatten(0, (B, T))
		if self.I.unet_encoder.use_msfeat:
			cond_list = self.I.unet_encoder.texture_unet.forward_onlyEncoder(real_vid_uv_input)
			if T == 1 and self.frm_per_vid > 1:  # 如果是单张图片的输入，那么就视为输入了四张相同的图片
				cond_list = [cond.unflatten(0, (-1, T)).expand(-1, 4, -1, -1, -1).flatten(0, 1) for cond in cond_list]
			texture_feat_offsets, _ = self.I.unet_encoder.texture_unet.forward_onlyDecoder(T=cond_list[0].shape[0] // B, cond_list=cond_list)
			texture_feat_offsets = {key: texture_feat_offsets[key].unsqueeze(2).expand(-1, -1, T, -1, -1, -1).flatten(0, 1) for key in texture_feat_offsets}

			if T == 1 and self.frm_per_vid > 1: triplane_input = triplane_input.expand(-1, self.frm_per_vid, -1, -1, -1)
			triplane_feat_offsets = self.I.unet_encoder.triplane_unet(triplane_input)
			if self.I.unet_encoder.triplane_unet.use_gru: triplane_feat_offsets = triplane_feat_offsets[0]
			triplane_feat_offsets = {key: triplane_feat_offsets[key].unsqueeze(2).expand(-1, -1, T, -1, -1, -1).flatten(0, 1) for key in triplane_feat_offsets}

			I_output = self.I.generator.synthesis_withCondition(vid_ws, vid_c, {'uvcoords_image': vid_v}, noise_mode='const', only_image=only_image,
															  static_feats_conditions=triplane_feat_offsets, texture_feats_conditions=texture_feat_offsets)
		else:
			if gt_texture_feats is None:
				real_vid_uv_input = self.I.get_unet_uvinput(real_vid_uv.flatten(0, 1), real_vid_deltax.flatten(0, 1)).unflatten(0, (B, T))
				cond_list = self.I.unet_encoder.texture_unet.forward_onlyEncoder(real_vid_uv_input)
				if T == 1 and self.frm_per_vid > 1: 	# 如果是单张图片的输入，那么就视为输入了四张相同的图片
					cond_list = [cond.unflatten(0, (-1, T)).expand(-1, 4, -1, -1, -1).flatten(0, 1) for cond in cond_list]

				texture_offsets, _ = self.I.unet_encoder.texture_unet.forward_onlyDecoder(T=cond_list[0].shape[0]//B, cond_list=cond_list, return_list=True)
				assert len(texture_offsets) == len(texture_feats)
				texture_feats = [feat + offset for feat, offset in zip(texture_feats, texture_offsets)]
			texture_feats = [feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in texture_feats]

			triplane_input = torch.cat([real_vid_frm, real_vid_deltax], dim=-3)
			if T==1 and self.frm_per_vid > 1: triplane_input = triplane_input.expand(-1, self.frm_per_vid, -1, -1, -1)
			if not self.I.unet_encoder.use_msfeat:	# Triplane_unet输出3*32通道的offset
				if gt_static_feats is None:
					# triplane_offset = self.I.unet_encoder.triplane_unet(triplane_input.flatten(0, 1))
					triplane_offset, _ = self.I.unet_encoder.triplane_unet(triplane_input)
					# triplane_offset = self.I.unet_encoder.triplane_unet(triplane_input[:, 0])
					if self.I.unet_encoder.triplane_unet.seq2seq:
						static_feats = [feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in static_feats]
						static_feats = static_feats[:-1] + [static_feats[-1] + triplane_offset]

					else:
						updated_static_feats = static_feats[:-1] + [static_feats[-1] + triplane_offset]
						static_feats = [feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in updated_static_feats]
				I_output = self.I.generator.synthesis_withTexture(vid_ws, texture_feats, vid_c, {'uvcoords_image': vid_v}, static_feats=static_feats,
																  noise_mode='const')
			else:
				triplane_feat_offsets = self.I.unet_encoder.triplane_unet(triplane_input)
				if self.I.unet_encoder.triplane_unet.use_gru: triplane_feat_offsets = triplane_feat_offsets[0]
				if isinstance(triplane_feat_offsets, list):
					triplane_feat_offsets = [feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in triplane_feat_offsets]
				elif isinstance(triplane_feat_offsets, dict):
					triplane_feat_offsets = {key: triplane_feat_offsets[key].unsqueeze(2).expand(-1, -1, T, -1, -1, -1).flatten(0, 1) for key in triplane_feat_offsets}
				else: assert False, type(triplane_feat_offsets)
				I_output = self.I.generator.synthesis_withTexture(vid_ws, texture_feats, vid_c, {'uvcoords_image': vid_v}, static_feats_conditions=triplane_feat_offsets, noise_mode='const')
		if only_image:
			return {'image': I_output['image'].unflatten(0, (B, T)), 'e4e_image': real_vid_e4e}#, 'x_input': real_vid_uv_input.flatten(0, 1)}
		else:
			return I_output

	def run_G_seq(self, real_vid_c, real_vid_v, real_vid_uv, ws, real_vid_frm, real_vid_e4e, gt_texture_feats=None, gt_static_feats=None, only_image=True):	# 并行计算
		'''
		real_vid_c, real_vid_v, real_vid_uv, real_vid_frm: [B, T, ]
		ws: [B,]
		'''
		B, T = real_vid_c.shape[:2]
		with torch.no_grad():
			if ws is None: ws = self.I.encode(real_vid_frm[:, 0])
			vid_ws = ws.unsqueeze(1).expand(-1, real_vid_c.shape[1], -1, -1).flatten(0, 1)
			vid_c = real_vid_c.flatten(0, 1)
			vid_v = real_vid_v.flatten(0, 1)

			texture_feats = self.I.generator.texture_backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const') if gt_texture_feats is None else gt_texture_feats
			static_feats = self.I.generator.backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const') if gt_static_feats is None else gt_static_feats
			if real_vid_e4e is None:
				real_vid_e4e = self.I.generator.synthesis_withTexture(ws=vid_ws, c=vid_c, mesh_condition={'uvcoords_image': vid_v}, noise_mode='const',
									texture_feats=[feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in texture_feats],
									static_feats=[feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in static_feats])['image'].unflatten(0, (-1, T))

			real_vid_deltax = real_vid_e4e - real_vid_frm

		if gt_texture_feats is None:
			real_vid_uv_input = self.I.get_unet_uvinput(real_vid_uv.flatten(0, 1), real_vid_deltax.flatten(0, 1)).unflatten(0, (B, T))
			cond_list = self.I.unet_encoder.texture_unet.forward_onlyEncoder(real_vid_uv_input)
			if T == 1 and self.frm_per_vid > 1:  # 如果是单张图片的输入，那么就视为输入了四张相同的图片
				cond_list = [cond.unflatten(0, (-1, T)).expand(-1, 4, -1, -1, -1).flatten(0, 1) for cond in cond_list]
			texture_feat_offsets, _ = self.I.unet_encoder.texture_unet.forward_onlyDecoder(T=cond_list[0].shape[0] // B, cond_list=cond_list)
			# if self.I.unet_encoder.texture_unet.use_gru: texture_feat_offsets = texture_feat_offsets[0]
			if self.I.unet_encoder.texture_unet_use_msfeat:
				updated_texture_feats = self.I.generator.texture_backbone.synthesis(ws, cond_list=None, return_list=True, feat_conditions=texture_feat_offsets,
																					update_emas=False, noise_mode='const')
			else:
				if len(texture_feat_offsets) == len(texture_feats):
					updated_texture_feats = [feat + offset for feat, offset in zip(texture_feats, texture_feat_offsets)]
				else:  # len(texture_offsets) < len(e4e_texture_feats)
					updated_texture_feats = [feat + offset for feat, offset in
									 zip(texture_feats, texture_feat_offsets[:len(texture_feat_offsets)])] + texture_feats[len(texture_feat_offsets):]
		else:
			updated_texture_feats = texture_feats

		if gt_static_feats is None:
			triplane_input = torch.cat([real_vid_frm, real_vid_deltax], dim=-3)
			triplane_feat_offsets = self.I.unet_encoder.triplane_unet(triplane_input)
			if self.I.unet_encoder.triplane_unet.use_gru: triplane_feat_offsets = triplane_feat_offsets[0]
			if self.I.unet_encoder.triplane_unet_use_msfeat:
				updated_static_feats = self.I.generator.backbone.synthesis(ws, cond_list=None, return_list=True, feat_conditions=triplane_feat_offsets,
																		   update_emas=False, noise_mode='const')
			else:
				updated_static_feats = static_feats[:-1] + [static_feats[-1] + triplane_feat_offsets]
		else:
			updated_static_feats = static_feats

		I_output = self.I.generator.synthesis_withTexture(ws=vid_ws, c=vid_c, mesh_condition={'uvcoords_image': vid_v}, noise_mode='const',
														  texture_feats=[feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in updated_texture_feats],
														  static_feats=[feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in updated_static_feats])
		if only_image:
			return {'image': I_output['image'].unflatten(0, (B, T)), 'e4e_image': real_vid_e4e}#, 'x_input': real_vid_uv_input.flatten(0, 1)}
		else:
			return I_output

	def run_G_seq_transformer(self, real_vid_c, real_vid_v, real_vid_uv, ws, real_vid_frm, real_vid_e4e, gt_texture_feats=None, gt_static_feats=None, only_image=True):	# 并行计算
		'''
		real_vid_c, real_vid_v, real_vid_uv, real_vid_frm: [B, T, ]
		ws: [B,]
		'''
		B, T = real_vid_c.shape[:2]
		with torch.no_grad():
			if ws is None: ws = self.I.encode(real_vid_frm[:, 0])
			vid_ws = ws.unsqueeze(1).expand(-1, real_vid_c.shape[1], -1, -1).flatten(0, 1)
			vid_c = real_vid_c.flatten(0, 1)
			vid_v = real_vid_v.flatten(0, 1)

			texture_feats = self.I.generator.texture_backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const') if gt_texture_feats is None else gt_texture_feats
			static_feats = self.I.generator.backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const') if gt_static_feats is None else gt_static_feats
			if real_vid_e4e is None:
				real_vid_e4e = self.I.generator.synthesis_withTexture(vid_ws,
									[feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in texture_feats],
									vid_c, {'uvcoords_image': vid_v}, noise_mode='const',
									static_feats=[feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in static_feats])['image'].unflatten(0, (-1, T))

			real_vid_deltax = real_vid_e4e - real_vid_frm

		if gt_texture_feats is None:
			real_vid_uv_input = self.I.get_unet_uvinput(real_vid_uv.flatten(0, 1), real_vid_deltax.flatten(0, 1)).unflatten(0, (B, T))
			cond_list = self.I.unet_encoder.texture_unet.forward_onlyEncoder(real_vid_uv_input)
			if T == 1 and self.frm_per_vid > 1: 	# 如果是单张图片的输入，那么就视为输入了四张相同的图片
				cond_list = [cond.unflatten(0, (-1, T)).expand(-1, 4, -1, -1, -1).flatten(0, 1) for cond in cond_list]

			texture_offsets, _ = self.I.unet_encoder.texture_unet.forward_onlyDecoder(T=cond_list[0].shape[0]//B, cond_list=cond_list, return_list=True)
			assert len(texture_offsets) == len(texture_feats)
			texture_feats = [feat + offset for feat, offset in zip(texture_feats, texture_offsets)]
		texture_feats = [feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in texture_feats]

		triplane_input = torch.cat([real_vid_frm, real_vid_deltax], dim=-3)
		if T==1 and self.frm_per_vid > 1: triplane_input = triplane_input.expand(-1, self.frm_per_vid, -1, -1, -1)
		if gt_static_feats is None:
			triplane_offset = self.I.unet_encoder.triplane_unet(triplane_input.flatten(0, 1)).unflatten(0, (B, T))	# [B, T, C, H, W]
			triplane_offset = triplane_offset.permute(0, 2, 1, 3, 4).flatten(1, 2)  # [B, C, T, H, W] -> [B, C*T, H, W]
			assert T == 4, T
			expanded_triplane_offset = torch.nn.functional.pixel_shuffle(triplane_offset, upscale_factor=2)	# [B, C, 2H, 2W]
			for _ in range(2):
				expanded_triplane_offset = self.I.unet_encoder.cca(expanded_triplane_offset, expanded_triplane_offset, expanded_triplane_offset)
			triplane_offset = torch.nn.functional.pixel_unshuffle(expanded_triplane_offset, downscale_factor=2).unflatten(1, (-1, T))[:, :, 0]	# [B, C*T, H, W] -> [B, C, T, H, W] -> [B, C, H, W]
			updated_static_feats = static_feats[:-1] + [static_feats[-1] + triplane_offset]
			static_feats = [feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in updated_static_feats]
		I_output = self.I.generator.synthesis_withTexture(vid_ws, texture_feats, vid_c, {'uvcoords_image': vid_v}, static_feats=static_feats,
														  noise_mode='const')

		if only_image:
			return {'image': I_output['image'].unflatten(0, (B, T)), 'e4e_image': real_vid_e4e}
		else:
			return I_output

	def run_G_seq_multiT(self, real_vid_c, real_vid_v, real_vid_uv, ws, real_vid_frm, real_vid_e4e, gt_texture_feats=None, gt_static_feats=None,
						 only_image=True, return_idx=None):	# 并行计算
		'''
		real_vid_c, real_vid_v, real_vid_uv, real_vid_frm: [B, T, ]
		ws: [B,]
		'''
		B, T = real_vid_c.shape[:2]
		texture_rlist, triplane_rlist = None, None
		with torch.no_grad():
			if ws is None: ws = self.I.encode(real_vid_frm[:, 0])
			vid_ws = ws.unsqueeze(1).expand(-1, self.frm_per_vid, -1, -1).flatten(0, 1)
			# vid_c = real_vid_c.flatten(0, 1)
			# vid_v = real_vid_v.flatten(0, 1)

			texture_feats = self.I.generator.texture_backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const') if gt_texture_feats is None else gt_texture_feats
			static_feats = self.I.generator.backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const') if gt_static_feats is None else gt_static_feats
			# assert T > self.frm_per_vid
			num_iter = T // self.frm_per_vid
			for i in range(num_iter):
				real_vid_e4e = self.I.generator.synthesis_withTexture(ws=vid_ws, c=real_vid_c[:, i*self.frm_per_vid:(i+1)*self.frm_per_vid].flatten(0, 1),
									mesh_condition={'uvcoords_image': real_vid_v[:, i*self.frm_per_vid:(i+1)*self.frm_per_vid].flatten(0, 1)}, noise_mode='const',
									texture_feats=[feat.unsqueeze(1).expand(-1, self.frm_per_vid, -1, -1, -1).flatten(0, 1) for feat in texture_feats],
									static_feats=[feat.unsqueeze(1).expand(-1, self.frm_per_vid, -1, -1, -1).flatten(0, 1) for feat in static_feats])['image'].unflatten(0, (-1, self.frm_per_vid))

				real_vid_deltax = real_vid_e4e - real_vid_frm[:, i*self.frm_per_vid:(i+1)*self.frm_per_vid]
				real_vid_uv_input = self.I.get_unet_uvinput(real_vid_uv[:, i * self.frm_per_vid:(i + 1) * self.frm_per_vid].flatten(0, 1),
															real_vid_deltax.flatten(0, 1)).unflatten(0, (B, self.frm_per_vid))
				triplane_input = torch.cat([real_vid_frm[:, i * self.frm_per_vid:(i + 1) * self.frm_per_vid], real_vid_deltax], dim=-3)
				if i < num_iter - 1:
					cond_list = self.I.unet_encoder.texture_unet.forward_onlyEncoder(real_vid_uv_input)
					texture_rlist = self.I.unet_encoder.texture_unet.forward_onlyDecoder(T=cond_list[0].shape[0] // B, cond_list=cond_list, r_list=texture_rlist)[1]
					triplane_rlist = self.I.unet_encoder.triplane_unet(triplane_input, r_list=triplane_rlist)[1]

		# real_vid_uv_input = self.I.get_unet_uvinput(real_vid_uv[:, -self.frm_per_vid:].flatten(0, 1), real_vid_deltax.flatten(0, 1)).unflatten(0, (B, self.frm_per_vid))
		# triplane_input = torch.cat([real_vid_frm[:, -self.frm_per_vid:], real_vid_deltax], dim=-3)
		cond_list = self.I.unet_encoder.texture_unet.forward_onlyEncoder(real_vid_uv_input)
		texture_feat_offsets = self.I.unet_encoder.texture_unet.forward_onlyDecoder(T=cond_list[0].shape[0] // B, cond_list=cond_list, r_list=texture_rlist)[0]
		if len(texture_feat_offsets) == len(texture_feats):
			updated_texture_feats = [feat + offset for feat, offset in zip(texture_feats, texture_feat_offsets)]
		else:  # len(texture_offsets) < len(e4e_texture_feats)
			updated_texture_feats = [feat + offset for feat, offset in
									 zip(texture_feats, texture_feat_offsets[:len(texture_feat_offsets)])] + texture_feats[len(texture_feat_offsets):]

		triplane_feat_offsets = self.I.unet_encoder.triplane_unet(triplane_input, r_list=triplane_rlist)[0]
		updated_static_feats = self.I.generator.backbone.synthesis(ws, cond_list=None, return_list=True, feat_conditions=triplane_feat_offsets,
																   update_emas=False, noise_mode='const')
		if return_idx is None:
			out_imgs, out_e4e = [], []
			for i in range(num_iter):
				I_output = self.I.generator.synthesis_withTexture(ws=vid_ws, c=real_vid_c[:, i*self.frm_per_vid:(i+1)*self.frm_per_vid].flatten(0, 1),
																  mesh_condition={'uvcoords_image': real_vid_v[:, i*self.frm_per_vid:(i+1)*self.frm_per_vid].flatten(0, 1)},
																  noise_mode='const',
																  texture_feats=[feat.unsqueeze(1).expand(-1, self.frm_per_vid, -1, -1, -1).flatten(0, 1)
																				 for feat in updated_texture_feats],
																  static_feats=[feat.unsqueeze(1).expand(-1, self.frm_per_vid, -1, -1, -1).flatten(0, 1)
																				for feat in updated_static_feats])
				out_imgs.append(I_output['image'].unflatten(0, (B, self.frm_per_vid)))
				out_e4e.append(real_vid_e4e)
			return {'image': torch.cat(out_imgs, dim=1), 'e4e_image': torch.cat(out_e4e, dim=1)}
		else:
			assert isinstance(return_idx, list) and len(return_idx) == self.frm_per_vid
			I_output = self.I.generator.synthesis_withTexture(ws=vid_ws, c=real_vid_c[:, return_idx].flatten(0, 1),
															  mesh_condition={'uvcoords_image': real_vid_v[:, return_idx].flatten(0, 1)}, noise_mode='const',
															  texture_feats=[feat.unsqueeze(1).expand(-1, self.frm_per_vid, -1, -1, -1).flatten(0, 1) for feat in updated_texture_feats],
															  static_feats=[feat.unsqueeze(1).expand(-1, self.frm_per_vid, -1, -1, -1).flatten(0, 1) for feat in updated_static_feats])
			if only_image:
				return {'image': I_output['image'].unflatten(0, (B, self.frm_per_vid)), 'e4e_image': real_vid_e4e}#, 'x_input': real_vid_uv_input.flatten(0, 1)}
			else:
				return I_output

	def run_D(self, img, c, update_emas=False):
		logits = self.D(img, c, update_emas=update_emas)
		return logits

	def accumulate_gradients(self, phase, real_vid, real_vid_c, real_vid_v, real_vid_uv, real_vid_ws, real_vid_mm, real_vid_gens, gain):
		# real_vid:	[B, T, C, H, W]
		#
		optimize_texture = False
		optimize_staticTri = False
		wd_r1_gamma = 10
		T = real_vid_c.shape[1]
		assert phase in ['Ireal', 'Igen', 'Dmain', 'Dreg']

		loss_dict = {}
		if phase == 'Igen':
			with torch.autograd.profiler.record_function('Igen'):
				gen_z = torch.randn([real_vid.shape[0], self.I.generator.z_dim], device=self.device)
				with torch.no_grad():  # 此处是在生成虚拟数据，没有梯度流过
					ws = self.I.generator.mapping(gen_z, real_vid_c[:, 0], truncation_psi=0.6, update_emas=False)
					gt_texture_feats = self.I.generator.texture_backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const')
					gt_static_feats = self.I.generator.backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False, noise_mode='const')
					ws = ws.unsqueeze(1).expand(-1, T, -1, -1).flatten(0, 1)
					gen_output = self.I.generator.synthesis_withTexture(ws,
											[feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in gt_texture_feats],
											real_vid_c.flatten(0, 1), {'uvcoords_image': real_vid_v.flatten(0, 1)}, noise_mode='const',
											static_feats=[feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in gt_static_feats])
					real_vid = gen_output['image'].unflatten(0, (-1, T))

				I_output = self.run_G(real_vid_c, real_vid_v, real_vid_uv, ws=None, real_vid_frm=real_vid[:, :, :3], real_vid_e4e=None,
										  gt_texture_feats=gt_texture_feats if optimize_staticTri else None,
				                          gt_static_feats=gt_static_feats if optimize_texture else None, only_image=False)

				if self.loss_weight.adv > 0:
					gen_logits = self.run_D({'image_raw': I_output['feature_image'][:, :3], 'image': torch.cat([I_output['image']], dim=1)},
											real_vid_c.flatten(0, 1) * 0)
					loss_adv = torch.nn.functional.softplus(-gen_logits).mean()
					loss_dict['adv'] = loss_adv
					training_stats.report('G_Loss/gen/loss_adv', loss_adv)

				I_output['image'] = torch.nn.functional.interpolate(I_output['image'], size=(256, 256), mode='bilinear', align_corners=False,
																	antialias=True)#[:, :, 44:-44, 44:-44]
				gen_output['image'] = torch.nn.functional.interpolate(gen_output['image'], size=(256, 256), mode='bilinear', align_corners=False,
																	  antialias=True)#[:, :, 44:-44, 44:-44]

				# Image Level
				if self.loss_weight.l1 > 0:
					# l2_loss = self.l2_loss(gen_output['image'][:, :3], I_output['image'])
					l1_loss = F.l1_loss(gen_output['image'][:, :3], I_output['image'])
					loss_dict['l1'] = l1_loss
					training_stats.report('G_Loss/gen/loss_l1', l1_loss * self.loss_weight['l1'])
				if self.loss_weight.lpips > 0:
					lpips_loss = self.lpips_loss(gen_output['image'][:, :3], I_output['image'])
					loss_dict['lpips'] = lpips_loss
					training_stats.report('G_Loss/gen/loss_lpips', lpips_loss * self.loss_weight['lpips'])

				if self.loss_weight.raw_l1 > 0:
					raw_l1_loss = F.l1_loss(gen_output['feature_image'], I_output['feature_image'])
					loss_dict['raw_l1'] = raw_l1_loss
					training_stats.report('G_Loss/gen/loss_raw_l1', raw_l1_loss * self.loss_weight['raw_l1'])
				if self.loss_weight.tri > 0:
					tri_l1_loss = F.l1_loss(gen_output['triplane'], I_output['triplane'])
					loss_dict['tri'] = tri_l1_loss
					training_stats.report('G_Loss/gen/loss_tri_l1', tri_l1_loss * self.loss_weight['tri'])
				if self.loss_weight.lr_lpips > 0:
					lr_lpips_loss = self.lpips_loss(gen_output['feature_image'][:, :3], I_output['feature_image'][:, :3])
					loss_dict['lr_lpips'] = lr_lpips_loss
					training_stats.report('G_Loss/gen/loss_lr_lpips', lr_lpips_loss * self.loss_weight['lr_lpips'])

		if phase == 'Ireal':
			with torch.autograd.profiler.record_function('Ireal'):
				fake_idx = [0] + np.random.choice(np.arange(1, real_vid_c.shape[1]), self.frm_per_vid-1, replace=False).tolist() if self.multiT_training else list(range(real_vid_c.shape[1]))
				fake_vid = self.run_G(real_vid_c, real_vid_v, real_vid_uv, real_vid_ws, real_vid[:, :, :3], real_vid_gens, return_idx=fake_idx)['image']

				# Image Level
				if self.multiT_training:
					if real_vid.shape[2] > 3:
						fake_vid[:, 1:] = fake_vid[:, 1:] * real_vid[:, fake_idx[1:], -1:] + real_vid[:, fake_idx[1:], :3] * (1. - real_vid[:, fake_idx[1:], -1:])
					real_img = torch.nn.functional.interpolate(real_vid[:, fake_idx, :3].flatten(0, 1), size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
					fake_img = torch.nn.functional.interpolate(fake_vid.flatten(0, 1), size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
				else:
					if real_vid.shape[2] > 3 and T > 1:  # 首帧计算全图loss，其余帧计算头部loss
						fake_vid[:, 1:] = fake_vid[:, 1:] * real_vid[:, 1:, -1:] + real_vid[:, 1:, :3] * (1. - real_vid[:, 1:, -1:])
					real_img = torch.nn.functional.interpolate(real_vid[:, :, :3].flatten(0, 1), size=(256, 256), mode='bilinear',
															   align_corners=False, antialias=True)
					fake_img = torch.nn.functional.interpolate(fake_vid.flatten(0, 1), size=(256, 256), mode='bilinear', align_corners=False,
															   antialias=True)

				if self.loss_weight.l1 > 0:
					l1_loss = self.l1_loss(real_img, fake_img)
					loss_dict['l1'] = l1_loss
					training_stats.report('G_Loss/real/loss_l1', l1_loss)
				if self.loss_weight.lpips > 0:
					lpips_loss = self.lpips_loss(real_img, fake_img)
					loss_dict['lpips'] = lpips_loss
					training_stats.report('G_Loss/real/loss_lpips', lpips_loss)
				if self.loss_weight.mouth > 0:
					real_mouth = real_vid[:, :, :3].flatten(0, 1)
					fake_mouth = fake_vid.flatten(0, 1)
					real_mm = real_vid_mm.flatten(0, 1)
					real_mouth = torch.cat(
						[torch.nn.functional.interpolate(real_mouth[i:i + 1, :, m[0]:m[1], m[2]:m[3]], size=(64, 64), mode='bilinear', antialias=True)
						 for i, m in enumerate(real_mm)], dim=0)
					fake_mouth = torch.cat(
						[torch.nn.functional.interpolate(fake_mouth[i:i + 1, :, m[0]:m[1], m[2]:m[3]], size=(64, 64), mode='bilinear', antialias=True)
						 for i, m in enumerate(real_mm)], dim=0)
					mouth_loss = self.l2_loss(real_mouth, fake_mouth) + 0.5 * self.lpips_loss(real_mouth, fake_mouth)
					loss_dict['mouth'] = mouth_loss
					training_stats.report('G_Loss/real/loss_mouth', mouth_loss)
			# if self.loss_weight.id > 0:
				# 	id_loss = self.id_loss(real_vid[:, :, 3], fake_vid)
				# 	loss_dict['id'] = id_loss
				# 	training_stats.report('Loss/G/loss_id', id_loss)

		if phase in ['Ireal', 'Igen']:
			# Multi view; ramdom gen w may need supervised either
			loss = 0.0
			for key in loss_dict:
				loss += loss_dict[key] * self.loss_weight[key]
			loss.backward()

		if self.loss_weight.adv > 0:
			if phase.startswith('D'):
				# Generate real_img_raw; For Dual-Discriminator use
				gen_z = torch.randn([real_vid.shape[0], self.I.generator.z_dim], device=self.device)
				with torch.autograd.profiler.record_function('Dgen'):
					with torch.no_grad():  # 此处是在生成虚拟数据，没有梯度流过
						ws = self.I.generator.mapping(gen_z, real_vid_c[:, 0], truncation_psi=0.6, update_emas=False)
						gt_texture_feats = self.I.generator.texture_backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False,
																					   noise_mode='const')
						gt_static_feats = self.I.generator.backbone.synthesis(ws, cond_list=None, return_list=True, update_emas=False,
																		      noise_mode='const')
						ws = ws.unsqueeze(1).expand(-1, T, -1, -1).flatten(0, 1)
						gen_output = self.I.generator.synthesis_withTexture(ws,
												[feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in gt_texture_feats],
												real_vid_c.flatten(0, 1), {'uvcoords_image': real_vid_v.flatten(0, 1)}, noise_mode='const',
												static_feats=[feat.unsqueeze(1).expand(-1, T, -1, -1, -1).flatten(0, 1) for feat in gt_static_feats])
						real_vid = gen_output['image'].unflatten(0, (-1, T))
						I_output = self.run_G_seq(real_vid_c, real_vid_v, real_vid_uv, ws=None, real_vid_frm=real_vid[:, :, :3], real_vid_e4e=None,
												  gt_texture_feats=gt_texture_feats if optimize_staticTri else None,
												  gt_static_feats=gt_static_feats if optimize_texture else None, only_image=False)

					gen_logits = self.run_D({'image_raw': I_output['feature_image'][:, :3], 'image': I_output['image']}, real_vid_c.flatten(0, 1)*0)
					training_stats.report('D_Loss/scores/fake', gen_logits)
					loss_Dgen = torch.nn.functional.softplus(gen_logits)
					loss_Dgen.mean().backward()

					real_img_tmp = {'image': gen_output['image'][:, :3].detach().requires_grad_(True),
									'image_raw': gen_output['feature_image'][:, :3].detach().requires_grad_(True)}
					real_logits = self.run_D(real_img_tmp, real_vid_c.flatten(0, 1)*0)
					training_stats.report('D_Loss/scores/real', real_logits)
					loss_Dreal = torch.nn.functional.softplus(-real_logits)
					training_stats.report('D_Loss/loss', loss_Dgen + loss_Dreal)

					if phase == 'Dreg':
						with conv2d_gradfix.no_weight_gradients():
							r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']],
														   create_graph=True, only_inputs=True)

							r1_grads_image = r1_grads[0]
							r1_grads_image_raw = r1_grads[1]
						r1_penalty = r1_grads_image.square().sum([1, 2, 3]) + r1_grads_image_raw.square().sum([1, 2, 3])
						loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
						loss_Dr1 = loss_Dr1.mean()
						training_stats.report('D_Loss/r1_penalty', r1_penalty)
						training_stats.report('D_Loss/reg', loss_Dr1)

						loss_Dreal += loss_Dr1.mul(gain)
					loss_Dreal.mean().backward()
#----------------------------------------------------------------------------
