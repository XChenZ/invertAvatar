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

#----------------------------------------------------------------------------

class WLoss:
	def __init__(
		self, 
		device, 
		G,
		D,
		augment_pipe,
		r1_gamma,
		loss_weight,
		loss_path,
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

	def run_I_gen(self, z, c, v, neural_rendering_resolution, gen_condImg=None, gen_uv=None, update_emas=False, truncation_psi=1, mask_output=False,
				  return_feats=False):
		with torch.no_grad():  # 此处是在生成虚拟数据，没有梯度流过
			ws = self.I.generator.mapping(z, c, truncation_psi=truncation_psi, update_emas=update_emas)
			gen_output = self.I.generator.synthesis(ws, c, v, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas, return_featmap=True)
			if gen_condImg is not None:	# mask
				gen_output['image'] = torch.cat([gen_output['image'], gen_condImg], dim=1)
		I_output = self.I({'image': gen_output['image'], 'uv': gen_uv}, c, v, return_feats=return_feats)	# 原版用的是生成的c
		if mask_output and gen_condImg is not None:
			I_output['image'] = gen_output['image'][:, :3] * gen_condImg + I_output['image'] * (1. - gen_condImg)  # only non-face
			lr_mask = torch.nn.functional.interpolate(gen_condImg, size=(128, 128), mode='bilinear', align_corners=False, antialias=True)
			I_output['feature_image'] = gen_output['feature_image'] * lr_mask + I_output['feature_image'] * (1. - lr_mask)  # only non-face

		return I_output, ws, gen_output

	def run_D(self, img, c, update_emas=False):
		logits = self.D(img, c, update_emas=update_emas)
		return logits

	def accumulate_gradients(self, phase, real_img, real_c, real_v, gen_z, gen_c, gen_v, gen_condImg, gen_uv, gain, cur_nimg):
		wd_r1_gamma = 10
		mask_output = False
		assert phase in ['Igen', 'Ireal', 'Dgen', 'Dreal', 'Dgen_reg', 'Dreal_reg'], phase
		# real_Dcond_img = real_img[:, 3:]
		# real_img = {'image': real_img[:, :3]}
		# real_img = {'image': torch.nn.functional.interpolate(real_img[:, :3], size=(256, 256), mode='bilinear', align_corners=False, antialias=True)}
		# real_img['image'] = torch.nn.functional.interpolate(real_img['image'], size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
		loss_dict = {}
		if phase == 'Igen':
			with torch.autograd.profiler.record_function('Igen'):
				I_output, gen_w, gen_output = self.run_I_gen(gen_z, gen_c, gen_v, self.neural_rendering_resolution, gen_condImg, gen_uv,
															 truncation_psi=0.8, mask_output=mask_output, return_feats=self.loss_weight.texture > 0)

				if self.loss_weight.adv > 0:
					gen_logits = self.run_D({'image_raw': I_output['feature_image'][:, :3], 'image': torch.cat([I_output['image']], dim=1)}, gen_c*0)
					loss_adv = torch.nn.functional.softplus(-gen_logits).mean()
					loss_dict['adv'] = loss_adv
					training_stats.report('G_Loss/gen/loss_adv', loss_adv)

				I_output['image'] = torch.nn.functional.interpolate(I_output['image'], size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
				gen_output['image'] = torch.nn.functional.interpolate(gen_output['image'], size=(256, 256), mode='bilinear', align_corners=False, antialias=True)

				# Image Level
				if self.loss_weight.l1 > 0:
					# l2_loss = self.l2_loss(gen_output['image'][:, :3], I_output['image'])
					l1_loss = F.l1_loss(gen_output['image'][:, :3], I_output['image'])
					loss_dict['l1'] = l1_loss
					training_stats.report('G_Loss/gen/loss_gen_l1', l1_loss * self.loss_weight['l1'])
				if self.loss_weight.lpips > 0:
					lpips_loss = self.lpips_loss(gen_output['image'][:, :3], I_output['image'])
					loss_dict['lpips'] = lpips_loss
					training_stats.report('G_Loss/gen/loss_lpips', lpips_loss * self.loss_weight['lpips'])

				if self.loss_weight.raw_l1 > 0:
					raw_l1_loss = F.l1_loss(gen_output['feature_image'], I_output['feature_image'])
					loss_dict['raw_l1'] = raw_l1_loss
					training_stats.report('G_Loss/gen/loss_gen_raw_l1', raw_l1_loss * self.loss_weight['raw_l1'])
				if self.loss_weight.tri > 0:
					tri_l1_loss = F.l1_loss(gen_output['triplane'], I_output['triplane'])
					# print('loss_weight', torch.abs(gen_output['triplane']).max().item(), torch.abs(I_output['triplane']).max().item(),
					# 	  torch.abs(gen_output['triplane'] - I_output['triplane']).max().item())
					loss_dict['tri'] = tri_l1_loss
					training_stats.report('G_Loss/gen/loss_gen_tri_l1', tri_l1_loss * self.loss_weight['tri'])
				if self.loss_weight.texture > 0:
					texture_l1_loss = sum([F.l1_loss(gen_output['texture'][i], I_output['texture'][i]) for i in range(len(gen_output['texture']))])
					loss_dict['texture'] = texture_l1_loss
					training_stats.report('G_Loss/gen/loss_gen_texture_l1', texture_l1_loss * self.loss_weight['texture'])
				if self.loss_weight.lr_lpips > 0:
					lr_lpips_loss = self.lpips_loss(gen_output['feature_image'][:, :3], I_output['feature_image'][:, :3])
					loss_dict['lr_lpips'] = lr_lpips_loss
					training_stats.report('G_Loss/gen/loss_lr_lpips', lr_lpips_loss * self.loss_weight['lr_lpips'])

				if self.loss_weight.id > 0:
					id_loss = self.id_loss(gen_output['image'][:, :3], I_output['image'])
					loss_dict['id'] = id_loss
					training_stats.report('G_Loss/gen/loss_id', id_loss * self.loss_weight['id'])

		if phase == 'Ireal':
			with torch.autograd.profiler.record_function('Ireal'):
				I_output = self.I(real_img, real_c, real_v) # 原版用的是生成的c
				real_img_raw = filtered_resizing(real_img['image'][:, :3], size=128, f=self.resample_filter, filter_mode=self.filter_mode)
				if self.loss_weight.adv > 0:
					gen_logits = self.run_D({'image_raw': I_output['feature_image'][:, :3], 'image': torch.cat([I_output['image']], dim=1)}, gen_c * 0)
					loss_adv = torch.nn.functional.softplus(-gen_logits).mean()
					loss_dict['adv'] = loss_adv
					training_stats.report('G_Loss/real/loss_adv', loss_adv)

				# Image Level
				if real_img['image'].shape[1] > 3:
					I_output['image'] = I_output['image'] * real_img['image'][:, -1:] + real_img['image'][:, :3] * (1. - real_img['image'][:, -1:])	# only face
					# fake_img = real_img['image'][:, :3] * real_img['image'][:, -1:] + I_output['image'] * (1. - real_img['image'][:, -1:])	# only non-face
				fake = torch.nn.functional.interpolate(I_output['image'], size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
				real = torch.nn.functional.interpolate(real_img['image'][:, :3], size=(256, 256), mode='bilinear', align_corners=False, antialias=True)

				if self.loss_weight.l1 > 0:
					# l2_loss = self.l2_loss(real_img['image'], fake_img)
					l1_loss = F.l1_loss(real, fake)
					loss_dict['l1'] = l1_loss
					training_stats.report('G_Loss/real/loss_l1', l1_loss)
				if self.loss_weight.lpips > 0:
					lpips_loss = self.lpips_loss(real, fake)
					loss_dict['lpips'] = lpips_loss
					training_stats.report('G_Loss/real/loss_lpips', lpips_loss)
				if self.loss_weight.raw_l1 > 0:
					raw_l1_loss = F.l1_loss(real_img_raw, I_output['feature_image'][:, :3])
					loss_dict['raw_l1'] = raw_l1_loss
					training_stats.report('G_Loss/real/loss_gen_raw_l1', raw_l1_loss * self.loss_weight['raw_l1'])
				if self.loss_weight.lr_lpips > 0:
					lr_lpips_loss = self.lpips_loss(real_img_raw, I_output['feature_image'][:, :3])
					loss_dict['lr_lpips'] = lr_lpips_loss
					training_stats.report('G_Loss/real/loss_lr_lpips', lr_lpips_loss * self.loss_weight['lr_lpips'])
				if self.loss_weight.id > 0:
					id_loss = self.id_loss(real_img['image'][:, :3], I_output['image'])
					loss_dict['id'] = id_loss
					training_stats.report('G_Loss/real/loss_id', id_loss)

		if phase in ['Ireal', 'Igen']:
			# Multi view; ramdom gen w may need supervised either
			loss = 0.0		
			for key in loss_dict:
				loss += loss_dict[key] * self.loss_weight[key]
			loss.backward()

		if phase == 'Ireal_novel':
			if self.loss_weight.multiview_id > 0 or self.loss_weight.multiview_cx > 0:
				# To forward encoder again to save cuda memory
				loss_dict = {}
				ws = self.I.encode(real_img['image'])
				multiview_image = self.I.generator.synthesis(ws, gen_c, real_v)['image']

				# I_output_multiview = self.I(multiview_image, real_c, real_v, only_w=True)
				# loss_w_l1 = self.w_l1_loss(I_output_multiview['w'], I_output['w']) * 10
				# training_stats.report('Loss/G/loss_multiview_w_l1', loss_w_l1)
				# loss_dict['w_l1'] = loss_w_l1

				if self.loss_weight.multiview_id > 0:
					loss_multiview_id = self.id_loss(real_img['image'], multiview_image)
					loss_dict['multiview_id'] = loss_multiview_id
					training_stats.report('Loss/G/loss_multiview_id', loss_multiview_id)

				loss = 0.0
				for key in loss_dict:
					loss += loss_dict[key] * self.loss_weight[key]
				loss.backward()

		if self.loss_weight.adv > 0:
			if phase.startswith('D'):
				# Generate real_img_raw; For Dual-Discriminator use
				with torch.autograd.profiler.record_function('Dmain'):
					if phase.startswith('Dgen'):
						with torch.no_grad():
							I_output, _, gen_output = self.run_I_gen(gen_z, gen_c, gen_v, self.neural_rendering_resolution, gen_condImg, gen_uv,
																	 truncation_psi=0.8, mask_output=mask_output)
						real_img_tmp = {'image': gen_output['image'][:, :3].detach().requires_grad_(True),
										'image_raw': gen_output['feature_image'][:, :3].detach().requires_grad_(True)}
					elif phase.startswith('Dreal'):
						with torch.no_grad():
							I_output = self.I(real_img, real_c, real_v)  # 原版用的是生成的c
						real_img_raw = filtered_resizing(real_img['image'][:, :3], size=128, f=self.resample_filter, filter_mode=self.filter_mode)
						real_img_tmp = {'image': real_img['image'][:, :3].detach().requires_grad_(True),
										'image_raw': real_img_raw.detach().requires_grad_(True)}

					gen_logits = self.run_D({'image_raw': I_output['feature_image'][:, :3], 'image': I_output['image']}, gen_c*0)
					training_stats.report('%s/scores/fake' % phase.split('_')[0], gen_logits)
					loss_Dgen = torch.nn.functional.softplus(gen_logits)
					loss_Dgen.mean().backward()

					real_logits = self.run_D(real_img_tmp, gen_c*0)
					training_stats.report('%s/scores/real' % phase.split('_')[0], real_logits)
					loss_Dreal = torch.nn.functional.softplus(-real_logits)
					training_stats.report('%s/loss' % phase.split('_')[0], loss_Dgen + loss_Dreal)

					if phase.endswith('reg'):
						with conv2d_gradfix.no_weight_gradients():
							r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']],
														   create_graph=True, only_inputs=True)

							r1_grads_image = r1_grads[0]
							r1_grads_image_raw = r1_grads[1]
						r1_penalty = r1_grads_image.square().sum([1, 2, 3]) + r1_grads_image_raw.square().sum([1, 2, 3])
						loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
						loss_Dr1 = loss_Dr1.mean()
						training_stats.report('%s/r1_penalty' % phase.split('_')[0], r1_penalty)
						training_stats.report('%s/reg' % phase.split('_')[0], loss_Dr1)

						loss_Dreal += loss_Dr1.mul(gain)
					loss_Dreal.mean().backward()
#----------------------------------------------------------------------------
