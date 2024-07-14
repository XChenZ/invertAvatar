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

class Loss:
	def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
		raise NotImplementedError()

#----------------------------------------------------------------------------

class WLoss(Loss):
	def __init__(
		self, 
		device, 
		G, 
		D,
		WD,
		augment_pipe,
		r1_gamma,
		loss_weight,
		loss_path
	):
		
		super().__init__()
		self.device             = device
		self.I                  = G
		self.D                  = D
		self.w_D				= WD
		self.augment_pipe = augment_pipe
		self.r1_gamma = r1_gamma
		self.neural_rendering_resolution = self.I.generator.neural_rendering_resolution
		self.loss_weight = EasyDict(loss_weight)
		self.loss_path = EasyDict(loss_path)
		self.__init_loss()
		self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
		self.filter_mode = 'antialiased'

	def __init_loss(self):
		self.l1_loss = torch.nn.L1Loss(reduction='mean').eval()
		self.l2_loss = torch.nn.MSELoss(reduction='mean').eval()
		if self.loss_weight.lpips > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.loss_weight.id > 0:
			self.id_loss = IDLoss(self.loss_path.path_ir_se50).to(self.device).eval()
		if self.loss_weight.w_l1 > 0:
			self.w_l1_loss = torch.nn.SmoothL1Loss(reduction='mean').eval()
		if self.loss_weight.w_regular > 0:
			self.batch_latent_avg = self.I.latent_avg.unsqueeze(0).repeat(1, 14*2, 1)
		if self.loss_weight.w_discriminator > 0:
			# self.w_D = LatentCodesDiscriminator(512, 4).to(self.device)
			# self.w_D_opt = torch.optim.Adam(list(self.w_D.parameters()), lr=2e-5)
			self.real_w_pool = LatentCodesPool(pool_size=50)
			self.fake_w_pool = LatentCodesPool(pool_size=50)

	def run_I_gen(self, z, c, v, neural_rendering_resolution, update_emas=False, only_w=False, truncation_psi=1):
		with torch.no_grad():
			ws = self.I.generator.mapping(z, c, truncation_psi=truncation_psi, update_emas=update_emas)
			gen_output = self.I.generator.synthesis(ws, c, v, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
		I_output = self.I(gen_output['image'], c, v, only_w=only_w)	# 原版用的是生成的c
		return I_output, ws, gen_output

	def run_D(self, img, c, update_emas=False):
		logits = self.D(img, c, update_emas=update_emas)
		return logits

	def accumulate_gradients(self, phase, real_img, real_w, real_c, real_v, gen_z, gen_c, gen_v, gain, cur_nimg):
		wd_r1_gamma = 10
		iter_num = 1 #(cur_nimg//100) % 3 + 1 #np.random.randint(1, 4)	# 随机1，2，3次
		assert phase in ['Igen', 'Ireal', 'Ireal_novel', 'WDmain', 'WDreg']
		real_Dcond_img = real_img[:, 3:]
		# real_img = {'image': real_img[:, :3]}
		real_img = {'image': torch.nn.functional.interpolate(real_img[:, :3], size=(256, 256), mode='bilinear', align_corners=False, antialias=True)}
		loss_dict = {}
		if phase == 'Igen':
			with torch.autograd.profiler.record_function('Igen'):
				if False:	#self.loss_weight.w_l1 > 0:
					I_output, gen_w, _ = self.run_I_gen(gen_z, gen_c, gen_v, self.neural_rendering_resolution, only_w=True, truncation_psi=0.6)
					loss_w_l1 = self.w_l1_loss(gen_w, I_output['w']) * 10 #+ self.w_l1_loss(gen_c, I_output['c'])
					training_stats.report('Loss/G/loss_w_l1', loss_w_l1)
					loss_dict['w_l1'] = loss_w_l1
				else:
					I_output, gen_w, gen_output = self.run_I_gen(gen_z, gen_c, gen_v, self.neural_rendering_resolution, only_w=False, truncation_psi=0.6)
				I_output['image'] = torch.nn.functional.interpolate(I_output['image'], size=(256, 256), mode='bilinear', align_corners=False,
																	antialias=True)
				gen_output['image'] = torch.nn.functional.interpolate(gen_output['image'], size=(256, 256), mode='bilinear', align_corners=False,
																	antialias=True)

				if self.loss_weight.w_delta > 0:
					first_w = I_output['w'][:, 0:1, :]
					delta = I_output['w'][:, 1:, :] - first_w
					delta_loss = torch.norm(delta, 2, dim=2).mean()
					loss_dict['w_delta'] = delta_loss
					training_stats.report('Loss/G/loss_gen_w_delta', delta_loss)
				if self.loss_weight.w_regular > 0:
					loss_w_regular = self.l2_loss(self.batch_latent_avg.expand(gen_c.shape[0], -1, -1), I_output['w'])
					loss_dict['w_regular'] = loss_w_regular
					training_stats.report('Loss/G/loss_gen_w_regular', loss_w_regular)
				if True: # Igen不使用image-level loss, 节省运算时间
					if self.loss_weight.l2 > 0:
						l2_loss = self.l2_loss(gen_output['image'], I_output['image'])
						loss_dict['l2'] = l2_loss
						training_stats.report('Loss/G/loss_gen_l2', l2_loss)
					if self.loss_weight.lpips > 0:
						lpips_loss = self.lpips_loss(gen_output['image'], I_output['image'])
						loss_dict['lpips'] = lpips_loss
						training_stats.report('Loss/G/loss_gen_lpips', lpips_loss)
					if self.loss_weight.id > 0:
						id_loss = self.id_loss(gen_output['image'], I_output['image'])
						loss_dict['id'] = id_loss
						training_stats.report('Loss/G/loss_gen_id', id_loss)

		if phase == 'Ireal':
			with torch.autograd.profiler.record_function('Ireal'):
				with torch.no_grad():
					I_output = self.I.generator.synthesis(real_w, real_c, real_v)
					I_output['w'] = real_w
					# I_output = self.I(real_img['image'], real_c, real_v)
					I_output['image'] = torch.nn.functional.interpolate(I_output['image'], size=(256, 256), mode='bilinear', align_corners=False,
																		antialias=True)
					for k in range(iter_num-1):
						rec_img = I_output['image'].clone()
						init_ws = I_output['w'].clone()
						I_output = self.I.restyle_forward(torch.cat([real_img['image'], rec_img], dim=1), real_c, real_v, init_ws)
						I_output['image'] = torch.nn.functional.interpolate(I_output['image'], size=(256, 256), mode='bilinear', align_corners=False, antialias=True)

				loss = 0.0
				rec_img = I_output['image'].clone().detach()
				init_ws = I_output['w'].clone().detach()
				I_output = self.I.restyle_forward(torch.cat([real_img['image'], rec_img], dim=1), real_c, real_v, init_ws)
				I_output['image'] = torch.nn.functional.interpolate(I_output['image'], size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
				k = iter_num - 1
				if self.loss_weight.l2 > 0:
					l2_loss = self.l2_loss(real_img['image'], I_output['image'])
					loss_dict['l2'] = l2_loss
					training_stats.report('Loss/G/loss_l2_iter{:d}'.format(k), l2_loss)
				if self.loss_weight.lpips > 0:
					lpips_loss = self.lpips_loss(real_img['image'], I_output['image'])
					loss_dict['lpips'] = lpips_loss
					training_stats.report('Loss/G/loss_lpips_iter{:d}'.format(k), lpips_loss)
				if self.loss_weight.id > 0:
					id_loss = self.id_loss(real_img['image'], I_output['image'])
					loss_dict['id'] = id_loss
					training_stats.report('Loss/G/loss_id_iter{:d}'.format(k), id_loss)
				if self.loss_weight.w_delta > 0:
					delta = I_output['w'] - init_ws
					delta_loss = torch.norm(delta, 2, dim=2).mean()
					loss_dict['w_delta'] = delta_loss
					training_stats.report('Loss/G/loss_w_delta', delta_loss)

				for key in loss_dict:
					loss += loss_dict[key] * self.loss_weight[key]
				loss.backward()

		if phase == 'Ireal_novel':
			if self.loss_weight.multiview_id > 0 or self.loss_weight.multiview_cx > 0:
				# To forward encoder again to save cuda memory
				loss_dict = {}

				I_output = self.I(real_img['image'], real_c, real_v, only_w=True)
				multiview_image = self.I.generator.synthesis(I_output['w'], gen_c, real_v)['image']

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

		if phase in ['WDmain', 'WDboth']:
			with torch.autograd.profiler.record_function('Dgen_forward'):
				I_output = self.I(real_img['image'], real_c, real_v, only_w=True)
				fake_w = self.fake_w_pool.query(I_output['w'])
				fake_pred = self.w_D(fake_w)
				loss_Dgen = torch.nn.functional.softplus(fake_pred).mean()
				loss_Dgen.backward()

		if phase in ['WDmain', 'WDreg', 'WDboth']:
			name = 'WDreal' if phase == 'WDmain' else 'WDr1' if phase == 'WDreg' else 'WDreal_Dr1'
			with torch.autograd.profiler.record_function(name + '_forward'):
				with torch.no_grad():
					real_w = self.I.generator.mapping(gen_z, gen_c, truncation_psi=0.8)
				real_w = self.real_w_pool.query(real_w)
				real_pred = self.w_D(real_w)

				loss_Dreal = 0
				if phase in ['WDmain', 'WDboth']:
					loss_Dreal = torch.nn.functional.softplus(-real_pred).mean()
					training_stats.report('Loss/WD/loss', loss_Dgen + loss_Dreal)

				loss_Dr1 = 0
				if phase in ['WDreg', 'WDboth']:
					real_w = real_w.detach()
					real_w.requires_grad = True
					real_pred = self.w_D(real_w)
					grad_real, = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_w, create_graph=True)
					r1_loss = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
					loss_Dr1 = wd_r1_gamma / 2 * r1_loss * gain + 0 * real_pred[0]
					training_stats.report('Loss/WD/r1_penalty', r1_loss)

			with torch.autograd.profiler.record_function(name + '_backward'):
				(loss_Dreal + loss_Dr1).backward()

		loss_Dgen = 0
		if False:#self.loss_weight.adv > 0:
			real_img_raw = filtered_resizing(real_img['image'], size=self.neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)
			real_img = {'image': real_img['image'], 'image_raw': real_img_raw}

			if phase == 'Dmain':
				# Generate real_img_raw; For Dual-Discriminator use
				with torch.autograd.profiler.record_function('Dgen'):
					with torch.no_grad():
						I_output = self.I(real_img['image'])
					gen_logits = self.run_D({'image_raw': I_output['image_raw'], 'image': I_output['image']}, real_c)
					training_stats.report('Loss/scores/fake', gen_logits)
					training_stats.report('Loss/signs/fake', gen_logits.sign())
					loss_Dgen = torch.nn.functional.softplus(gen_logits).mean()
					loss_Dgen.backward()

			# # Dmain: Maximize logits for real images.
			# # Dr1: Apply R1 regularization.
			if phase in ['Dmain', 'Dreg']:
				name = 'Dreal' if phase == 'Dmain' else 'Dr1'
				with torch.autograd.profiler.record_function(name):
					real_img_tmp_image = real_img['image'].detach().requires_grad_(phase == 'Dreg')
					real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase == 'Dreg')
					real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

					real_logits = self.run_D(real_img_tmp, real_c)
					training_stats.report('Loss/scores/real', real_logits)
					training_stats.report('Loss/signs/real', real_logits.sign())

					loss_Dreal = 0
					if phase == 'Dmain':
						loss_Dreal = torch.nn.functional.softplus(-real_logits).mean()
						training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

					loss_Dr1 = 0
					if phase == 'Dreg':	#  如果固定D的话，为何要计算Dreg？？？？？
						with conv2d_gradfix.no_weight_gradients():
							r1_grads = torch.autograd.grad(
								outputs=[real_logits.sum()],
								inputs=[real_img_tmp['image'], real_img_tmp['image_raw']],
								create_graph=True,
								only_inputs=True
							)
							r1_grads_image = r1_grads[0]
							r1_grads_image_raw = r1_grads[1]
						r1_penalty = r1_grads_image.square().sum([1, 2, 3]) + r1_grads_image_raw.square().sum([1, 2, 3])
						loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
						loss_Dr1 = loss_Dr1.mean()
						training_stats.report('Loss/r1_penalty', r1_penalty)
						training_stats.report('Loss/D/reg', loss_Dr1)

					(loss_Dreal + loss_Dr1).mul(gain).backward()

#----------------------------------------------------------------------------
