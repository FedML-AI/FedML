import torch
import torchvision
import numpy as np
from collections import defaultdict
import logging

from core.security.attack.attack_base import BaseAttackMethod

from core.security.attack import inversefed

"""
ref: Geiping, Jonas, et al. "Inverting gradients-how easy is it to break privacy in federated learning?." 
Advances in Neural Information Processing Systems 33 (2020): 16937-16947.
https://github.com/JonasGeiping/invertinggradients/

attack @ server, added by Kai, 07/07/2022
Steps:
(1)
(2)
(3)
"""


class InvertAttack(BaseAttackMethod):
	def __init__(self, trained_model=False, arch='ResNet18', dataset='CIFAR10', data_path='../data/cifar10', \
					training_strategy='conservative', num_images=1, img_shape=(3, 32, 32), save_files=False, use_updates=False):
		# System setup
		self.setup = inversefed.utils.system_startup()
		defs = inversefed.training_strategy(training_strategy)
		loss_fn, _, validloader =  inversefed.construct_dataloaders(dataset, defs, data_path=data_path)
		self.save_files = save_files
		self.use_updates = use_updates
		self.img_shape = img_shape
		
		self.arch = arch.lower()
		if self.arch == 'resnet18':
			self.model = torchvision.models.resnet18(pretrained=trained_model)
			self.model.to(**self.setup)
			self.model.eval();

		self.dataset = dataset.lower()
		if self.dataset == 'cifar10':
			self.dm = torch.as_tensor(inversefed.consts.cifar10_mean, **self.setup)[:, None, None]
			self.ds = torch.as_tensor(inversefed.consts.cifar10_std, **self.setup)[:, None, None]

		# Build the input (ground-truth) gradient
		# TODO: load it from clients
		self.num_images = num_images
		img = validloader.dataset
		if not use_updates:
			self.ground_truth, self.input_gradient, self.labels = \
				self.create_fake_input(self.num_images, img, self.model, loss_fn, save_files)
		else:
			self.ground_truth, self.input_parameters, self.labels = \
				self.create_fake_input(self.num_images, img, self.model, loss_fn, save_files)

	def create_fake_input(self, num_images, img, model, loss_fn, save_files):
		if num_images == 1:
			idx = 0
			image, label = img[idx]
			labels = torch.as_tensor((label,), device=self.setup['device'])
			ground_truth = image.to(**self.setup).unsqueeze(0)
			ground_truth_denormalized = torch.clamp(ground_truth * self.ds + self.dm, 0, 1)
			if save_files:
				torchvision.utils.save_image(ground_truth_denormalized, f'{self.idx}_{self.arch}_{self.dataset}_input.png')
		else:
			ground_truth, labels = [], []
			idx = 0
			while len(labels) < num_images:
				image, label = img[idx]
				idx += 1
				if label not in labels:
					labels.append(torch.as_tensor((label,), device=self.setup['device']))
					ground_truth.append(image.to(**self.setup))
			ground_truth = torch.stack(ground_truth)
			labels = torch.cat(labels)
		
		model.zero_grad()
		target_loss, _, _ = loss_fn(model(ground_truth), labels)
		
		if not self.use_updates:
			input_gradient = torch.autograd.grad(target_loss, model.parameters())
			input_gradient = [grad.detach() for grad in input_gradient]
			full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
			logging.info(f'Full gradient norm is {full_norm:e}.')
			return ground_truth, input_gradient, labels
		else:
			self.local_lr = 1e-4
			self.local_steps = 5
			input_parameters = inversefed.reconstruction_algorithms.loss_steps(model, ground_truth, labels, 
                                                        lr=self.local_lr, local_steps=self.local_steps, use_updates=self.use_updates)
			input_parameters = [p.detach() for p in input_parameters]
			return ground_truth, input_parameters, labels

	def attack(self, local_w, global_w, refs=None):
		if not self.use_updates:
			rec_machine = inversefed.GradientReconstructor(self.model, (self.dm, self.ds), config=refs, num_images=self.num_images)
			output, stats = rec_machine.reconstruct(self.input_gradient, self.labels, self.img_shape)
		else:
			rec_machine = inversefed.FedAvgReconstructor(self.model, (self.dm, self.ds), self.local_steps, self.local_lr, config=refs, use_updates=self.use_updates)
			output, stats = rec_machine.reconstruct(self.input_parameters, self.labels, self.img_shape)
		
		test_mse = (output.detach() - self.ground_truth).pow(2).mean()
		feat_mse = (self.model(output.detach())- self.model(self.ground_truth)).pow(2).mean()  
		test_psnr = inversefed.metrics.psnr(output, self.ground_truth, factor=1/self.ds)
		logging.info(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")
		
		# save reconstructed images
		output_denormalized = torch.clamp(output * self.ds + self.dm, 0, 1)
		if self.save_files and (self.num_images == 1):
			torchvision.utils.save_image(output_denormalized, f'{self.idx}_{self.arch}_{self.dataset}_output.png')

