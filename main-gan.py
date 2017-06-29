import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os,sys,math,time
import torchvision.utils as vutils
import torch.optim as optim
from torchvision import datasets, transforms
import data, mlp#, mmd

import matplotlib.pyplot as plt
# plt.ion()


def train_autoencoder(data_loader, autoencoder, criterion, optimiser):

	for epoch in range(opt['epochs']):

		sample, _, _, _, local, point = data_loader.next()

		autoencoder.zero_grad()

		y, z = autoencoder( Variable(sample) )
		loss = criterion( nn.Sigmoid()(y), nn.Sigmoid()(Variable(sample)) )

		loss.backward()

		# logs.append(loss.data[0])

		optimiser.step()

		plt.plot(sample[:,0].numpy(), sample[:,1].numpy(), '+')
		plt.plot(y[:,0].data.numpy(), y[:,1].data.numpy(), '+')
		# plt.plot(local[:,0].numpy(), local[:,1].numpy(), '+')
		# plt.plot(point[:,0].numpy(), point[:,1].numpy(), 'o')

		plt.pause(0.01)
		plt.clf()

	torch.save(autoencoder.state_dict(),opt['autoencoder_file'])

def train_adversarial(data_loader, autoencoder, generator, critic, optimiserD, optimiserG):

	clamp_lower, clamp_upper = -0.01, 0.01
	M = 100
	gen_iterations = 0

	noise = torch.FloatTensor(opt['B'], opt['nz'])
	input = torch.FloatTensor(opt['B'], opt['dims'])

	one = torch.FloatTensor([1])
	mone = one * -1

	logs = [[], [], []]

	for epoch in range(opt['epochs']):
		i = 0
		while i < M:
			for p in critic.parameters():
				p.requires_grad = True

			if gen_iterations < 25 or gen_iterations % 500 == 0:
				Diters = 100
			else:
				Diters = 5#opt.Diters
			j = 0
			while j < Diters and i < M:
				j += 1
				i += 1

				for p in critic.parameters():
					p.data.clamp_(clamp_lower, clamp_upper)

				critic.zero_grad()

				sample, _, _, _, local, point = data_loader.next()
				inputv = Variable(input.copy_(sample))
				errD_real = critic(inputv)
				errD_real.backward(one)

				noisev = Variable(noise.normal_(0, 1), volatile=True) # totally freeze netG
				fake = Variable(generator(noisev).data)
				errD_fake = critic(fake)
				errD_fake.backward(mone)

				errD = errD_real - errD_fake

				optimiserD.step()

			for p in critic.parameters():
				p.requires_grad = False # to avoid computation

			generator.zero_grad()

			# -------- train G real -------- #
			noisev = Variable(noise.normal_(0, 1))
			real = generator(noisev)
			inputv = real

			errG = critic(inputv)
			errG.backward(one)
			# ------------------------------ #

			optimiserG.step()
			gen_iterations += 1

			logs[0].append(errD.data[0])
			logs[1].append(errG.data[0])

			print(gen_iterations, errD.data[0], errG.data[0])

		plt.subplot(211)
		plt.plot(sample[:,0].numpy(), sample[:,1].numpy(), '+')
		plt.plot(real.data[:,0].numpy(), real.data[:,1].numpy(), '+')

		plt.subplot(212)
		plt.plot(logs[0])
		plt.plot(logs[1])

		plt.pause(0.01)
		plt.clf()

		# sys.exit()

if __name__ == '__main__':

	path = '/Users/jordancampbell/helix/phd/'
	# path = '/helix/GAN/'

	print(" ---- Setup ---- ")

	opt = { 'B': 200,
		'ng': 4,
		'J': 10,
		'M': 1,
		'nz': 100,          # input noise size
		'nh': 64,           # hidden dimensions
		'n_units': 100,
		# 'dims': 32,       # data dimensionality
		'dims': 2,          # data dimensionality
		'alpha': 0.9,       # friction term, eq. to momentum of 0.9
		'eta': 0.00005,       # learning rate
		'epochs': 200,
		'gpu': False,
		'mnist': False,
		'gpu': False,
		'data_path': path + 'DropoutGAN/data/mnist/',
		'output': path + 'DropoutGAN/output',
		'autoencoder_file': path + "DropoutGAN/models/gan-autoencoder-circle-v1-0.pth",
		'classifier_file':  path + "DropoutGAN/models/gan-classifier-circle-v1-0.pth",
		'experiment': 'mnist'
	}

	data_loader = data.DataConditionalCircle(opt)

	autoencoder = mlp.AutoEncoder(opt)
	AE_criterion = nn.BCELoss()
	AE_optimiser = optim.Adam(autoencoder.parameters(), lr=opt['eta'])

	# classifier = mlp.Classifier(opt)
	# CLF_criterion = nn.BCELoss()
	# CLF_optimiser = optim.Adam(classifier.parameters(), lr=opt['eta'])

	generator = mlp.Generator(opt['dims'], opt['nz'], 512, 0)
	critic = mlp.Critic(opt['dims'], opt['nz'], 512, 0)

	generator.apply(mlp.weights_init)
	critic.apply(mlp.weights_init)

	optimiserD = optim.RMSprop(critic.parameters(), lr = opt['eta'])
	optimiserG = optim.RMSprop(generator.parameters(), lr = opt['eta'])

	# train_autoencoder(data_loader, autoencoder, AE_criterion, AE_optimiser)
	train_adversarial(data_loader, autoencoder, generator, critic, optimiserD, optimiserG)













#
