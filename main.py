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

def train_autoencoder(opt, data_loader, autoencoder, criterion, optimiser):
	logs = []
	for epoch in range(opt['epochs']):
		autoencoder.zero_grad()

		x, _, _, _ = data_loader.next()

		y, z = autoencoder( Variable(x) )
		loss = criterion( nn.Sigmoid()(y), nn.Sigmoid()(Variable(x)) )

		loss.backward()

		logs.append(loss.data[0])

		optimiser.step()

		if ((epoch+1)%10) == 0:
			plt.subplot(211)
			plt.plot(x[:,0].numpy(), x[:,1].numpy(), '+')
			plt.plot(y.data[:,0].numpy(), y.data[:,1].numpy(), '+')

			plt.subplot(212)
			plt.plot(logs)
			plt.pause(0.001)
			plt.clf()

			print(epoch, np.mean(logs[-100:]))

	torch.save(autoencoder.state_dict(), "/Users/jordancampbell/helix/phd/DropoutGAN/models/autoencoder-v1-0.pth")

def train_classifier():
	"""
		Principled Dropout Classifier.
		To be implemented.
	"""
	pass

def train_adversarial():
	pass


if __name__ == '__main__':

	print(" ---- Setup ---- ")

	opt = { 'B': 100,
		'ng': 4,
		'J': 10,
		'M': 1,
		'nz': 100,          # input noise size
		'nh': 64,           # hidden dimensions
		'n_units': 100,
		# 'dims': 32,       # data dimensionality
		'dims': 2,          # data dimensionality
		'alpha': 0.9,       # friction term, eq. to momentum of 0.9
		'eta': 0.001,       # learning rate
		'epochs': 200,
		'gpu': False,
		'mnist': False,
		'output': "/output/"
	}

	experiment = 'circle'

	if experiment == 'circle':
		opt['B'] = 100
		data_loader = data.DataCircle(opt)

		autoencoder = mlp.AutoEncoder(opt)

		criterion = nn.BCELoss()
		optimiser = optim.Adam(autoencoder.parameters(), lr=opt['eta'])

	train_autoencoder(opt, data_loader, autoencoder, criterion, optimiser)
	train_classifier()







#
