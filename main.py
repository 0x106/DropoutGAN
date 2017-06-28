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

	resp = ''
	if os.path.isfile(opt['autoencoder_file']):
		print('Do you want to overwrite the autoencoder?')
		print('File:', opt['autoencoder_file'])
		resp = raw_input('[y/n] --> ')

	if resp == 'n':
		print('Exiting.')
		return

	logs = []
	for epoch in range(opt['epochs']):
		autoencoder.zero_grad()

		x, _, _, _ = data_loader.next()

		y, z = autoencoder( Variable(x) )
		loss = criterion( nn.Sigmoid()(y), nn.Sigmoid()(Variable(x)) )

		loss.backward()

		logs.append(loss.data[0])

		optimiser.step()

		if opt['experiment'] == 'mnist':
			if ((epoch+1)%100) == 0:
				vutils.save_image(y.data, '{}/real_samples-{}.png'.format(opt['output'], epoch), normalize=True)
		else:
			if ((epoch+1)%10) == 0:
				plt.subplot(211)
				plt.plot(x[:,0].numpy(), x[:,1].numpy(), '+')
				plt.plot(y.data[:,0].numpy(), y.data[:,1].numpy(), '+')

				plt.subplot(212)
				plt.plot(logs)
				plt.pause(0.001)
				plt.clf()

		print(epoch, np.mean(logs[-100:]))

	torch.save(autoencoder.state_dict(),opt['autoencoder_file'])

def misc_test_classifier(opt, data_loader, autoencoder, classifier, criterion, optimiser):

	# resp = ''
	# if os.path.isfile(opt['classifier_file']):
	# 	print('Do you want to overwrite the classifier?')
	# 	print('File:', opt['classifier_file'])
	# 	resp = input('[y/n] --> ')
	#
	# if resp == 'n':
	# 	print('Exiting.')
	# 	return

	logs0 = [[], []]
	logs1 = [[], []]
	for epoch in range(opt['epochs']):
		classifier[0].zero_grad()
		classifier[1].zero_grad()

		x, _, _, targets = data_loader.next()

		y, z = autoencoder( Variable(x) )

		# z.data.fill_(0.5)
		output0 = classifier[0](Variable(x), Variable(z.detach().data), is_training=True)

		# z.data.normal_(0,0.1)
		# z = nn.Sigmoid()(z)
		# z.data.fill_(1.)
		output1 = classifier[1](Variable(x), Variable(z.detach().data), is_training=True)

		loss0 = criterion(output0, Variable(targets))
		loss1 = criterion(output1, Variable(targets))

		loss0.backward()
		loss1.backward()

		optimiser[0].step()
		optimiser[1].step()

		# output0 = classifier[0](Variable(x), Variable(z.detach().data))
		output1 = classifier[1](Variable(x), Variable(z.detach().data))

		predictions0 = torch.round(output0.data)
		correct0 = predictions0.eq(targets)[:,0]

		predictions1 = torch.round(output1.data)
		correct1 = predictions1.eq(targets)[:,0]

		logs0[0].append(loss0.data[0])
		logs0[1].append(correct0.sum() / output0.size(0))

		logs1[0].append(loss1.data[0])
		logs1[1].append(correct1.sum() / output1.size(0))

		if ((epoch+1)%10) == 0:
			# plt.subplot(311)
			# # plt.plot(x[:,0].numpy(), x[:,1].numpy(), '+')
			# for i in range(x.size(0)):
			# 	if correct0[i]:
			# 		plt.plot(x[i,0], x[i,1], '+', color='blue')
			# 	# else:
			# 		# plt.plot(x[i,0], x[i,1], '+', color='red')
			#
			# plt.subplot(312)
			# # plt.plot(x[:,0].numpy(), x[:,1].numpy(), '+')
			# for i in range(x.size(0)):
			# 	if correct1[i]:
			# 		plt.plot(x[i,0], x[i,1], '+', color='blue')
				# else:
					# plt.plot(x[i,0], x[i,1], '+', color='red')

			plt.subplot(211)
			plt.plot(logs0[0], label = '0-l')
			plt.plot(logs1[0], label = '0-a')

			plt.subplot(212)
			plt.plot(logs0[1], label = '1-l')
			plt.plot(logs1[1], label = '1-a')
			# plt.legend()
			plt.pause(0.001)
			plt.clf()

		print(epoch, np.mean(logs0[0][-100:]), np.mean(logs0[1][-100:]), np.mean(logs1[0][-100:]), np.mean(logs1[1][-100:]))
	# torch.save(classifier[0].state_dict(),opt['classifier_file'])

def train_classifier(opt, data_loader, autoencoder, classifier, criterion, optimiser):

	resp = ''
	if os.path.isfile(opt['classifier_file']):
		print('Do you want to overwrite the classifier?')
		print('File:', opt['classifier_file'])
		resp = input('[y/n] --> ')

	if resp == 'n':
		print('Exiting.')
		return

	logs = [[], []]
	for epoch in range(opt['epochs']):
		classifier.zero_grad()

		x, _, _, targets = data_loader.next()
		y, z = autoencoder( Variable(x) )

		z.data.fill_(1.)
		output = classifier(Variable(x), Variable(z.detach().data), is_training=True)

		loss = criterion(output, Variable(targets))

		loss.backward()

		optimiser.step()

		a,b = test_classifier(opt, data_loader, autoencoder, classifier, criterion)
		logs[0].append(a)
		logs[1].append(b)
		plt.plot(logs[0])
		plt.plot(logs[1])
		plt.pause(0.01)
		plt.clf()

		# logs[0].append(loss.data[0])
		# prediction = output.data.max(1)[1]
		# correct = prediction[:,0].eq(targets).sum()
		# logs[1].append(float(correct) / opt['B'])

		# print(output.data[0])
		# print(prediction[0,0], targets[0])
		# print(prediction.size(), targets.size())

		print(epoch, np.mean(logs[0][-100:]), np.mean(logs[1][-100:]))


	# torch.save(classifier[0].state_dict(),opt['classifier_file'])


def test_classifier(opt, data_loader, autoencoder, classifier, criterion):

	# logs = [[], []]
	for epoch in range(opt['epochs']):

		x, _, _, targets = data_loader.next_test()
		y, z = autoencoder( Variable(x) )

		z.data.fill_(1.)
		output = classifier(Variable(x), Variable(z.detach().data), is_training=True)

		loss = criterion(output, Variable(targets))

		# logs[0].append(loss.data[0])
		prediction = output.data.max(1)[1]
		correct = prediction[:,0].eq(targets).sum()
		# logs[1].append(float(correct) / opt['B'])
#
		# print(epoch, np.mean(logs[0][-100:]), np.mean(logs[1][-100:]))
		return loss.data[0], float(correct) / opt['B']


	# torch.save(classifier[0].state_dict(),opt['classifier_file'])


def train_adversarial():
	pass

if __name__ == '__main__':

	print(" ---- Setup ---- ")

	train_AE_CLF = False
	if len(sys.argv) > 1:
		if sys.argv[1] == 'train-AE-CLF':
			train_AE_CLF = True

	opt = { 'B': 1000,
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
		'epochs': 5000,
		'gpu': False,
		'mnist': False,
		'gpu': False,
		'data_path': '/helix/GAN/DropoutGAN/data/mnist/',
		'output': '/helix/GAN/DropoutGAN/output',
		'autoencoder_file':  "/helix/GAN/DropoutGAN/models/autoencoder-mnist-v1-0.pth",
		'classifier_file':  "/helix/GAN/DropoutGAN/models/classifier-mnist-v1-0.pth",
		'experiment': 'mnist'
	}

	if opt['experiment'] == 'circle':
		# opt['B'] = 1000
		data_loader = data.DataCircle(opt)

		autoencoder = mlp.AutoEncoder(opt)
		AE_criterion = nn.BCELoss()
		AE_optimiser = optim.Adam(autoencoder.parameters(), lr=opt['eta'])

		classifier = [mlp.Classifier(opt), mlp.Classifier(opt)]
		CLF_criterion = nn.BCELoss()
		CLF_optimiser = [optim.Adam(classifier[0].parameters(), lr=opt['eta']), optim.Adam(classifier[1].parameters(), lr=opt['eta'])]

	if opt['experiment'] == 'mnist':

		opt['B'] = 100
		opt['dims'] = 28
		# opt['dims'] = 28
		data_loader = data.MNISTDataGenerator(opt)

		autoencoder = mlp.MNISTAutoEncoder(opt)
		AE_criterion = nn.BCELoss()
		AE_optimiser = optim.Adam(autoencoder.parameters(), lr=opt['eta'])

		classifier = mlp.MNISTClassifier(opt)
		CLF_criterion = nn.CrossEntropyLoss()
		CLF_optimiser = optim.Adam(classifier.parameters(), lr=opt['eta'])

	autoencoder.load_state_dict(torch.load(opt['autoencoder_file']))
	# classifier[0].load_state_dict(torch.load(opt['classifier_file']))

	if train_AE_CLF:
		# train_autoencoder(opt, data_loader, autoencoder, AE_criterion, AE_optimiser)
		train_classifier(opt, data_loader, autoencoder, classifier, CLF_criterion, CLF_optimiser)

	# import exp
	# exp.value_surface(classifier[0], autoencoder, data_loader)






#
