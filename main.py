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

print(" ---- Setup ---- ")

opt = { 'B': 100,
	'ng': 4,
	'J': 10,
	'M': 1,
	'nz': 100,          # input noise size
	'nh': 64,           # hidden dimensions
	'n_units': 100,
	# 'dims': 32,       # data dimensionality
	'dims': 784,          # data dimensionality
	'alpha': 0.9,       # friction term, eq. to momentum of 0.9
	'eta': 0.001,       # learning rate
	'epochs': 10,
	'gpu': False,
	'mnist': False,
	'output': "/output/"}

print(opt)

opt['B'] = 100
data_loader = data.DataCircle(opt)

autoencoder = mlp.MNISTAutoEncoder(opt)
criterion = nn.BCELoss()
decode_criterion = nn.MSELoss()
optimiser = optim.Adam(autoencoder.parameters(), lr=opt['eta'])

decode_target = Variable(torch.FloatTensor(opt['B'],opt['n_units']).fill_(0.5))

classifier = mlp.MNISTClassifier(opt)
classifier_criterion = nn.CrossEntropyLoss()
classifier_optimiser = optim.Adam(classifier.parameters(), lr=opt['eta'])

# autoencoder.load_state_dict(torch.load("/Users/jordancampbell/helix/phd/DropoutGAN/models/autoencoder.pth"))
# classifier.load_state_dict(torch.load("/Users/jordancampbell/helix/phd/DropoutGAN/models/dropout_classifier.pth"))

mnist = torch.utils.data.DataLoader( datasets.MNIST('../data', train=True, download=True,
				   transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
				   ])), batch_size=opt['B'], shuffle=True)

# print(mnist)

data_iter = iter(mnist)
batch, _ = data_iter.next()
# print(len(batch), batch[0].size())

# print(batch[0])
# sys.exit()

# sys.exit()

# import exp
# exp.value_surface(classifier, autoencoder)
# sys.exit()

def train_autoencoder():
	logs = [[]]
	for epoch in range(opt['epochs']):
		data_iter = iter(mnist)
		for idx in range(len(data_iter)):

			autoencoder.zero_grad()

			X, _ = data_iter.next()

			Y, Z = autoencoder(Variable(X))
			loss = criterion(nn.Sigmoid()(Y), nn.Sigmoid()(Variable(X)))

			loss.backward()
			logs[0].append(loss.data[0])

			optimiser.step()

		print(epoch, np.mean(logs[0][-100:]))

		vutils.save_image(Y.data, '{}/real_samples-{}-{}.png'.format('/Users/jordancampbell/helix/phd/DropoutGAN/output/', epoch, idx), normalize=True)

		plt.plot(logs[0])
		plt.pause(0.01)
		plt.clf()

	torch.save(autoencoder.state_dict(), "/Users/jordancampbell/helix/phd/DropoutGAN/models/autoencoder-mnist.pth")
	sys.exit()

def train_classifier():

	autoencoder.load_state_dict(torch.load("/Users/jordancampbell/helix/phd/DropoutGAN/models/autoencoder-mnist.pth"))
	targets = Variable(torch.FloatTensor(opt['B'], 10))
	logs = []
	for epoch in range(opt['epochs']):
		data_iter = iter(mnist)
		for idx in range(len(data_iter)):

			classifier.zero_grad()

			X, T = data_iter.next()
			# targets.data.fill_(0)
			# targets[T] = 1.
			# print(targets[:4])
			# print(T.size())
			# sys.exit()
			# targets.data.copy_(T.float())

			_, Z = autoencoder(Variable(X))
			Z.data.fill_(1.)
			output = classifier(Variable(X), Variable(Z.detach().data))
			# print(targets)

			loss = classifier_criterion(output, Variable(T))
			loss.backward()
			classifier_optimiser.step()

			logs.append(loss.data[0])
		print(epoch, np.mean(logs[-100:]))

			# plt.subplot(211)
			# for i in range(X.size(0)):
			# 	if output[i].data[0] < 0.5:
			# 		plt.plot(X[i,0], X[i,1], '+', color='blue')
			# 	else:
			# 		plt.plot(X[i,0], X[i,1], '+', color='red')
			#
			# plt.subplot(212)
			# plt.plot(logs)
			#
			# plt.pause(0.1)
			# plt.clf()

	# torch.save(classifier.state_dict(), "/Users/jordancampbell/helix/phd/DropoutGAN/models/dropout_classifier-mixed-circles-01.pth")
	sys.exit()

# train_autoencoder()
train_classifier()

# Every configuration of dropped weights determines a unique network.
# Using dropout has a number of benefits, such as preventing over-fitting and
#   increasing model capacity.
# My hypothesis is that we can improve model performance if we map each input
#   to a unique configuration, then we can train the network by first sampling
#   from the posterior distribution of configurations given inputs.







#
# for epoch in range(opt['epochs']):
# 	X, _,_ , _ = data_loader.next()
# 	Y, Z = autoencoder(Variable(X))
# 	plt.plot(X[:,0].numpy(), X[:,1].numpy(), '+')
# 	plt.plot(Y[:,0].data.numpy(), Y[:,1].data.numpy(), '+')
# 	plt.pause(0.01)
# 	plt.clf()
#
# sys.exit()


# angle = 0.
# input = Variable(torch.FloatTensor(1,2).fill_(0))
# X = torch.FloatTensor(opt['B'],2)
# for i in range(100):
#
# 	# X, _,_ , _ = data_loader.next()
# 	theta = torch.FloatTensor(opt['B']).copy_(torch.from_numpy(np.linspace(0.,2.*math.pi,num=opt['B'])))
# 	X[:,0] = torch.sin(theta)
# 	X[:,1] = torch.cos(theta)
#
# 	Y,Z = autoencoder(Variable(X))
#
# 	plt.subplot(211)
# 	plt.plot(X[:,0].numpy(), X[:,1].numpy(), '+')
# 	plt.plot(Y[:,0].data.numpy(), Y[:,1].data.numpy(), '+')
# 	# plt.plot([X[:,0].numpy(), Y[:,0].data.numpy()], [X[:,1].numpy(), Y[:,1].data.numpy()])
#
# 	plt.subplot(212)
# 	z = [[] for k in range(10)]
# 	for k in range(10):
# 		# plt.plot(Z[:,k].data[0])
# 		for j in range(opt['B']):
# 			z[k].append(Z[j,k].data[0])
# 		plt.plot(z[k])
#
# 	# plt.plot(X[:,0].numpy(), X[:,1].numpy(), '+')
# 	# plt.plot(Z[:,np.random.randint(opt['n_units'])].data.numpy(), Z[:,np.random.randint(opt['n_units'])].data.numpy(), '+')
# 	# plt.plot(Z[:,np.random.randint(opt['n_units'])].data.numpy(), Z[:,np.random.randint(opt['n_units'])].data.numpy(), '+')
# 	# plt.plot(Z[:,np.random.randint(opt['n_units'])].data.numpy(), Z[:,np.random.randint(opt['n_units'])].data.numpy(), '+')
# 	# plt.plot(Z[:,np.random.randint(opt['n_units'])].data.numpy(), Z[:,np.random.randint(opt['n_units'])].data.numpy(), '+')
# 	# plt.plot(Z[:,np.random.randint(opt['n_units'])].data.numpy(), Z[:,np.random.randint(opt['n_units'])].data.numpy(), '+')
# 	# plt.plot(Z[:,np.random.randint(opt['n_units'])].data.numpy(), Z[:,np.random.randint(opt['n_units'])].data.numpy(), '+')
# 	# plt.plot(Z[:,np.random.randint(opt['n_units'])].data.numpy(), Z[:,np.random.randint(opt['n_units'])].data.numpy(), '+')
# #
# 	# plt.plot([X[:,0].numpy(), Z[:,0].data.numpy()], [X[:,1].numpy(), Z[:,1].data.numpy()])
#
# 	plt.pause(0.01)
# 	plt.clf()
#
# 	# angle += (2.*math.pi)/100
# 	# plt.pause(0.1)
# # plt.clf()
#
# # plt.show()






#
