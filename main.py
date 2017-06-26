import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os,sys,math,time
import torchvision.utils as vutils
import torch.optim as optim
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
	'n_units': 1000,
	# 'dims': 32,       # data dimensionality
	'dims': 2,          # data dimensionality
	'alpha': 0.9,       # friction term, eq. to momentum of 0.9
	'eta': 0.001,       # learning rate
	'epochs': 5000,
	'gpu': False,
	'mnist': False,
	'output': "/output/"}

print(opt)

opt['B'] = 400
data_loader = data.DataCircle(opt)

autoencoder = mlp.AutoEncoder(opt)
criterion = nn.MSELoss()
decode_criterion = nn.MSELoss()
optimiser = optim.Adam(autoencoder.parameters(), lr=opt['eta'])

decode_target = Variable(torch.FloatTensor(opt['B'],opt['n_units']).fill_(0.5))

classifier = mlp.Classifier(opt)
classifier_criterion = nn.BCELoss()
classifier_optimiser = optim.Adam(classifier.parameters(), lr=opt['eta'])

autoencoder.load_state_dict(torch.load("/Users/jordancampbell/helix/phd/DropoutGAN/models/autoencoder.pth"))
classifier.load_state_dict(torch.load("/Users/jordancampbell/helix/phd/DropoutGAN/models/dropout_classifier.pth"))

import exp
exp.value_surface(classifier, autoencoder)
sys.exit()

logs = []
for epoch in range(40):#opt['epochs']):

	X, _,_ , T = data_loader.next()

	_, Z = autoencoder(Variable(X))

	output = classifier(Variable(X), Variable(Z.detach().data))

	loss = classifier_criterion(output, Variable(T))
	# loss.backward()
	# classifier_optimiser.step()

	logs.append(loss.data[0])

	# print(output)

	plt.subplot(211)
	for i in range(X.size(0)):
		# if T[i] < 0.5:
			# plt.plot(X[i,0], X[i,1], 'o', color='blue')
		# else:
			# plt.plot(X[i,0], X[i,1], 'o', color='red')
		# print(output[i].data[0])
		if output[i].data[0] < 0.5:
			plt.plot(X[i,0], X[i,1], '+', color='blue')
		else:
			plt.plot(X[i,0], X[i,1], '+', color='red')

	plt.subplot(212)
	plt.plot(logs)


	plt.pause(0.1)
	plt.clf()

torch.save(classifier.state_dict(), "/Users/jordancampbell/helix/phd/DropoutGAN/models/dropout_classifier.pth")
sys.exit()

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

# train autoencoder
# logs = [[], [], []]
# for epoch in range(opt['epochs']):
#
# 	X, _,_ , _ = data_loader.next()
#
# 	Y, Z = autoencoder(Variable(X))
# 	loss = criterion(Y, Variable(X))
#
# 	decode_loss = decode_criterion(Z, decode_target)
#
# 	loss.backward()#retain_variables=True)
# 	# decode_loss.backward()
#
# 	logs[0].append(decode_loss.data[0])
# 	logs[1].append(loss.data[0])
# 	# print(Z.mean())
# 	logs[2].append(Z.mean().data[0])
#
# 	if epoch < 200:
# 		optimiser.step()
# 	else:
# 		break
#
# 	plt.subplot(211)
# 	plt.plot(X[:,0].numpy(), X[:,1].numpy(), '+')
# 	plt.plot(Y[:,0].data.numpy(), Y[:,1].data.numpy(), '+')
#
# 	plt.subplot(212)
# 	# plt.plot(logs[0])
# 	plt.plot(logs[1])
# 	# plt.plot(logs[2])
#
# 	plt.pause(0.01)
# 	plt.clf()
#
# torch.save(autoencoder.state_dict(), "/Users/jordancampbell/helix/phd/DropoutGAN/models/autoencoder.pth")
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
