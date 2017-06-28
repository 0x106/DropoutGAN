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

def value_surface(classifier, autoencoder, data_loader):

	# x = np.arange(-5, 5, 0.1)
	# y = np.arange(-5, 5, 0.1)
	# xx, yy = np.meshgrid(x, y, sparse=True)
	# z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
	# h = plt.contourf(x,y,z)
	# plt.show()

	# print(x.shape, y.shape, xx.shape, yy.shape, z.shape)

	N = 400

	x = np.linspace(-1, 1, num = N)
	y = np.linspace(-1, 1, num = N)
	xx, yy = np.meshgrid(x, y, sparse=True)

	grid = torch.FloatTensor(N*N,2)

	for i in range(N):
		for k in range(N):
			grid[i*N+k,0] = y[i]
			grid[i*N+k,1] = x[k]
	print("constructed grid")

	_, Z = autoencoder(Variable(grid))
	output = classifier(Variable(grid), (Z))

	output = (torch.round(output.view(N, N))).data.numpy()

	print("computed output")

	h = plt.contourf(x,y,output)
	plt.colorbar()
	# fig.colorbar(cs, ax=ax, shrink=0.9)

	x, _, _, targets = data_loader.next()
	# plt.plot(x[:,0].numpy(), x[:,1].numpy(), '+')
	for i in range(x.size(0)):
		if targets[i] < 0.5:
			plt.plot(x[i,0], x[i,1], '+', color='blue')
		else:
			plt.plot(x[i,0], x[i,1], '+', color='red')

	print("plotted points")

	plt.show()

	#
	# print(output.shape)
	#
	# # print(output.size())
	#
	# # img = imshow()
	#
	# # plt.plot(grid[:,0].numpy(), grid[:,1].numpy(), '+')
	# # plt.show()
