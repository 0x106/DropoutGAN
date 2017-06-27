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

def value_surface(classifier, autoencoder):

	# x = np.arange(-5, 5, 0.1)
	# y = np.arange(-5, 5, 0.1)
	# xx, yy = np.meshgrid(x, y, sparse=True)
	# z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
	# h = plt.contourf(x,y,z)
	# plt.show()

	# print(x.shape, y.shape, xx.shape, yy.shape, z.shape)

	N = 200

	x = np.linspace(-1, 1, num = N)
	y = np.linspace(-1, 1, num = N)
	xx, yy = np.meshgrid(x, y, sparse=True)
	print(xx.shape, yy.shape)

	grid = torch.FloatTensor(N*N,2)

	for i in range(N):
		for k in range(N):
			grid[i*N+k,0] = y[i]
			grid[i*N+k,1] = x[k]

	_, Z = autoencoder(Variable(grid))
	output = classifier(Variable(grid), (Z))

	output = (output.view(N, N)).data.numpy()

	h = plt.contourf(x,y,output)
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
