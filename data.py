import numpy as np
import torch
import os, sys, math
import torchvision.datasets as dset
import torchvision.transforms as transforms

def normalise(data, _new_min=-1.,_new_max=1):
	min_ = np.min(data)
	max_ = np.max(data)
	normalised =  _new_min + (((_new_max - _new_min) * (data - min_)) / (max_ - min_))
	return normalised#, min_, max_


class DataCircle():

	def __init__(self, opt):
		self.B = opt['B']

		# the two data circles
		self.P = torch.FloatTensor(self.B, 2).fill_(0)
		self.Q = torch.FloatTensor(self.B, 2).fill_(0)

		self.M = torch.FloatTensor(self.B*2, 2).fill_(0)
		self.T = torch.FloatTensor(self.B).fill_(0)
		self.targets = torch.FloatTensor(self.B*2,2).fill_(0)

	def next(self):

		theta = torch.rand(2, self.B) * 2. * math.pi
		alpha = 6. + torch.FloatTensor(2, self.B).normal_(0.,0.1)

		self.P[:,0] = -2. + alpha[0] * torch.sin(theta[0])
		self.P[:,1] = alpha[0] * torch.cos(theta[0])

		self.Q[:,0] = 2. + alpha[1] * torch.sin(theta[1])
		self.Q[:,1] = alpha[1] * torch.cos(theta[1])

		self.M[:self.B].copy_(self.P)
		self.M[self.B:].copy_(self.Q)

		self.P[:,0].copy_(torch.from_numpy(normalise(self.P[:,0].numpy())))
		self.P[:,1].copy_(torch.from_numpy(normalise(self.P[:,1].numpy())))
		self.M[:,0].copy_(torch.from_numpy(normalise(self.M[:,0].numpy())))
		self.M[:,1].copy_(torch.from_numpy(normalise(self.M[:,1].numpy())))

		self.T.fill_(0.)
		for i in range(self.B):
			if self.P[i,1] > 0.:
				self.T[i] = 1

		# self.T[:self.B].fill_(1.)
		# self.T[self.B:].fill_(0.)

		# indices = torch.LongTensor(self.B*2).copy_(torch.from_numpy(np.random.permutation(self.B*2)))

		# self.M = self.M[indices]
		# self.T = self.T[indices]

		# self.targets[:,0].copy_(self.T)
		# self.targets[:,1].copy_(1. - self.T)

		# return self.P.cuda(), self.Q.cuda(), self.M[:self.B].cuda(), self.targets[:self.B].cuda()
		return self.P, self.Q, self.M[:self.B], self.T#self.targets[:self.B]


class Data():

	def __init__(self, opt):
		self.B = opt['B']

		# the two data circles
		self.P = torch.FloatTensor(self.B, 2).fill_(0)
		self.Q = torch.FloatTensor(self.B, 2).fill_(0)

		self.M = torch.FloatTensor(self.B*2, 2).fill_(0)
		self.T = torch.FloatTensor(self.B*2).fill_(0)
		self.targets = torch.FloatTensor(self.B*2,2).fill_(0)

	def next(self):

		self.P.normal_(0,1)
		self.Q.normal_(0,1)

		self.P[:,0] += -4
		self.Q[:,0] += 4

		self.M[:self.B].copy_(self.P)
		self.M[self.B:].copy_(self.Q)

		self.M[:,0].copy_(torch.from_numpy(normalise(self.M[:,0].numpy())))
		self.M[:,1].copy_(torch.from_numpy(normalise(self.M[:,1].numpy())))

		# self.P[:,0].copy_(torch.from_numpy(normalise(self.P[:,0].numpy())))
		# self.P[:,1].copy_(torch.from_numpy(normalise(self.P[:,1].numpy())))

		self.T[:self.B].fill_(1.)
		self.T[self.B:].fill_(0.)

		indices = torch.LongTensor(self.B*2).copy_(torch.from_numpy(np.random.permutation(self.B*2)))

		self.M = self.M[indices]
		self.T = self.T[indices]

		self.targets[:,0].copy_(self.T)
		self.targets[:,1].copy_(1. - self.T)

		# return self.P.cuda(), self.Q.cuda(), self.M[:self.B].cuda(), self.targets[:self.B].cuda()
		return self.P, self.Q, self.M[:self.B], self.targets[:self.B]


class MNISTDataGenerator():

	def __init__(self, opt):

		self.B = opt['B']
		self.cuda = opt['gpu']

		data_path = '../data/mnist'
		if self.cuda:
			data_path = '/input'

		self.trData = dset.MNIST(data_path, train=True, download=True,
					   transform=transforms.Compose([
						#    transforms.ToPILImage(),
						   transforms.Scale(32),
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ]))



		self.N = len(self.trData)
		# self.trData = self.trData[:self.N]
		self.sample = torch.FloatTensor(self.B, 1, opt['dims'], opt['dims'])
		self.train_loader = torch.utils.data.DataLoader(self.trData, batch_size=self.B, shuffle=True)
		self.train_iter = iter(self.train_loader)

	def next(self):
		# sample comes from the train_loader
		try:
			self.sample, _ = self.train_iter.next()
		except:
			self.train_iter = iter(self.train_loader)
			self.sample, _ = self.train_iter.next()

		if self.sample.size(0) < self.B:
			self.train_iter = iter(self.train_loader)
			self.sample, _ = self.train_iter.next()

		return self.sample#.cuda()#.size()

		# return (self.sample.view(self.sample.size(0), self.sample.size(2)*self.sample.size(3)))#.cuda()#, self.local, self.point



class DataConditionalCircle():

	# if dataset == 'mixed-conditional-circle':
	# 	angles = torch.rand(B) * 2. * math.pi
	# 	lengths = 6. + ft(B).normal_(0,0.1)
	# 	sample[:,0] = lengths * torch.sin(angles)
	# 	sample[:,1] = lengths * torch.cos(angles)
	#
	# 	K = B // 4
	#
	# 	for i in range(4):
	# 		plength = lengths[np.random.randint(B)]
	# 		pangle = angles[np.random.randint(B)]
	# 		point[i*K:(i+1)*K,0].fill_((plength * np.sin(pangle)))
	# 		point[i*K:(i+1)*K,1].fill_((plength * np.cos(pangle)))
	#
	# 		langles = pangle + ft(K,1).normal_(0,0.2)
	# 		llengths = 6. + ft(K,1).normal_(0,0.1)
	# 		local[i*K:(i+1)*K,0] = llengths * torch.sin(langles)
	# 		local[i*K:(i+1)*K,1] = llengths * torch.cos(langles)


	def __init__(self, opt):
		self.B = opt['B']

		# the two data circles
		self.P = torch.FloatTensor(self.B, 2).fill_(0)
		self.Q = torch.FloatTensor(self.B, 2).fill_(0)

		self.M = torch.FloatTensor(self.B*2, 2).fill_(0)
		self.T = torch.FloatTensor(self.B*2).fill_(0)
		self.targets = torch.FloatTensor(self.B*2,2).fill_(0)

		self.local = torch.FloatTensor(self.B, 2).fill_(0)
		self.point = torch.FloatTensor(self.B, 2).fill_(0)

	def get_local(self):

		K = self.B

		theta = torch.rand(2, self.B) * 2. * math.pi
		alpha = 6. + torch.FloatTensor(2, self.B).normal_(0.,0.1)

		plength = alpha[0,np.random.randint(self.B)]
		pangle = theta[0,np.random.randint(self.B)]

		langles = pangle + torch.FloatTensor(K,1).normal_(0,0.2)
		llengths = 6. + torch.FloatTensor(K,1).normal_(0,0.1)
		self.local[:,0] = -2. + llengths * torch.sin(langles)
		self.local[:,1] = llengths * torch.cos(langles)

		return self.local

	def next(self):

		theta = torch.rand(2, self.B) * 2. * math.pi
		alpha = 6. + torch.FloatTensor(2, self.B).normal_(0.,0.1)

		K = self.B // 4
		for i in range(4):
			plength = alpha[0,np.random.randint(self.B)]
			pangle = theta[0,np.random.randint(self.B)]
			self.point[i*K:(i+1)*K,0].fill_((-2. + plength * np.sin(pangle)))
			self.point[i*K:(i+1)*K,1].fill_((plength * np.cos(pangle)))

			langles = pangle + torch.FloatTensor(K,1).normal_(0,0.2)
			llengths = 6. + torch.FloatTensor(K,1).normal_(0,0.1)
			self.local[i*K:(i+1)*K,0] = -2. + llengths * torch.sin(langles)
			self.local[i*K:(i+1)*K,1] = llengths * torch.cos(langles)


		self.P[:,0] = -2. + alpha[0] * torch.sin(theta[0])
		self.P[:,1] = alpha[0] * torch.cos(theta[0])

		self.Q[:,0] = 2. + alpha[1] * torch.sin(theta[1])
		self.Q[:,1] = alpha[1] * torch.cos(theta[1])

		self.M[:self.B].copy_(self.P)
		self.M[self.B:].copy_(self.Q)


		# self.P[:,0].copy_(torch.from_numpy(normalise(self.P[:,0].numpy())))
		# self.P[:,1].copy_(torch.from_numpy(normalise(self.P[:,1].numpy())))
		# self.M[:,0].copy_(torch.from_numpy(normalise(self.M[:,0].numpy())))
		# self.M[:,1].copy_(torch.from_numpy(normalise(self.M[:,1].numpy())))

		self.T[:self.B].fill_(1.)
		self.T[self.B:].fill_(0.)

		indices = torch.LongTensor(self.B*2).copy_(torch.from_numpy(np.random.permutation(self.B*2)))

		self.M = self.M[indices]
		self.T = self.T[indices]

		self.targets[:,0].copy_(self.T)
		self.targets[:,1].copy_(1. - self.T)

		# return self.P.cuda(), self.Q.cuda(), self.M[:self.B].cuda(), self.targets[:self.B].cuda()
		return self.P, self.Q, self.M[:self.B], self.targets[:self.B], self.local, self.point