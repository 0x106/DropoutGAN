import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import math, sys
import numpy as np

# def log_gaussian(x, mu, sigma):
#     return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)
#
# def log_gaussian_logsigma(x, mu, logsigma):
#     return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)


def weights_prior(network,opt):
    sigma = 1.
    likelihood = Variable(torch.FloatTensor(1).fill_(1.))
    if opt['gpu']:
        likelihood.data = likelihood.data.cuda()
    alpha = float(-0.5 * np.log(2 * math.pi) - np.log(np.abs(sigma)))
    for p in network.parameters():
        likelihood = likelihood + torch.sum(alpha - torch.pow(p,2) / (2. * sigma **2))
    likelihood = likelihood / network.num_weights
    return likelihood


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.1)#0.06)
        # m.bias.data.normal_(0, 1.)#0.06)
        m.bias.data.fill_(0)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
                nn.Linear(opt['nz'], opt['nh']),
                nn.BatchNorm1d(opt['nh'] ),
                nn.ReLU(),
                nn.Linear(opt['nh'], opt['nh']*2),
                nn.BatchNorm1d(opt['nh'] * 2),
                nn.ReLU(),
                nn.Linear(opt['nh']*2, opt['nh']*4),
                nn.BatchNorm1d(opt['nh'] * 4),
                nn.ReLU(),
                nn.Linear(opt['nh']*4, opt['dims']),
                # nn.Tanh()
        )
        self.num_weights = 0
        for l in self.main.parameters():
            if l.dim() == 1:
                self.num_weights += l.size(0)
            else:
                self.num_weights += l.size(0) * l.size(1)

    def forward(self, x):
        x = self.main(x)
        return x

    def clear_weights(self):
        for p in self.parameters():
            p.data.fill_(0.)

    def copy(self, mlp):
        for (src, dst) in zip(mlp.parameters(), self.parameters()):
            dst.data.copy_(src.data)

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
                        nn.Linear(opt['dims'], opt['nh']*4),
                        nn.ReLU(),
                        nn.Linear(opt['nh']*4, opt['nh']*2),
                        nn.BatchNorm1d(opt['nh'] * 2),
                        nn.ReLU(),
                        nn.Linear(opt['nh']*2, opt['nh']),
                        nn.BatchNorm1d(opt['nh']),
                        nn.ReLU(),
                        nn.Linear(opt['nh'], 1),
                        nn.Sigmoid()
                    )
        self.num_weights = 0
        for l in self.main.parameters():
            if l.dim() == 1:
                self.num_weights += l.size(0)
            else:
                self.num_weights += l.size(0) * l.size(1)

    def forward(self, x):
        # x = x.view(x.size(0), 28 * 28)
        x = self.main(x)

        return x

    def clear_weights(self):
        for p in self.parameters():
            p.data.fill_(0.)

    def copy(self, mlp):
        for (src, dst) in zip(mlp.parameters(), self.parameters()):
            dst.data.copy_(src.data)

class _netG(nn.Module):
    def __init__(self, opt):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     opt['nz'], opt['nh'] * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt['nh'] * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(opt['nh'] * 8, opt['nh'] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt['nh'] * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(opt['nh'] * 4, opt['nh'] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt['nh'] * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(opt['nh'] * 2,     1, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(opt['nh']),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # nn.ConvTranspose2d(    opt['nh'],      1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.num_weights = 0
        for l in self.main.parameters():
            if l.dim() == 1:
                self.num_weights += l.size(0)
            else:
                self.num_weights += l.size(0) * l.size(1)
    def forward(self, input):
        output = self.main(input)
        return output
        # return nn.parallel.data_parallel(self.main, input, None)

class _netD(nn.Module):
    def __init__(self, opt):
        super(_netD, self).__init__()
        self.main = nn.Sequential(

            # input is (nc) x 64 x 64
            # nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(1, opt['nh'] * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(opt['nh'] * 2, opt['nh'] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt['nh'] * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(opt['nh'] * 4, opt['nh'] * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt['nh'] * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(opt['nh'] * 8, 1, 4, 1, 0, bias=False),

            nn.Sigmoid()
        )
        self.num_weights = 0
        for l in self.main.parameters():
            if l.dim() == 1:
                self.num_weights += l.size(0)
            else:
                self.num_weights += l.size(0) * l.size(1)
    def forward(self, input):
        output = self.main(input)
        # output = nn.parallel.data_parallel(self.main, input, None)
        return output.view(-1, 1)

class GeneratorConditional(nn.Module):
    def __init__(self, opt):
        super(GeneratorConditional, self).__init__()

        self.main = nn.Sequential(
                        nn.Linear(opt['nz']+opt['dims'], opt['nh']),
                        nn.BatchNorm1d(opt['nh'] ),
                        nn.ReLU(),
                        nn.Linear(opt['nh'], opt['nh']*2),
                        nn.BatchNorm1d(opt['nh'] * 2),
                        nn.ReLU(),
                        nn.Linear(opt['nh']*2, opt['nh']*4),
                        nn.BatchNorm1d(opt['nh'] * 4),
                        nn.ReLU(),
                        nn.Linear(opt['nh']*4, opt['dims']),
                        # nn.Tanh()
                    )
        self.num_weights = 0
        for l in self.main.parameters():
            if l.dim() == 1:
                self.num_weights += l.size(0)
            else:
                self.num_weights += l.size(0) * l.size(1)

    def forward(self, x, cond):
        x = torch.cat((x,cond), 1)
        x = self.main(x)
        # x = x.view(x.size(0), 1, 28, 28)
        return x

    def clear_weights(self):
        for p in self.parameters():
            p.data.fill_(0.)

    def copy(self, mlp):
        for (src, dst) in zip(mlp.parameters(), self.parameters()):
            dst.data.copy_(src.data)

class AutoEncoder(nn.Module):
    def __init__(self, opt):
        super(AutoEncoder, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(opt['dims'], opt['nh']),
            nn.ReLU(True),
            nn.Linear(opt['nh'], opt['nh']*2),
            nn.BatchNorm1d(opt['nh']*2),
            nn.ReLU(True),
            nn.Linear(opt['nh']*2, opt['nh']*4),
            nn.BatchNorm1d(opt['nh']*4),
            nn.ReLU(True),
            nn.Linear(opt['nh']*4, opt['n_units']),
        )

        self.decode = nn.Sequential(
            nn.BatchNorm1d(opt['n_units']),
            nn.ReLU(True),
            nn.Linear(opt['n_units'], opt['nh']*4),
            nn.BatchNorm1d(opt['nh']*4),
            nn.ReLU(True),
            nn.Linear(opt['nh']*4, opt['nh']*2),
            nn.BatchNorm1d(opt['nh']*2),
            nn.ReLU(True),
            nn.Linear(opt['nh']*2, opt['nh']),
            nn.BatchNorm1d(opt['nh']),
            nn.ReLU(True),
            nn.Linear(opt['nh'], opt['dims']),
            nn.Tanh()
        )

        self.num_weights = 0
        for l in self.encode.parameters():
            if l.dim() == 1:
                self.num_weights += l.size(0)
            else:
                self.num_weights += l.size(0) * l.size(1)
        for l in self.decode.parameters():
            if l.dim() == 1:
                self.num_weights += l.size(0)
            else:
                self.num_weights += l.size(0) * l.size(1)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, nn.Sigmoid()(encoded) # decoded should match input, encoded are the
                                #   dropout probabilities.

class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(opt['dims'], opt['nh'])
        self.fc2 = nn.Linear(opt['nh'], opt['n_units'])
        self.fc3 = nn.Linear(opt['n_units'], 1)

    def forward(self, x, probs, is_training=False):

        x = nn.ReLU(True)(self.fc1(x))
        x = nn.ReLU(True)( self.dropout( self.fc2(x), probs, is_training ) )
        x = nn.Sigmoid()(self.fc3(x))

        return x

    def dropout(self, layer, probs, is_training):

        if is_training:

            probs = torch.round(probs) * 0.9 + 0.05

            drop = torch.bernoulli(torch.round(probs))
            layer = drop * layer
            layer = layer * (1./(torch.sum(drop)/probs.size(0))).data[0]

        return layer

class MNISTClassifier(nn.Module):
    def __init__(self, opt):
        super(MNISTClassifier, self).__init__()

        self.dims = opt['dims']
        self.lr = opt['eta']

        self.fc1 = nn.Linear(self.dims * self.dims, opt['nh']*12)
        self.fc2 = nn.Linear(opt['nh']*12, opt['nh']*10)
        self.fc3 = nn.Linear(opt['nh']*10, opt['nh']*4)
        self.fc4 = nn.Linear(opt['nh']*4, opt['n_units'])
        self.fc5 = nn.Linear(opt['n_units'], 10)

    def setup(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, probs, is_training=False):
        x = x.view(x.size(0), self.dims * self.dims)
        x = nn.ReLU(True)(self.fc1(x))
        x = nn.ReLU(True)(self.fc2(x))
        x = nn.ReLU(True)(self.fc3(x))
        x = nn.ReLU(True)( self.dropout( self.fc4(x), probs, is_training) )
        x = nn.Sigmoid()(self.fc5(x))

        return x

    def criterion(self, x, t):
        return self.criterion(x, t)

    def optimiser(self):
        self.optimiser.step()

    def dropout(self, layer, probs, is_training):

        if is_training:
            # probs = torch.round(probs) * 0.9 + 0.05

            drop = torch.bernoulli(torch.round(probs))
            layer = drop * layer
            # layer = layer * (1./(torch.sum(drop)/probs.size(0))).data[0]

        return layer

class MNISTAutoEncoder(nn.Module):
    def __init__(self, opt):
        super(MNISTAutoEncoder, self).__init__()

        self.dims = opt['dims']

        self.encode = nn.Sequential(
            nn.Linear(opt['dims']*opt['dims'], opt['nh']),
            nn.ReLU(True),
            nn.Linear(opt['nh'], opt['nh']*2),
            nn.BatchNorm1d(opt['nh']*2),
            nn.ReLU(True),
            nn.Linear(opt['nh']*2, opt['nh']*4),
            nn.BatchNorm1d(opt['nh']*4),
            nn.ReLU(True),
            nn.Linear(opt['nh']*4, opt['n_units']),
        )

        self.decode = nn.Sequential(
            nn.BatchNorm1d(opt['n_units']),
            nn.ReLU(True),
            nn.Linear(opt['n_units'], opt['nh']*4),
            nn.BatchNorm1d(opt['nh']*4),
            nn.ReLU(True),
            nn.Linear(opt['nh']*4, opt['nh']*2),
            nn.BatchNorm1d(opt['nh']*2),
            nn.ReLU(True),
            nn.Linear(opt['nh']*2, opt['nh']),
            nn.BatchNorm1d(opt['nh']),
            nn.ReLU(True),
            nn.Linear(opt['nh'], opt['dims']*opt['dims']),
            # nn.Tanh()
        )

        self.num_weights = 0
        for l in self.encode.parameters():
            if l.dim() == 1:
                self.num_weights += l.size(0)
            else:
                self.num_weights += l.size(0) * l.size(1)
        for l in self.decode.parameters():
            if l.dim() == 1:
                self.num_weights += l.size(0)
            else:
                self.num_weights += l.size(0) * l.size(1)

    def forward(self, x):
        x = x.view(x.size(0), self.dims*self.dims)
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded.view(x.size(0), 1, self.dims, self.dims), nn.Sigmoid()(encoded) # decoded should match input, encoded are the
                                #   dropout probabilities.
