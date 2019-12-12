"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
# from __future__ import print_function
# import argparse
# import os
import numpy as np
# import random
import torch
from torchvision.models.inception import inception_v3
from torch.nn import functional as F
import torch.nn as nn
from scipy.stats import entropy
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim as optim
# import torch.utils.data
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# import torchvision.utils as vutils
from torch.autograd import Variable
# from utils import weights_init, compute_acc
# from network import _netG, _netD, _netD_CIFAR10, _netG_CIFAR10
# from folder import ImageFolder
# from embedders import BERTEncoder
# from matplotlib import pyplot as plt

# cifar_text_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True, help='cifar10 | imagenet')
# parser.add_argument('--dataroot', required=True, help='path to dataset')
# parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
# parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
# parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
# parser.add_argument('--nz', type=int, default=200, help='size of the latent z vector')
# parser.add_argument('--ngf', type=int, default=64)
# parser.add_argument('--ndf', type=int, default=64)
# parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
# parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
# parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# parser.add_argument('--cuda', action='store_true', help='enables cuda')
# parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
# parser.add_argument('--netG', default='', help="path to netG (to continue training)")
# parser.add_argument('--netD', default='', help="path to netD (to continue training)")
# parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
# parser.add_argument('--manualSeed', type=int, help='manual seed')
# parser.add_argument('--embed_size', default=100, type=int, help='embed size')
# parser.add_argument('--ndsetum_classes', type=int, default=10, help='Number of classes for AC-GAN')
# parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')

# opt = parser.parse_args()
# print(opt)

# # specify the gpu id if using only 1 gpu
# if opt.ngpu == 1:
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

# try:
#     os.makedirs(opt.outf)
# except OSError:
#     pass

# if opt.manualSeed is None:
#     opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)
# if opt.cuda:
#     torch.cuda.manual_seed_all(opt.manualSeed)

# cudnn.benchmark = True

# if torch.cuda.is_available() and not opt.cuda:
#     print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# # datase t
# if opt.dataset == 'imagenet':
#     # folder dataset
#     dataset = ImageFolder(
#         root=opt.dataroot,
#         transform=transforms.Compose([
#             transforms.Scale(opt.imageSize),
#             transforms.CenterCrop(opt.imageSize),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]),
#         classes_idx=(10, 20)
#     )
# elif opt.dataset == 'cifar10':
#     dataset = dset.CIFAR10(
#         root=opt.dataroot, download=True,
#         transform=transforms.Compose([
#             transforms.Scale(opt.imageSize),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]))
# else:
#     raise NotImplementedError("No such dataset {}".format(opt.dataset))

# assert dataset
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
#                                          shuffle=True, num_workers=int(opt.workers))

# # some hyper parameters
# ngpu = int(opt.ngpu)
# nz = int(opt.nz)
# ngf = int(opt.ngf)
# ndf = int(opt.ndf)
# # num_classes = int(opt.num_classes)
# num_classes = 10
# nc = 3

# # Define the generator and initialize the weights
# if opt.dataset == 'imagenet':
#     netG = _netG(ngpu, nz)
# else:
#     netG = _netG_CIFAR10(ngpu, nz)
# netG.apply(weights_init)
# if opt.netG != '':
#     netG.load_state_dict(torch.load(opt.netG, map_location=torch.device('cpu')))
# print(netG)
# netG.eval()
# # Define the discriminator and initialize the weights
# if opt.dataset == 'imagenet':
#     netD = _netD(ngpu, num_classes)
# else:
#     netD = _netD_CIFAR10(ngpu, num_classes)
# netD.apply(weights_init)
# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
# print(netD)

# # loss functions
# dis_criterion = nn.BCELoss()
# aux_criterion = nn.NLLLoss()

# # tensor placeholders
# input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
# noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
# eval_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
# dis_label = torch.FloatTensor(opt.batchSize)
# aux_label = torch.LongTensor(opt.batchSize)
# real_label = 1
# fake_label = 0

# # if using cuda
# if opt.cuda:
#     netD.cuda()
#     netG.cuda()
#     dis_criterion.cuda()
#     aux_criterion.cuda()
#     input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()
#     noise, eval_noise = noise.cuda(), eval_noise.cuda()

# # define variables
# input = Variable(input)
# noise = Variable(noise)
# eval_noise = Variable(eval_noise)
# dis_label = Variable(dis_label)
# aux_label = Variable(aux_label)
# encoder = BERTEncoder()
# # noise for evaluation
# eval_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
# eval_label = np.random.randint(0, num_classes, opt.batchSize)
# if opt.dataset == 'cifar10':
#             captions = [cifar_text_labels[per_label] for per_label in eval_label]
#             embedding = encoder(eval_label, captions)
#             embedding = embedding.detach().numpy()
# eval_noise_[np.arange(opt.batchSize), :opt.embed_size] = embedding[:, :opt.embed_size]
# eval_noise_ = (torch.from_numpy(eval_noise_))
# eval_noise.data.copy_(eval_noise_.view(opt.batchSize, nz, 1, 1))

# # sampled image
# fake = netG(eval_noise)
# imgs = fake.reshape(opt.batchSize, 3, 32, 32).detach().numpy()

# fake = (fake + 1) / 2

# # displaying sampled image
# # print(fake.data)
# # print(cifar_text_labels[eval_label[0]])
# # t = fake.squeeze()
# # print(t.size())
# # print(t[:,:10,0])
# # t = t.permute(0,2,3,1)
# # t = t.detach()
# # print(t.size())
# # plt.imshow(t)
# # plt.show()
# print([cifar_text_labels[i] for i in eval_label])
# fake = fake.reshape(opt.batchSize, 3, 32, 32)
# grid = vutils.make_grid(fake, nrow = 5, padding = 0)
# plt.imshow(grid.permute(1,2,0).detach().numpy())
# plt.show()

# import torch
# from torch import nn
# from torch.autograd import Variable
# from torch.nn import functional as F
# import torch.utils.data

# from torchvision.models.inception import inception_v3

# import numpy as np
# from scipy.stats import entropy

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

# print ("Calculating Inception Score...")
# print (inception_score(imgs, cuda=False, batch_size=32, resize=True, splits=10))

# Inception Score (ACGAN 10000 samples) mean = 3.2023, std = 0.0611