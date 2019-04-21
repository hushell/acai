# coding: utf-8

import argparse
import itertools
import os
import random
#import ujson
#from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter # import tensorboard
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from visualization import make_sample_grid_and_save
from model import Encoder, Decoder, Discriminator


##########################################################################
## Configs
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist', help='cifar10 | lsun | mnist')
parser.add_argument('--dataroot', type=str, default='./datasets', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=64, help='size of the batches')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--gpu_id', type=int, default=3, help='which GPU')
parser.add_argument('--resumeDir', type=str, help='Resume directory')
parser.add_argument('--outDir', type=str, default = 'outDir', help='output model directory')
parser.add_argument('--scale', type=int, default=3, help='nb of downsampling in the autoencoder')
parser.add_argument('--depth', type=int, default = 16, help='depth in the autoencoder')
parser.add_argument('--latent', type=int, default = 2, help='depth in the autoencoder')
parser.add_argument('--reg', type=float, default = 0.2, help='hyper parameter lambda in the paper')
parser.add_argument('--sweight', type=float, default = 0.5, help='weight of sketch AE loss')
parser.add_argument('--manualSeed', type=int, default=100, help='manual seed')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)
cudnn.benchmark = True
device = torch.device("cuda:0" if opt.gpu_id >= 0 else "cpu")

if not os.path.exists(opt.outDir) :
	os.mkdir(opt.outDir)

##########################################################################
## Data
if opt.dataset == 'lsun':
    dataset = dset.LSUN(root=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


##########################################################################
## Networks
discriminator = Discriminator(opt.scale, opt.depth, opt.latent, nc).to(device)
encoder = Encoder(opt.scale, opt.depth, opt.latent, nc).to(device)
decoder = Decoder(opt.scale, opt.depth, opt.latent, nc).to(device)

if opt.resumeDir:
    msg = 'Loading pretrained model from {}'.format(opt.resumeDir)
    print (msg)
    discriminator.load_state_dict( torch.load(os.path.join(opt.resumeDir, 'discriminator.pth')) )
    encoder.load_state_dict( torch.load(os.path.join(opt.resumeDir, 'encoder.pth')) )
    decoder.load_state_dict( torch.load(os.path.join(opt.resumeDir, 'decoder.pth')) )

# Loss
MSE = torch.nn.functional.mse_loss


##########################################################################
## Optimizers & LR schedulers
optimizerG = torch.optim.Adam(itertools.chain(encoder.parameters(),
                                              decoder.parameters()),
                              lr=opt.lr, betas=(0.5, 0.999), weight_decay=1e-5)
optimizerD = torch.optim.Adam(itertools.chain(discriminator.parameters()),
                              lr=opt.lr, betas=(0.5, 0.999), weight_decay=1e-5)

#lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, factor=0.5, verbose=True, mode='max')


##########################################################################
## Training
writer = SummaryWriter(opt.outDir)

for epoch in range(opt.nEpochs):
    GAE = np.zeros(len(dataloader))
    GReg = np.zeros(len(dataloader))
    G = np.zeros(len(dataloader))
    DFit = np.zeros(len(dataloader))
    DMix = np.zeros(len(dataloader))
    D = np.zeros(len(dataloader))

    for i, batch in enumerate(dataloader):
        # Set model input
        image = batch[0].to(device)

	# Autoencoder with interpolation
        z = encoder(image)
        image_hat = decoder(z)
        alpha = torch.rand(image.shape[0], 1, 1, 1, device=device)#.expand_as(z)
        z_mix = alpha * z + (1 - alpha) * z.flip(0)
        image_alpha = decoder(z_mix)
        disc_alpha = discriminator(image_alpha)

        loss_ae = MSE(image_hat, image)
        loss_reg = torch.mean(disc_alpha**2)
        lossG = loss_ae + loss_reg * opt.sweight

        optimizerG.zero_grad()
        lossG.backward(retain_graph=True)
        optimizerG.step()

	# Discriminator
        #disc_alpha = discriminator(image_alpha)

        image_reg = image_alpha + opt.reg * (image - image_alpha)
        disc_reg = discriminator(image_reg)

        loss_fit = MSE(disc_alpha.squeeze(), alpha.squeeze())
        loss_mix = torch.mean(disc_reg**2)
        lossD = loss_fit + loss_mix

        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()

        # Training log
        GAE[i] = loss_ae.item()
        GReg[i] = loss_reg.item()
        G[i] = lossG.item()
        DFit[i] = loss_fit.item()
        DMix[i] = loss_mix.item()
        D[i] = lossD.item()

        # Print information
        if i % 100 == 99 :
            msg = '\nEpoch {:d}, Batch {:d} ## G {:.4f}, [G/AE {:.4f}], [G/Reg {:.4f}] ## D {:.4f}, [D/fitting {:.4f}], [D/Mix {:.4f}]'.format(
                    epoch, i + 1,
                    G[:i+1].mean(), GAE[:i+1].mean(), GReg[:i+1].mean(),
                    D[:i+1].mean(), DFit[:i+1].mean(), DMix[:i+1].mean())
            print (msg)

    # Save train loss for one epoch
    writer.add_scalar('G/AE', np.mean(GAE), epoch)
    writer.add_scalar('G/Reg', np.mean(GReg), epoch)
    writer.add_scalar('G/G', np.mean(G), epoch)
    writer.add_scalar('D/Fit', np.mean(DFit), epoch)
    writer.add_scalar('D/Mix', np.mean(DMix), epoch)
    writer.add_scalar('D/D', np.mean(D), epoch)

    # Save images
    linearInterImg, sphereInterImg = make_sample_grid_and_save(encoder, decoder, dataloader)
    writer.add_image('val/LinearInterpolation', linearInterImg, epoch)
    writer.add_image('val/SphereInterpolation', sphereInterImg, epoch)

    # Save checkpoint
    torch.save(discriminator.state_dict(), os.path.join(opt.outDir, 'discriminator.pth'))
    torch.save(encoder.state_dict(), os.path.join(opt.outDir, 'encoder.pth'))
    torch.save(decoder.state_dict(), os.path.join(opt.outDir, 'decoder.pth'))

