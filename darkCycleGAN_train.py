# -*- coding: utf-8 -*-
# CycleGAN with darknet53 discriminator implementation on Pytorch
import torch
#import numpy as np
#import time
from torch.autograd import Variable
#from visdom import Visdom #not compatible with colab

import os
import csv
from torch.utils.data import DataLoader
import itertools
from models_and_utils import *

#print('cur pth:',os.getcwd())

# variables
epoch = 0                                             # starting epoch
n_epochs = 200                                        # total number of epochs for training
batch_sz = 4                                          # batch size
dataroot = 'data'                                     # path to data
saveroot = 'weights/try1'                             # path to save directory
lossfile = 'losses/mcg_losses1.csv'                   # path to save training losses in csv file
lr = 0.0002                                           # initial learning rate
decay_epoch = 100                                     # linearly decay lr after this epoch
crop_sz = 128                                         # input size of data crop
in_ch = 3                                             # input number of channels
out_ch = 3                                            # output number of channels
cuda_op = False                                       # use GPU option
n_cpu = 8                                             # number of cpu threads used during batch generation

assert (epoch>=0), 'Starting epoch must be >0!'

print('Training Variables:\n-------------------')
print('epoch:{} \nn_epochs:{} \nbatch_sz:{} \ndataroot:{} \nsaveroot:{} \nlossfile:{} \nlr:{} \ndecay_epoch:{} \ncrop_sz:{} \nin_ch:{} \nout_ch:{} \ncuda_:{} \nn_cpu:{}'.format(epoch, n_epochs, batch_sz, dataroot, saveroot, lossfile, lr, decay_epoch, crop_sz, in_ch, out_ch, cuda_op, n_cpu))
print('-------------------')
if not os.path.exists(saveroot):
  os.makedirs(saveroot)
  print('Created "saveroot" directory: ',saveroot)
if not os.path.exists('losses'):
  os.makedirs('losses')
  print('Created "losses" directory')
  
#Init networks
G_rain2clear = Generator(in_ch, out_ch)
G_clear2rain = Generator(in_ch, out_ch)
D_rain = DarkDiscriminator(in_ch)
D_clear = DarkDiscriminator(in_ch)

#print('Generator:\n',G_rain2clear)
#print('\nDiscriminator:\n', D_rain)

if cuda_op:
  G_rain2clear.cuda()
  G_clear2rain.cuda()
  D_rain.cuda()
  D_clear.cuda()

if epoch==0:
  G_rain2clear.apply(weights_init_normal)
  G_clear2rain.apply(weights_init_normal)
  D_rain.apply(weights_init_normal)
  D_clear.apply(weights_init_normal)
  print('Initialized new set of weights')
else:
  mod_w = os.path.join(saveroot, 'G_rain2clear.pth')
  assert (os.path.exists(mod_w)), 'G_rain2clear pretrained weights do not exits!'
  G_rain2clear.load_state_dict(torch.load(mod_w))
  mod_w = os.path.join(saveroot, 'G_clear2rain.pth')
  assert (os.path.exists(mod_w)), 'G_clear2rain pretrained weights do not exits!'
  G_clear2rain.load_state_dict(torch.load(mod_w))
  mod_w = os.path.join(saveroot, 'D_rain.pth')
  assert (os.path.exists(mod_w)), 'D_rain pretrained weights do not exits!'
  D_rain.load_state_dict(torch.load(mod_w))
  mod_w = os.path.join(saveroot, 'D_clear.pth')
  assert (os.path.exists(mod_w)), 'D_clear pretrained weights do not exits!'
  D_clear.load_state_dict(torch.load(mod_w))
  print('Loaded pretrained weights from epoch {}'.format(epoch))

# Losses
GAN_loss = nn.MSELoss()
cycle_loss = nn.L1Loss()
identity_loss = nn.L1Loss()
dark_perception_loss = nn.MSELoss()

# Optimizers and LR schedulers
optim_G = torch.optim.Adam(itertools.chain(G_rain2clear.parameters(), G_clear2rain.parameters()),
                           lr=lr, betas=(0.5, 0.999))
optim_D_rain = torch.optim.Adam(D_rain.parameters(), lr=lr, betas=(0.5, 0.999))
optim_D_clear = torch.optim.Adam(D_clear.parameters(), lr=lr, betas=(0.5, 0.999))
lr_sh_G = torch.optim.lr_scheduler.LambdaLR(optim_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_sh_D_rain = torch.optim.lr_scheduler.LambdaLR(optim_D_rain, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_sh_D_clear = torch.optim.lr_scheduler.LambdaLR(optim_D_clear, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

# Inputs and Tensors
Tensor = torch.cuda.FloatTensor if cuda_op else torch.Tensor
in_rain = Tensor(batch_sz, in_ch, crop_sz, crop_sz)
in_clear = Tensor(batch_sz, out_ch, crop_sz, crop_sz)
target_real = Variable(Tensor(batch_sz).fill_(1.), requires_grad=False)
target_fake = Variable(Tensor(batch_sz).fill_(0.), requires_grad=False)

fake_clear_buffer = ReplayBuffer()
fake_rain_buffer = ReplayBuffer()

# Dataset loader
tfm = [ transforms.Resize(int(crop_sz*1.12), Image.BICUBIC), 
        transforms.RandomCrop(crop_sz), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(dataroot, transforms_=tfm, unaligned=True), 
                        batch_size=batch_sz, shuffle=True, num_workers=n_cpu)

with open(lossfile, 'w', newline='\n') as f:
  fwrite = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
  fwrite.writerow(['Epoch','Iteration','G','loss_identity_rain','loss_identity_clear','loss_GAN_rain2clear',
                   'loss_GAN_clear2rain','loss_cycle_RCR','loss_cycle_CRC','loss_G (total)',
                   'D_rain','loss_D_real_R','loss_D_fake_R','loss_D_rain (total)',
                   'D_clear','loss_D_real_C','loss_D_fake_C','loss_D_clear (total)'])

for ep in range(epoch, n_epochs):
  for i, batch in enumerate(dataloader):
    # set model inputs
    real_rain = Variable(in_rain.copy_(batch['rain']))
    real_clear = Variable(in_clear.copy_(batch['clear']))
    print('real rain shape', real_rain.shape)

    # Generators rain2clear and clear2rain
    optim_G.zero_grad()

    # Identity loss
    # G_rain2clear(clear) should equal clear if real clear is fed
    same_clear = G_rain2clear(real_clear)
    loss_identity_clear = identity_loss(same_clear, real_clear)*5.0
    # G_clear2rain(rain) should equal rain if real rain is fed
    same_rain = G_clear2rain(real_rain)
    loss_identity_rain = identity_loss(same_rain, real_rain)*5.0

    # GAN loss
    fake_clear = G_rain2clear(real_rain)
    pred_fake_clear = D_clear(fake_clear)
    print('pred clear shape', pred_fake_clear[3].shape, 'target real shape',target_real.shape)
    loss_GAN_rain2clear = GAN_loss(pred_fake_clear[3], target_real)
    fake_rain = G_clear2rain(real_clear)
    pred_fake_rain = D_rain(fake_rain)
    loss_GAN_clear2rain = GAN_loss(pred_fake_rain[3], target_real)

    # Cycle loss
    rec_rain = G_clear2rain(fake_clear)
    loss_cycle_RCR = cycle_loss(rec_rain, real_rain)*10.
    rec_clear = G_rain2clear(fake_rain)
    loss_cycle_CRC = cycle_loss(rec_clear, real_clear)*10.

    # Dark Perceptual loss
    #percep_loss_rain_256 = dark_perception_loss(pred_fake_rain[0], )
    #percep_loss_rain_512 = dark_perception_loss(pred_fake_rain[1], )
    #percep_loss_rain_1024 = dark_perception_loss(pred_fake_rain[2], )
    #percep_loss_clear_256 = dark_perception_loss(pred_fake_clear[0], )
    #percep_loss_clear_512 = dark_perception_loss(pred_fake_clear[1], )
    #percep_loss_clear_1024 = dark_perception_loss(pred_fake_clear[2], )

    # Total loss
    loss_G = loss_identity_rain + loss_identity_clear + loss_GAN_rain2clear + loss_GAN_clear2rain + loss_cycle_RCR + loss_cycle_CRC
    loss_G.backward()
    optim_G.step()

    # Discriminator D_rain
    optim_D_rain.zero_grad()

    # Real loss
    pred_real = D_rain(real_rain)
    loss_D_real_R = GAN_loss(pred_real[3], target_real)

    # Fake loss
    fake_rain = fake_rain_buffer.push_and_pop(fake_rain)
    pred_fake = D_rain(fake_rain.detach())
    loss_D_fake_R = GAN_loss(pred_fake[3], target_fake)

    # Total loss
    loss_D_rain = (loss_D_real_R + loss_D_fake_R)*0.5
    loss_D_rain.backward()
    optim_D_rain.step()

    # Discriminator D_clear
    optim_D_clear.zero_grad()

    # Real loss
    pred_real = D_clear(real_clear)
    loss_D_real_C = GAN_loss(pred_real[3], target_real)
        
    # Fake loss
    fake_clear = fake_clear_buffer.push_and_pop(fake_clear)
    pred_fake = D_clear(fake_clear.detach())
    loss_D_fake_C = GAN_loss(pred_fake[3], target_fake)

    # Total loss
    loss_D_clear = (loss_D_real_C + loss_D_fake_C)*0.5
    loss_D_clear.backward()
    optim_D_clear.step()

    # logger
    print('\nEpoch: {} of {}, Iter: {} of {}'.format(ep+1, n_epochs, i+1, len(dataloader)))
    print('G LOSS -> loss_id_clear: {}, loss_id_rain: {}, loss_GAN_R2C: {}, loss_GAN_C2R: {}, loss_cyc_RCR: {}, loss_cyc_CRC: {}, G_tot: {}'.format(loss_identity_clear, loss_identity_rain, loss_GAN_rain2clear, loss_GAN_clear2rain, loss_cycle_RCR, loss_cycle_CRC, loss_G))
    print('D rain LOSS -> loss_D_real_R: {}, loss_D_fake_R: {}, D_R_tot: {}'.format(loss_D_real_R, loss_D_fake_R, loss_D_rain))
    print('D clear LOSS -> loss_D_real_C: {}, loss_D_fake_C: {}, D_C_tot: {}'.format(loss_D_real_C, loss_D_fake_C, loss_D_clear))
    with open(lossfile, 'a', newline='\n') as f:
      fwrite = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
      fwrite.writerow([ep+1,i+1,'',loss_identity_rain.item(),loss_identity_clear.item(),loss_GAN_rain2clear.item(),
                       loss_GAN_clear2rain.item(),loss_cycle_RCR.item(),loss_cycle_CRC.item(),loss_G.item(),
                       '',loss_D_real_R.item(),loss_D_fake_R.item(),loss_D_rain.item(),
                       '',loss_D_real_C.item(),loss_D_fake_C.item(),loss_D_clear.item()])

  # Update lr
  lr_sh_G.step()
  lr_sh_D_rain.step()
  lr_sh_D_clear.step()

  # Save models checkpoints
  torch.save(G_rain2clear.state_dict(), os.path.join(saveroot,'G_rain2clear.pth'))
  torch.save(G_clear2rain.state_dict(), os.path.join(saveroot,'G_clear2rain.pth'))
  torch.save(D_rain.state_dict(), os.path.join(saveroot,'D_rain.pth'))
  torch.save(D_clear.state_dict(), os.path.join(saveroot,'D_clear.pth'))
