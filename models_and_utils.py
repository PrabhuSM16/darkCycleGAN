# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
from PIL import Image
import glob
import random
import os

######################### MODELS #########################
# Standard resblock
class ResidualBlock(nn.Module):
  def __init__(self, in_features):
    super(ResidualBlock, self).__init__()
    conv_block = [  nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features, in_features, 3),
                    nn.InstanceNorm2d(in_features),
                    nn.ReLU(inplace=True),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_features, in_features, 3),
                    nn.InstanceNorm2d(in_features)  ]
    self.conv_block = nn.Sequential(*conv_block)

  def forward(self, x):
    return x + self.conv_block(x)

# CycleGAN generator
class Generator(nn.Module):
  def __init__(self, input_nc, output_nc, n_residual_blocks=9):
    super(Generator, self).__init__()
    # Initial convolution block       
    model = [   nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, 64, 7),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True) ]
    # Downsampling
    in_features = 64
    out_features = in_features*2
    for _ in range(2):
      model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                  nn.InstanceNorm2d(out_features),
                  nn.ReLU(inplace=True) ]
      in_features = out_features
      out_features = in_features*2
    # Residual blocks
    for _ in range(n_residual_blocks):
      model += [ResidualBlock(in_features)]
    # Upsampling
    out_features = in_features//2
    for _ in range(2):
      model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(out_features),
                  nn.ReLU(inplace=True) ]
      in_features = out_features
      out_features = in_features//2
    # Output layer
    model += [  nn.ReflectionPad2d(3),
                nn.Conv2d(64, output_nc, 7),
                nn.Tanh() ]
    self.model = nn.Sequential(*model)

  def forward(self, x):
    return self.model(x)

# conv-batchnorm-leakyReLU block
def conv_batch(in_, out_, kernel_size=3, padding=1, stride=1):
  return nn.Sequential(nn.Conv2d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                       nn.BatchNorm2d(out_),
                       nn.LeakyReLU())

# darknet resblock
class DarkResBlock(nn.Module):
  def __init__(self, in_):
    super(DarkResBlock, self).__init__()
    reduced = int(in_/2)
    self.layer1 = conv_batch(in_, reduced, kernel_size=1, padding=0)
    self.layer2 = conv_batch(reduced, in_)

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out += x
    return out

# darknet53 model
class Darknet53(nn.Module):
  def __init__(self, block, in_ch):
    super(Darknet53, self).__init__()
    self.conv1 = conv_batch(in_ch, 32)                                  # No change in shape  (128)
    self.conv2 = conv_batch(32, 64, stride=2)                           # Shape is halved     (64)
    self.res1 = self.make_layer(block, in_channels=64, num_blocks=1)    # No change in shape  (64)
    self.conv3 = conv_batch(64, 128, stride=2)                          # Shape is halved     (32)
    self.res2 = self.make_layer(block, in_channels=128, num_blocks=2)   # No change in shape  (32)
    self.conv4 = conv_batch(128, 256, stride=2)                         # Shape is halved     (16)
    self.res3 = self.make_layer(block, in_channels=256, num_blocks=8)   # No change in shape  (16)
    self.conv5 = conv_batch(256, 512, stride=2)                         # Shape is halved     (8)
    self.res4 = self.make_layer(block, in_channels=512, num_blocks=8)   # No change in shape  (8)
    self.conv6 = conv_batch(512, 1024, stride=2)                        # Shape is halved     (4)
    self.res5 = self.make_layer(block, in_channels=1024, num_blocks=4)  # No change in shape  (4)
    self.classify1 = nn.Conv2d(1024, 1, kernel_size=4, padding=0)       # Shape is halved     (2)
    #self.classify2 = nn.Conv2d(512, 1, stride=2)                       # Shape is halved     (1)

  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.res1(out)
    out = self.conv3(out)
    out = self.res2(out)
    out = self.conv4(out)
    res256 = self.res3(out)
    out = self.conv5(res256)
    res512 = self.res4(out)
    out = self.conv6(res512)
    res1024 = self.res5(out)
    out = self.classify1(res1024).view(x.shape[0],-1)
    #out = self.classify2(out)
    return [res256, res512, res1024, out]

  def make_layer(self, block, in_channels, num_blocks):
    layers = []
    for i in range(0, num_blocks):
      layers.append(block(in_channels))
    return nn.Sequential(*layers)

# darknet discriminator
def DarkDiscriminator(in_ch):
  return Darknet53(DarkResBlock, in_ch)


######################### UTILS #########################
class ReplayBuffer():
  def __init__(self, max_sz=50):
    assert (max_sz>0), 'Empty Buffer or trying to create black hole. Beware.'
    self.max_sz = max_sz
    self.data = []

  def push_and_pop(self, data):
    to_return = []
    for element in data.data:
      element = torch.unsqueeze(element, 0)
      if len(self.data) < self.max_sz:
        self.data.append(element)
        to_return.append(element)
      else:
        if random.uniform(0,1) > 0.5:
          i = random.randint(0, self.max_sz-1)
          to_return.append(self.data[i].clone())
          self.data[i] = element
        else:
          to_return.append(element)
    return Variable(torch.cat(to_return))

class LambdaLR():
  def __init__(self, n_epochs, offset, decay_start_epoch):
    assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
    self.n_epochs = n_epochs
    self.offset = offset
    self.decay_start_epoch = decay_start_epoch

  def step(self, epoch):
    return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm2d') != -1:
    torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    torch.nn.init.constant_(m.bias.data, 0.0)

def tensor2image(tensor):
  image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
  if image.shape[0] == 1:
    image = np.tile(image, (3,1,1))
  return image.astype(np.uint8)

class ImageDataset(Dataset):
  def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
    self.transform = transforms.Compose(transforms_)
    self.unaligned = unaligned
    self.files_rain = sorted(glob.glob(os.path.join(root, mode, 'rain') + '/*.*'))
    self.files_clear = sorted(glob.glob(os.path.join(root, mode, 'clear') + '/*.*'))
    print('len clear {} len rain {}'.format(len(self.files_clear),len(self.files_rain)))

  def __getitem__(self, index):
    item_rain = self.transform(Image.open(self.files_rain[index % len(self.files_rain)]))
    if self.unaligned:
      item_clear = self.transform(Image.open(self.files_clear[random.randint(0, len(self.files_clear) - 1)]))
    else:
      item_clear = self.transform(Image.open(self.files_clear[index % len(self.files_clear)]))
    return {'rain': item_rain, 'clear': item_clear}

  def __len__(self):
    return max(len(self.files_rain), len(self.files_clear))
