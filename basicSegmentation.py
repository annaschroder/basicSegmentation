#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:46:16 2022

@author: Anna
"""

# following tutorial: https://medium.com/bitgrit-data-science-publication/building-an-image-classification-model-with-pytorch-from-scratch-f10452073212

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn # basic building block for neural neteorks


# load and normalise the data - why?
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   # 3 (RGB) parameters for mean and std
                                                                # using 0.5 means we have a range of [-1,1]
                                                                # why normalise: helps CNN perform better: gets data within a range and reduces skewness as its centres around zero. helps CNN learn faster and better
                                                                

# number of training samples in one iteration or one forward/backwards pass
# each batch trains the network successively and consideres the updated weights from the previous batch
# do we onely run through each batch once for training or multiple times?
# higher batch = more accurate, but higher memory requirement
batch_size = 4

# num workers - allows multi-precess learning
# root: CREATES a folder
# shuffle: reshuffle at each epoch
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# qu: why not shuffle test set? Is it just unecessary or would it negatively effect our model?
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# load and view images - my code
image_N=3
fig, axs = plt.subplots(image_N, image_N)
for i in range(image_N**2):
    axs[int(i/image_N),i%image_N].imshow(trainset.data[i,:,:,0])
plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

# load images - their code:
def imshow(img):
  ''' function to show image '''
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy() # convert to numpy objects
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

# get random training images with iter function
dataiter = iter(trainloader)
images, labels = dataiter.next()

# call function on our images
imshow(torchvision.utils.make_grid(images))

# print the class of the image
print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))


# define a CNN
class Net(nn.Module):       # nn.Module? inputting the module for use in the class?
    ''' simple CNN '''
    
    def __init__(self):
        ''' initialise network ''' 
        super(Net,self).__init__()      # which class is being inherited here?
        # 3 input channel, 6 output channel - RGB img, therefore 3 input, **why 6 output?
        # 5x5 convolutional kernel
        self.conv1=nn.Conv2d(3,6,5)
        # max pool over (2,2) window, ** why (2,2)?
        self.pool=nn.Maxpool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)        # output now 16?
        # fully connected layers - why are 3 layers chosen, why are each of the inputs as stated?
        # applying a linear transformation - go over maths of CNN to understand why
        self.fc1 = nn.Linear(16*5*5,120)        #** why     120 output features?
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self,x):
        '''forward propagation algorithm'''
        




# to do: play around with layers, deeper etc and see how it affects the results 
# what gives you the best outcome?
# change kernel size, add padding
        
# to do: load own data, try in pytorch lightening
