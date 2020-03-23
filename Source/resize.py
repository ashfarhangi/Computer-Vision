#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import io
import base64
from IPython.display import HTML
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
# import skvideo.io
from data_loader import DataClass
# video = io.open('test.avi', 'r+b').read()
FOLDER = "data/resize"
# cv2.imshow(video)


# In[2]:


counter = {i:0 for i in range(101)}
train_counter = {i:0 for i in range(100)}
val_counter = {i:0 for i in range(100)}
arr = []
with open("data/all_images1.txt") as f:
    for line in f:
        label = int(line.split(" ")[1])
        arr.append(label)


# In[3]:


arr = np.asarray(arr)


# In[4]:


arr[::-1].sort()


# In[5]:


arr[:100]


# In[6]:


1 *  296 * 3 * 160 * 240 * 32


# In[7]:


FOLDER_DATASET = "data/"
IMAGE_DATASET = "UCF101_images/"

dataloader = {'train' : DataClass(FOLDER_DATASET, IMAGE_DATASET, "train1.txt"),
              'validation' : DataClass(FOLDER_DATASET, IMAGE_DATASET, "val1.txt")}


# In[8]:


input, label = dataloader['train'].getbatch(3)


# In[9]:


input = Variable(torch.from_numpy(input).float())


# In[10]:


class CNNGRU(nn.Module):
    def __init__(self):
        super(CNNGRU, self).__init__()
        self.input_dim = 1000
        self.hidden_layers = 101
        self.rnn_layers = 2
#         self.classes = 101
#         self.sample_rate = 12
        
        self.conv = torchvision.models.resnet18(pretrained=True)
        for param in self.conv.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(self.input_dim, self.hidden_layers, self.rnn_layers)
        self.gru = nn.GRU(self.input_dim, self.hidden_layers, self.rnn_layers, dropout=0.2)
#         self.linear = nn.Linear(
#             in_features=self.hidden_layers, out_features=self.classes)

    def forward(self, x):
        n, t,c, w, h = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)
        x = x.view(t*n,c,w,h)
        conv_output = self.conv(x) #convolve allframes       
        conv_output = conv_output.view(n,t,-1).transpose(1,0)
#         conv_output = self.conv(x).view(x.size(0),x.size(1),-1).transpose(1,0)
        out, _ = self.gru(conv_output) # pass convolution to gru
        lstm_output = out[-1, :, :].data
#         print(lstm_output.size())
#         output = self.linear(lstm_output) #linear layer 
        return lstm_output


# In[12]:


use_gpu = False
model_ft = CNNGRU()
if use_gpu:
    model_ft = model_ft.cuda()
# print(model_ft)
criterion = nn.CrossEntropyLoss()

#Remove all parameters not to be optimized
ignored_params = list(map(id, model_ft.conv.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params,
                     model_ft.parameters())
                     
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD([{'params': base_params}], lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[13]:


model_parameters = filter(lambda p: p.requires_grad, model_ft.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
params


# In[14]:


12613561 * 32 


# In[15]:


size = input.size()
size[0]


# In[16]:


def func():
    input, label = dataloader['train'].getbatch(3)
    input = Variable(torch.from_numpy(input).float())
    model_ft(input)
get_ipython().run_line_magic('timeit', 'func()')


# In[ ]:





# In[17]:


#Takes about 15minutes to be completed
with open("data/all_images1.txt") as f:
    for line in f:
        image_folder = line.split(" ")[0]
        length = line.split(" ")[1]
        image_url =  "data/" + "UCF101_images/" + image_folder
        image_resize_url =  "data/" + "UCF101_images_r/" + image_folder
        
        
        for i in range(0, int(length)): #pad the beginning
            image = cv2.imread(image_url + "_" + str(i) + ".jpg")                
            image = cv2.resize(image, (267,200), interpolation = cv2.INTER_AREA)
            cv2.imwrite(image_url + "_r_" + str(i) + ".jpg",image)

