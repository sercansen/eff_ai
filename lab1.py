# librairies
from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from models.resnet import ResNet18

import matplotlib.pyplot as plt
import numpy as np
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device '+str(device))

trainloader = DataLoader(minicifar_train,batch_size=32,sampler=train_sampler)
validloader = DataLoader(minicifar_train,batch_size=32,sampler=valid_sampler)
testloader = DataLoader(minicifar_test,batch_size=32)

# model
net = ResNet18()
net = net.to(device)
net.to(device=device)

lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr,momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)