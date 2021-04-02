import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image

import argparse
import json

from collections import OrderedDict

import basis
import fmodel

arch= {'vgg16': 25088,
       'densenet121': 1024}

parser = argparse.ArgumentParser(description = 'Parser for train.py')

parser.add_argument('--data_dir', action="store", default="./flowers/")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
parser.add_argument('--arch', action="store", default="vgg16")
parser.add_argument('--learning_rate', action="store", type=float,default=0.001)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=4096)
parser.add_argument('--epochs', action="store", default=3, type=int)
parser.add_argument('--dropout', action="store", type=float, default=0.5)
parser.add_argument('--gpu', action="store", default="gpu")

args = parser.parse_args()
dataset = args.data_dir
path = args.save_dir
lr = args.learning_rate
structure = args.arch
hidden_units = args.hidden_units
gpu = args.gpu
epochs = args.epochs
dropout = args.dropout

if gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'
    
def main():
    trainloader, validloader, testloader, train_data = basis.load_data(dataset)
    model, criterion = fmodel.setup_network(structure, dropout, hidden_units, gpu)
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
    
    #Train Model
    steps = 0
    running_loss = 0
    print_every = 10
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
       
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            #Determine the losses and probabilities
            logpbs = model.forward(inputs)
            loss = criterion (logpbs, labels)
        
            loss.backward()
            optimizer.step()
        
            #Keeping track on the progress
            running_loss += loss.item()
        
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                    
                        logpbs = model.forward(inputs)
                        batch_loss = criterion(logpbs, labels)
                        valid_loss += batch_loss.item()
                    
                        #Accuracy
                        pbs= torch.exp(logpbs)
                        toppbs,top_class=pbs.topk(1,dim=1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.cuda.FloatTensor))
        
        print("Training loss: {:.3f}.. ".format(running_loss/(len(trainloader))),
              "Validation loss: {:.3f}..".format(valid_loss/len(validloader)),
                  "Accuracy: {:.3f}".format(accuracy/len(validloader)))
    
        model.train()
    
#Saving Checkpoint
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'epochs': epochs,
                  'hidden_units': hidden_units,
                  'dropout': dropout,
                  'learning_rate': lr,
                  'class_to_idx': model.class_to_idx,
                  'classifier': model.classifier,
                  'optimizer': optimizer.state_dict,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')
    print('Checkpoint Saved')

if __name__ == '__main__':
    main()
