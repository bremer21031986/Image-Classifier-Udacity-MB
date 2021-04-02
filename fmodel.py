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

arch= {'vgg16': 25088}

def setup_network(structure='vgg16', dropout=0.5, hidden_layer=4096, lr=0.001, device='gpu'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
        
    #defining an untrainged feed-forward network as classifiert
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088,4096)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(0.5)),
                            ('fc2', nn.Linear(4096, 256)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(0.5)),
                            ('fc3', nn.Linear(256, 102)),
                            ('output', nn.LogSoftmax(dim=1))]))
    
    print(model)
    #Setting a loss Function
    criterion = nn.NLLLoss()

    #Training only the classifier parameters with Adam and            Learnrate 0.001 - 0.005 seems too high
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device)
    
    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()

    return model, criterion

def save_checkpoint(train_data, model=0, path='checkpoint_ICP2.pth', structure='vgg16', hidden_layer = 4096, dropout=0.5, lr=0.001, epochs = 3):
    model.class_to_idx =  train_data.class_to_idx
    torch.save({'structure' :structure,
                'hidden_layer':hidden_layer,
                'dropout':dropout,
                'learning_rate':lr,
                'epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
               path)
    

def load_checkpoint(path='checkpoint_ICP2'):
    checkpoint = torch.load(path)
    lr = checkpoint['learning_rate']
    hiffen_layer = checkpoint['hidden_layer']
    dropout = checkpoint['dropout']
    epochs = checkpoint['epochs']
    structure = checkpoint['structure']
    
    model,_ = setup_network(structure, dropout, hidden_layer, lr)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def predict(image_path, model, topk=5, device='gpu'):
    model.to('cuda')
    model.eval()
    img = process_image(image_path)
    img = img.numpy()
    img = torch.from_numpy(np.array([img])).float()
    
    with torch.no_grad():
        output = model.forward(img.cuda())
        
    pb = torch.exp(output).data
    return pb.topk(topk)

def process_image(image):
    img_pil = Image.open(image)
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    image = img_transforms(img_pil)
    
    return image
    
