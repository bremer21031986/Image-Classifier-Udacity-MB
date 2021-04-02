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

import fmodel

parser = argparse.ArgumentParser(description = 'Parser for predict.py')

parser.add_argument('input', default='./flowers/test/56/image_02779.jpg', nargs='?', action='store', type=str)
parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = parser.parse_args()
path_image = args.input
number_of_outputs = args.top_k
device = args.gpu

path = args.checkpoint

def main():
    model = fmodel.load_checkpoint(path)
    with open ('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
        
    pbs = fmodel.predict(path_image, model, number_of_outputs, device)
    labels = [cat_to_name[str(index + 1)] for index in np.array(pbs[1][0])]
    pb = np.array(pbs[0][0])
    i = 0
    while i < number_of_outputs:
        print('{} with prob of {}'.format(labels[i], pb[i]))
        i += 1
        print('Final Prediction')

if __name__ == '__main__':
    main()
