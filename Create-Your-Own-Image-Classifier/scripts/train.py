## MODEL TRAINING


# Import packages
import sys, argparse, os
from pathlib import Path

# Imports from notebook
import torchvision
import torchvision.models as models

from torchvision.datasets import Flowers102 as flowers
from torchvision import transforms, disable_beta_transforms_warning
# disable pytorch warnings
disable_beta_transforms_warning()
from torchvision.transforms import v2

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import scipy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json



# Local imports
from functions import *
from utils_fn import *


if __name__ == "__main__":

    # Load data with dataloaders
    train_dataloader = dataloader("train")                    
    val_dataloader = dataloader("val")
    test_dataloader = dataloader("test")



    # Build your network

    # Load the pre-trained vgg16 model from torchvision models library

    resnet = models.resnet50(weights="DEFAULT")
    mobilenet= models.mobilenet_v2(weights="DEFAULT")
    efficientnet = models.efficientnet_b0(weights="DEFAULT")

    pre_trained = {'resnet': resnet, 'mobilenet': mobilenet, 'efficientnet': efficientnet}

    args = get_input_args("train")
    model = pre_trained[args.arch]

    ## Define new adapted model
    classifier_fc = get_first_fc_layer(model)

    fc1_in = classifier_fc.in_features
    fc1_out = args.hidden_units[0]
    fc2_in, fc2_out = args.hidden_units


    classifier = nn.Sequential(nn.Linear(fc1_in, fc1_out),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.1),
                                   nn.Linear(fc2_in, fc2_out), # output_size = classes of flowers in the dataset
                                )

    # Attach new layers to be trained on the model
    if hasattr(model, 'classifier'):
        model.classifier = classifier
    else:
        model.fc = classifier

    #print(model,"\n")

    # Prepare model for training
    freeze_layers(model)


    device = check_accelerator(args.gpu)

    # Define the loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()

    if hasattr(model, 'classifier'):
      optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.learning_rate)
    else:
      optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.learning_rate)
    

    # define scheduler to decay the learning rate
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.0005, patience=5, verbose=True)

    epochs = args.epochs

    train_model(train_dataloader,
                    val_dataloader,
                    epochs=epochs,
                    loss_function=criterion,
                    optimizer=optimizer,
                    device=device,
                    model=model)

    evaluate(test_dataloader,
                 criterion,
                 device,
                 model,
                 epochs=1)




    # Save the checkpoint

    PATH = args.save_dir

    #torch.save(model, PATH)
    save_model(model, PATH)

