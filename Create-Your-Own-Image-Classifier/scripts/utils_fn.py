## UTILITY FUNCTIONS COLLECTION

# Import packages

import torchvision.models as models

from torchvision.datasets import Flowers102 as flowers
from torchvision import transforms, disable_beta_transforms_warning
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

# disable pytorch warnings
disable_beta_transforms_warning()

import sys, argparse, os
from pathlib import Path




# TRAINING RELATED

def dataloader(split: str) -> str | DataLoader:
    
    # set directory paths

    data_dir = os.path.join(os.getcwd(), 'flowers')

    train_dir = os.path.join(data_dir,'train')
    valid_dir = os.path.join(data_dir,'valid')
    test_dir = os.path.join(data_dir,'test')
    
    train_transforms = transforms.Compose(
            [transforms.Resize(size=256),
             transforms.RandomResizedCrop(size=(224,224)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(15),
             transforms.RandomGrayscale(p=0.1),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    # Transformation for validation and test datasets
    eval_transforms = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    #  Use torchvision to load the data

    train_data = flowers(
        root=test_dir,
        split="train",
        transform=train_transforms,
        target_transform=None,
        download=True
    )

    val_data = flowers(
        root=valid_dir,
        split="val",
        transform=eval_transforms,
        target_transform=None,
        download=True
    )

    test_data = flowers(
        root=test_dir,
        split="test",
        transform=eval_transforms,
        target_transform=None,
        download=True
    )

    # define the dataloaders
    if split == "train":
      train_dataloader = DataLoader(train_data,
                                                batch_size=40,
                                                shuffle=True)
      return train_dataloader
    
    elif split == "test":
      test_dataloader = DataLoader(test_data,
                                              batch_size=40,
                                              shuffle=False)
      return test_dataloader

    elif split == "val":
      val_dataloader = DataLoader(val_data,
                                              batch_size=40,
                                              shuffle=False)
      return val_dataloader

    else:
      print('Set split parameter making choice = ["train", "test", "val"]')


    
    
# OTHER HELPER FUNCTIONS

# workspace configurations
def check_accelerator(user_input: str) -> str:
    """Helper function to set and use accelerator if available"""

    if torch.cuda.is_available() and user_input == "cuda":
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and user_input == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}\n")

    return device


def find_file(filename: str, search_dir=Path.cwd()) -> str: 
    for path in search_dir.rglob(filename):  
        return path.resolve()  # Return the first match 
    return None 


# CLI related
def get_input_args(phase: str):
    """
    Retrieves and parses the CLI arguments provided by the user when
    they run the program from a terminal. Default values are used for the missing arguments. 
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    
    # set directory paths

    data_dir = os.path.join(os.getcwd(), 'flowers')
    
    train_dir = os.path.join(data_dir,'train')
    valid_dir = os.path.join(data_dir,'valid')
    test_dir = os.path.join(data_dir,'test')

    save_dir = Path(os.path.join(data_dir, 'model.pth')).resolve()
    

    # add Train specific arguments
    train_parser = argparse.ArgumentParser(
                        prog='train',
                        description="Trains a neural network based on the chosen parameters",
                        epilog='Default directory to save downloads is cwd')


    train_parser.add_argument("data_dir", type=str, default=data_dir,
                        help="Provide directory for saving downloads and data")
    
    train_parser.add_argument("-a", "--arch", type=str, 
                               choices=["mobilenet", "efficientnet", "resnet"],
                               default="mobilenet",
                        help="Choose a model among 'mobilenet', 'efficientnet', 'resnet'")

    train_parser.add_argument("-s", "--save_dir", type=str, 
                              default=save_dir,
                        help="Determine save directory for model checkpoint")

    train_parser.add_argument("-l", "--learning_rate", type=float, default=0.0005,
                        help="Determine learning rate")
    
    train_parser.add_argument("-e", "--epochs", type=int, default=50,
                        help="Determine the number of epochs for training and validation")

    train_parser.add_argument("-u", "--hidden_units", type=dict, default=(512, 102),
                        help="Determine number of hidden units 'fc2' layers inputs and outputs in classifier")
    
    train_parser.add_argument("-g", "--gpu", type=str, choices=["cuda", "cpu", "fps"], default='cuda',
                        help="Specify which option ['cuda', 'cpu', 'fps'] is available for acceleration; Default = GPU")

    
    # add Predict specific arguments
    predict_parser = argparse.ArgumentParser(
                        prog='predict',
                        description="Use trained model to predict image class based on the chosen parameters",
                        epilog='Do not inherit from any other parser. Ignore Defaults for required arguments')

    
    predict_parser.add_argument("image_path", type=str,
                        help="Provide path to image to be classified")
    
    predict_parser.add_argument("checkpoint", type=str, default=save_dir,
                        help="Provide model checkpoint path")
    
    predict_parser.add_argument("-a", "--arch", type=str, 
                               choices=["mobilenet", "efficientnet", "resnet"],
                               default="mobilenet",
                        help="Choose a model among 'mobilenet', 'efficientnet', 'resnet'")
    
    predict_parser.add_argument("-k", "--top_k", type=int, default=5,
                        help="Determine the top_k most likely classes to be returned")
    
    predict_parser.add_argument("-c", "--category_names", type=str, default="cat_to_name.json",
                        help="Provide category_names file path")
    
    predict_parser.add_argument("-g", "--gpu", type=str, choices=["cuda", "cpu", "fps"], default='cuda',
                        help="Specify which option ['cuda', 'cpu', 'fps'] is available for acceleration; Default = GPU")

    if phase == "train":
        return train_parser.parse_args()
    elif phase == "predict":
        return predict_parser.parse_args()
    else:
        print("OPTIONS: str --> choice: ['train', 'predict']")
    





def save_checkpoint(model, save_path, epochs, cat_file, loss=0.0001):
    EPOCH = epochs
    LOSS = loss
    PATH = save_path

    # Save the model parameters
    
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': LOSS,
                'cat_to_name': cat_file
                }, PATH)
    
    print("model training checkpoint saved")
    
    
def save_model(model, path):
    """ Save the whole model"""
    torch.save(model, path)
    print(f"###  Trained model {model.__class__.__name__} saved at {path}  ###")
        
    
    
# Prediction related functions

def load_checkpoint(model, save_path, device):
    checkpoint = torch.load(save_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    if "cat_to_name" in checkpoint.keys():
        cat_to_name = checkpoint['cat_to_name']

    # Prepare for evaluation and inference
    model.eval()
    model = model.to(device)
    return model


def get_first_fc_layer(model):
    # Check if the model has a classifier or fc attribute
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            for layer in model.classifier:
                if isinstance(layer, nn.Linear):
                    return layer
    elif hasattr(model, 'fc'):
        # For models like ResNet, the fully connected layer is in 'fc'
        if isinstance(model.fc, nn.Linear):
            return model.fc
    return "No fc layer found in model head"



def freeze_layers(model):
    # Freeze all layers
    for params in model.parameters():
        params.requires_grad = False
        
    # Check if the model has a classifier or fc attribute
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            for layer in model.classifier:
                if isinstance(layer, nn.Linear):
                    for params in model.classifier.parameters():
                        params.requires_grad = True
    elif hasattr(model, 'fc'):
        # For models like ResNet, the fully connected layer is in 'fc'
        if isinstance(model.fc, nn.Sequential):
            for params in model.fc.parameters():
                params.requires_grad = True
    return "No fc layer found in model head"



    
        


