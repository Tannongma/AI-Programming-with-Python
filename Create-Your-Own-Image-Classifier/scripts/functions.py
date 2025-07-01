## COLLECTION OF TRAINING AND PREDICTION RELATED FUNCTIONS

# Import packages

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

import sys, argparse, os
from pathlib import Path
from dataclasses import dataclass

# Local imports

#from utils_fn import find_file
from utils_fn import *

    
## Train model

def train_model(X_dataloader,
                eval_dataloader,
                epochs,
                loss_function,
                optimizer,
                device,
                model,
                print_every=10):

    model = model.to(device)
    X_avg_loss = 0

   # training loop
    for epoch in range(epochs):
        X_total_loss = 0
        batch_samples = 0

        for batch_num, (X, y) in enumerate(X_dataloader):

            model.train()
            # Move tensors to the device
            X = X.to(device)
            y = y.to(device)

            # sets model gradients to zero calculated gradients
            optimizer.zero_grad()

            # forward pass
            X_logits = model(X)

            # compute loss and perform backpropagation
            X_loss = loss_function(X_logits, y)
            X_loss.backward()

            # update the weights
            optimizer.step()

            X_total_loss += X_loss.item()
            batch_samples += X_logits.shape[0]
            X_avg_loss = X_total_loss/batch_samples

            if (batch_num % print_every == 0):
                print(
                  f"\nEpoch [{epoch + 1}/{epochs}], Training Batch [{batch_num + 1}/{len(X_dataloader)}], Batch Training Loss: {X_loss.item():.4f}, Epoch Average Training Loss: {X_avg_loss:.4f}.\n"
              )

            if (batch_num+1) == 26:
                print(
                  f"\nEpoch Training Metrics = [{epoch + 1}/{epochs}], Training Batch [{batch_num + 1}/{len(X_dataloader)}], Batch Training Loss: {X_loss.item():.4f}, Epoch Average Training Loss: {X_avg_loss:.4f}.\n"
              )
                # Print learning rate
                #print(f"Epoch [{epoch + 1}/{epochs}], Learning Rate: {optimizer.param_groups[0]['lr']:.4f}\n")
            
            optimizer.zero_grad()

        # evaluation loop
        val_loss = compute_accuracy(eval_dataloader, loss_function, device, model)

        # step scheduler
        #scheduler.step(val_loss)




# define accuracy metric

@torch.no_grad()
def compute_accuracy(data_loader, loss_function, device, model, print_every=10):

    model = model.to(device)
    model.eval()

    total_eval_loss = 0
    sum_eval_loss = 0
    sum_eval_accuracy = 0
    batch_samples = 0
    num_batchs = 0

    for batch_num, (X, y) in enumerate(data_loader):

        # Move tensors to the device
        X = X.to(device)
        y = y.to(device)

        # forward pass
        X_logits = model(X)

        # compute evaluation loss
        eval_loss = loss_function(X_logits, y)
        sum_eval_loss += eval_loss.item()


        X_preds_classes = X_logits.argmax(dim=1)

        batch_samples += X_logits.shape[0]
        num_batchs += 1

        pred_correct = X_preds_classes == y.view(*X_preds_classes.shape)

        batch_accuracy = pred_correct.to(torch.float).mean().item()
        sum_eval_accuracy += batch_accuracy

        avg_eval_loss = sum_eval_loss/batch_samples
        avg_eval_accuracy = sum_eval_accuracy/num_batchs


        if (batch_num % print_every == 0):
            # compute and print accuracy for the interval data

            print(
                f"   Evaluation Batch [{batch_num + 1}/{len(data_loader)}], Evaluation Loss: {eval_loss.item():.4f}, Average Evaluation Loss: {avg_eval_loss:.4f}, Accuracy: {avg_eval_accuracy:.4f}."
                )

        if (batch_num+1) == len(data_loader):
            # compute and print accuracy for the epoch

            print(
                f"\n   Epoch Evaluation Metrics = [{batch_num + 1}/{len(data_loader)}], Evaluation Loss: {eval_loss.item():.4f}, Average Evaluation Loss: {avg_eval_loss:.4f}, Accuracy: {avg_eval_accuracy:.4f}.\n"
                )

    return eval_loss.item()




## Test model
@torch.no_grad()
def evaluate(data_loader,
             loss_function,
             device,
             model,
             epochs,
             print_every=10):

    for epoch in range(epochs):
        print(f"Evaluation Epoch [{epoch + 1}/{epochs}]")

        compute_accuracy(data_loader,
                         loss_function,
                         device,
                         model,
                         print_every)




# Predict and show

def process_image(image: str):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    eval_transforms = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = eval_transforms(pil_image)
    return img_tensor



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax



def show_prediction(image_path, 
                    topk, 
                    model, 
                    device, 
                    cat_file):
    ''' Display an image and the corresponding predicted classes.'''
    
    fgpath = find_file(cat_file).as_posix()

    with open(fgpath, 'r') as f:
        cat_to_name = json.load(f)

    img_tensor = process_image(image_path)
    x = img_tensor.unsqueeze(dim=0)

    x = x.to(device)
    model = model.to(device)
    with torch.no_grad():
        logits = model(x)
    topk_values, topk_indices = torch.topk(input=logits, k=topk, dim=1)

    topk_values_np = topk_values.cpu().numpy()
    topk_indices_np = topk_indices.cpu().numpy()

    prediction = predict(image_path=image_path, 
                         model=model, 
                         device=device, 
                         cat_file=cat_file, 
                         topk=1)

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    imshow(img_tensor, ax=ax1)

    ax1.axis('off')
    ax1.set_title(prediction)
    ax2.barh(y=np.arange(topk), width=torch.softmax(topk_values, dim=1).cpu().numpy()[0])
    ax2.set_aspect(0.15)
    ax2.set_yticks(np.arange(topk))
    ax2.set_yticklabels([cat_to_name[str(index.item() + 1)] for index in topk_indices_np[0]], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.05)

    plt.tight_layout()
    
    topk_dict = dict()
    list_topk_classes = [cat_to_name[str(index.item() + 1)] for index in topk_indices_np[0]]
    topk_probabilities = torch.softmax(topk_values, dim=1).cpu().numpy()[0].tolist()
    
    for k, p in zip(list_topk_classes, topk_probabilities):
        topk_dict[k] = p
    
    return topk_dict


def predict(image_path: str, 
            model, device: str, 
            cat_file: str, 
            topk: int=5) -> dict[str, str]:
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    fgpath = find_file(cat_file).as_posix()

    with open(fgpath, 'r') as f:
        cat_to_name = json.load(f)

    img_tensor = process_image(image_path)
    x = img_tensor.unsqueeze(dim=0)
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
    top_logits = torch.topk(input=logits, k=topk, dim=1)

    class_list = list()
    flowers_names = list()
    predict = dict()
    for index in top_logits.indices:
        predicted = index.item() + 1
        class_list.append(predicted)
        flowers_names.append(cat_to_name[str(predicted)])
        predict[str(predicted)] = cat_to_name[str(predicted)]

    return predict



