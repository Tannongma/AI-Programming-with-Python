## PREDICT

# Import packages

import torchvision.models as models

from torchvision.datasets import Flowers102 as flowers
from torchvision import transforms, disable_beta_transforms_warning
# disable pytorch warnings
disable_beta_transforms_warning()
from torchvision.transforms import v2

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

import scipy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import sys, argparse, os
from pathlib import Path

# Local imports
from functions import *
from utils_fn import *


if __name__ == "__main__":

    # Load model checkpoint

    args = get_input_args("predict")
    SAVE_PATH = args.checkpoint
    cat_file = args.category_names
    IMG_PATH = args.image_path
    device = args.gpu

    #model = load_checkpoint(model, SAVE_PATH)
    model = torch.load(SAVE_PATH, 
                       weights_only=False, 
                       map_location=torch.device(device))
    model.eval()


    prediction = predict(image_path=IMG_PATH, 
            model=model, 
            cat_file=cat_file,
            device=device,
            topk=1)
   


    topk_dict = show_prediction(image_path=IMG_PATH,
                    model=model,
                    device=device,
                    cat_file=cat_file,
                    topk=args.top_k)
     
    sorted_topk_pairs = sorted(topk_dict.items(), key=lambda item: item[1], reverse=True) 
    
    print("\n\n**** PREDICTION RESULTS ****\n")

    print(f"-- Flower Name: {prediction}\n")
    
    print(f"-- Top {len(sorted_topk_pairs)} classes: \n")
    for k in sorted_topk_pairs:
        print(f"        - {k[0]} ({k[1]*100:.2f}%)\n")
        
    