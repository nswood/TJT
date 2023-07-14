import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import numpy as np
import h5py
import json

import sklearn
import numpy.random as random
import corner
import builtins
import scipy
import time
from tqdm import tqdm 
import utils 
import sys
import glob
import models
from losses import *

# Imports neural net tools
import itertools
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd.variable import *
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, auc
from torchmetrics import Accuracy
from torchsummary import summary
from sklearn.decomposition import PCA
import torchsummary
from sklearn.preprocessing import OneHotEncoder
from loguru import logger
from dataset_loader import zpr_loader

p = utils.ArgumentParser()
p.add_args(
    
    ('--vpath', p.STR), ('--opath', p.STR),
    ('--sv', p.STORE_TRUE),
    )
args = p.parse_args()
def main():
    
    (x_pf, x_sv, jet_features, jet_truthlabel) = zpr_loader(args.vpath,maxfiles=10)
    
    
    print("Plotting all features. This might take a few minutes")
    utils.plot_features(jet_features,jet_truthlabel,utils._singleton_labels,args.opath)
    utils.plot_features(x_sv,jet_truthlabel,utils._SV_features_labels,args.opath,"SV")
    utils.plot_features(x_pf,jet_truthlabel,utils._p_features_labels,args.opath,"Particle")
    #utils.plot_features(singletonFeatureData,jet_truthlabel,utils._singleton_features_labels,args.opath)
if __name__ == "__main__":
    main()
        