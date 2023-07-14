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
#import setGPU
import sklearn
import numpy.random as random
import corner
import builtins
import scipy
import time
from tqdm import tqdm 
import utils #import *
import sys
import glob
import models
import losses
# Imports neural net tools
import itertools
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd.variable import *
import torch.optim as optim
import torch.nn.functional as F
#from fast_soft_sort.pytorch_ops import soft_rank
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score,  auc
from torchmetrics import Accuracy
from torchsummary import summary
import torchsummary
from sklearn.preprocessing import OneHotEncoder
from loguru import logger

from Train_Util import Trainer
p = utils.ArgumentParser()
p.add_args(
    ('--mname', p.STR),
    ('--loss', p.STR), ('--model', p.STR), ('--nepochs', p.INT),
    ('--ipath', p.STR), ('--vpath', p.STR), ('--opath', p.STR),
    ('--mpath', p.STR),('--prepath', p.STR),('--continue_training', p.STORE_TRUE), ('--sv', p.STORE_TRUE),
    ('--De', p.FLOAT), ('--Do',p.FLOAT), ('--hidden',p.FLOAT),
    ('--nparts', p.INT),('--LAMBDA_ADV',p.FLOAT),('--nclasses',p.INT), 
    ('--plot_text', p.STR), ('--mini_dataset',p.STORE_TRUE),
    ('--plot_features', p.STORE_TRUE), ('--run_captum',p.STORE_TRUE), ('--test_run',p.STORE_TRUE),
    ('--make_PN', p.STORE_TRUE), ('--make_N2',p.STORE_TRUE), ('--is_binary',p.STORE_TRUE),
    ('--is_peaky',p.STORE_TRUE), ('--no_heavy_flavorQCD',p.STORE_TRUE),
    ('--one_hot_encode_pdgId',p.STORE_TRUE),('--SV',p.STORE_TRUE),
    ('--temperature', p.FLOAT), ('--n_out_nodes',p.INT),
    ('--qcd_only',p.STORE_TRUE), ('--seed_only',p.STORE_TRUE),
    ('--abseta',p.STORE_TRUE), ('--kinematics_only',p.STORE_TRUE),
    ('--istransformer',p.STORE_TRUE),
    ('--num_encoders', p.INT),('--is_decoder',p.STORE_TRUE),
    ('--embedding_size', p.INT), ('--hidden_size', p.INT), ('--feature_size', p.INT), ('--feature_sv_size', p.INT),
    ('--num_attention_heads', p.INT), ('--intermediate_size', p.INT),
    ('--label_size', p.INT), ('--num_hidden_layers', p.INT), ('--batchsize', p.INT),
    ('--mask_charged', p.STORE_TRUE), ('--lr', {'type': float}),
    ('--attention_band', p.INT),
    ('--epoch_offset', p.INT),
    ('--from_snapshot'),
    ('--lr_schedule', p.STORE_TRUE), '--plot',
    ('--pt_weight', p.STORE_TRUE), ('--num_max_files', p.INT),
    ('--num_max_particles', p.INT), ('--dr_adj', p.FLOAT),
    ('--beta', p.STORE_TRUE), ('--load_gpu', p.STORE_TRUE),
    ('--lr_policy'), ('--grad_acc', p.INT), ('--pretrain', p.STORE_TRUE), 
    ('--small_feature', p.STORE_TRUE),('--disco_reg', p.STORE_TRUE),
    ('--replace_mean', p.STORE_TRUE),('--hyperbolic', p.STORE_TRUE),('--hybrid', p.STORE_TRUE),
    
)

#DDP Congigs
p.add_argument('--gpu', default=None, type=int)
p.add_argument('--device', default='cuda', help='device')

p.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
p.add_argument('--rank', default=-1, type=int, 
                    help='node rank for distributed training')
p.add_argument('--dist-url', default='env://', type=str, 
                    help='url used to set up distributed training')
p.add_argument('--dist-backend', default='nccl', type=str, 
                    help='distributed backend')
p.add_argument('--local_rank', default=-1, type=int, 
                    help='local rank for distributed training')
p.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')

args = p.parse_args()

args.nparts = 100



def load_model():
    if args.model =='DNN':
        model = models.DNN("DNN",particleDataTrain.shape[1]*particleDataTrain.shape[2],labelsTrain.shape[1]).to(device)
    
    elif args.model=='PN': 
    
        #model = models.ParticleNetTagger("PN",n_particles,n_vertex,labelsTrain.shape[1])
        if args.PN_v1:
            model = models.ParticleNetTagger("PN",particleDataTrain.shape[2],vertexDataTrain.shape[2],labelsTrain.shape[1],for_inference=_softmax, fc_params=[(128,0.1),(64,0.1)],conv_params=[(16, (64, 64, 64)),(16, (128, 128, 128)),],event_branch=args.event,sigmoid=_sigmoid)
        else: 
            model = models.ParticleNetTagger("PN",particleDataTrain.shape[2],vertexDataTrain.shape[2],labelsTrain.shape[1],for_inference=_softmax,event_branch=args.event,sigmoid=_sigmoid)
        # Batch,Nparts,Nfeatures
        #maskpfTrain = np.zeros((particleDataTrain.shape[0],particleDataTrain.shape[1]),dtype=bool)
        maskpfTrain = np.where(particleDataTrain[:,:,0]>0.,1., 0.)
        maskpfVal = np.where(particleDataVal[:,:,0]>0., 1., 0.)
        maskpfTest = np.where(particleDataTest[:,:,0]>0., 1., 0.)
        masksvTrain = np.where(vertexDataTrain[:,:,0]>0., 1., 0.)
        masksvVal = np.where(vertexDataVal[:,:,0]>0., 1., 0.)
        masksvTest = np.where(vertexDataTest[:,:,0]>0., 1., 0.)
    
        print("mask shape",maskpfTrain.shape)
        
        #maskpfTrain = np.repeat(maskpfTrain,6, axis=2) 
        maskpfTrain = np.expand_dims(maskpfTrain,axis=-1)
        maskpfVal = np.expand_dims(maskpfVal,axis=-1)
        maskpfTest = np.expand_dims(maskpfTest,axis=-1)
        masksvTrain = np.expand_dims(masksvTrain,axis=-1)
        masksvVal = np.expand_dims(masksvVal,axis=-1)
        masksvTest = np.expand_dims(masksvTest,axis=-1)
 
        maskpfTrain = np.swapaxes(maskpfTrain,1,2)
        maskpfVal = np.swapaxes(maskpfVal,1,2)
        maskpfTest = np.swapaxes(maskpfTest,1,2)
        masksvTrain = np.swapaxes(masksvTrain,1,2)
        masksvVal = np.swapaxes(masksvVal,1,2)
        masksvTest = np.swapaxes(masksvTest,1,2)
    
        particleDataTrain = np.swapaxes(particleDataTrain,1,2)
        particleDataVal = np.swapaxes(particleDataVal,1,2)
        particleDataTest = np.swapaxes(particleDataTest,1,2)
        vertexDataTrain = np.swapaxes(vertexDataTrain,1,2)
        vertexDataVal = np.swapaxes(vertexDataVal,1,2)
        vertexDataTest = np.swapaxes(vertexDataTest,1,2)
        print("particle shape",particleDataTrain.shape)
        #sys.exit(1)
    
    elif args.model=='IN_SV_event':
        model = models.GraphNetv2("IN_SV_event",n_particles,labelsTrain.shape[1],6,n_vertices=5,params_v=13,params_e=27,pv_branch=True,event_branch=True, hidden=args.hidden, De=args.De, Do=args.Do,sigmoid=_sigmoid,softmax=_softmax)
        particleDataTrain = np.swapaxes(particleDataTrain,1,2)
        particleDataVal = np.swapaxes(particleDataVal,1,2)
        particleDataTest = np.swapaxes(particleDataTest,1,2)
        vertexDataTrain = np.swapaxes(vertexDataTrain,1,2)
        vertexDataVal = np.swapaxes(vertexDataVal,1,2)
        vertexDataTest = np.swapaxes(vertexDataTest,1,2)
        #singletonFeatureDataTrain = np.swapaxes(singletonFeatureDataTrain,1,2)
        #singletonFeatureDataVal = np.swapaxes(singletonFeatureDataVal,1,2)
        #singletonFeatureDataTest = np.swapaxes(singletonFeatureDataTest,1,2)
    
    elif args.model=='IN_SV': 
        model = models.GraphNetv2("IN_SV",n_particles,labelsTrain.shape[1],6,n_vertices=5, params_v=13, pv_branch=True, hidden=args.hidden, De=args.De, Do=args.Do,sigmoid=_sigmoid,softmax=_softmax)
    
        #somehow the (parts,features) axes get flipped in the IN 
        particleDataTrain = np.swapaxes(particleDataTrain,1,2)
        particleDataVal = np.swapaxes(particleDataVal,1,2)
        particleDataTest = np.swapaxes(particleDataTest,1,2)
        vertexDataTrain = np.swapaxes(vertexDataTrain,1,2)
        vertexDataVal = np.swapaxes(vertexDataVal,1,2)
        vertexDataTest = np.swapaxes(vertexDataTest,1,2)
    
    elif args.model=='IN_noSV': 
        model = models.GraphNetv2("IN_noSV",n_particles,labelsTrain.shape[1],6,hidden=args.hidden, De=args.De, Do=args.Do,sigmoid=_sigmoid,softmax=_softmax,event_branch=False,)
    
        #somehow the (parts,features) axes get flipped in the IN 
        particleDataTrain = np.swapaxes(particleDataTrain,1,2)
        particleDataVal = np.swapaxes(particleDataVal,1,2)
        particleDataTest = np.swapaxes(particleDataTest,1,2)
    
    elif args.model=='transformer':
        model = models.Transformer(args,args.mname,True,False, args.sv)
    else:
        raise ValueError("Don't understand model ", args.model)
    return model

def load_train_objs():
    model = load_model()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    return model, optimizer





def main():
    
    model, optimizer = load_train_objs()
    model = model.cuda()
    if args.continue_training:
        outdir = "/".join(args.mpath.split("/")[:-1])
    else:    
        outdir = f"./{args.opath}/{model.name.replace(' ','_')}"
    loaded_dict = torch.load(args.mpath)
    state_dict = model.state_dict()
    updated_dict = {}
    
    for key in loaded_dict.keys():
        if key.startswith('module.'):
            updated_key = key[len('module.'):]
        else:
            updated_key = key
        updated_dict[updated_key] = loaded_dict[key]

    model.load_state_dict(updated_dict)
    outdir = utils.makedir(outdir,args.continue_training)
    rand_x_pf = torch.randn(10,100,13).cuda()
    print(f'Softmax enabled: {model.softmax}')
    if args.sv:
            rand_x_sv = torch.rand(10,5,14).cuda()
            torch.onnx.export(model,
                  (rand_x_pf,rand_x_sv),
                  f"{outdir}/"+args.mname+".onnx",
                  export_params=True,
                  opset_version=12,
                  do_constant_folding=True,
                  input_names=['pf','sv'],
                  output_names=["outputs"],
                  dynamic_axes={'pf' : {0 : 'batch_size'},'sv' : {0 : 'batch_size'},'outputs' : {0 : 'batch_size'}},
            )
    else: 
        torch.onnx.export(model,
              rand_x_pf,
              f"{outdir}/"+args.mname+".onnx",
              export_params=True,
              opset_version=12,
              do_constant_folding=True,
              input_names=['pf'],
              output_names=["outputs"],
              dynamic_axes={'pf' : {0 : 'batch_size'},'outputs' : {0 : 'batch_size'}},
            )
    print('saved ONNX')
main()