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

from pretrain_dataloader import gen_matched_data,gen_matched_simple_data

p = utils.ArgumentParser()
p.add_args(
    ('--mname', p.STR),
    ('--loss', p.STR), ('--model', p.STR), ('--nepochs', p.INT),
    ('--ipath', p.STR), ('--vpath', p.STR), ('--opath', p.STR),
    ('--mpath', p.STR), ('--continue_training', p.STORE_TRUE), ('--sv', p.STORE_TRUE),
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
    ('--lr_policy'), ('--grad_acc', p.INT), ('--pretrain', p.STORE_TRUE))

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
np.random.seed(42)


from Pretrain_util_BCE import PreTrainer


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))



def load_data():
    if args.load_gpu:
        from dataset_loader_gpu import zpr_loader
        if not args.mpath or (args.mpath and args.continue_training):
            data_train = zpr_loader(args.ipath,maxfiles=args.num_max_files) 
            train_sampler = torch.utils.data.distributed.DistributedSampler(data_train, shuffle=True)

        data_val = zpr_loader(args.vpath,maxfiles=args.num_max_files)
        val_sampler = None
        if not args.mpath or (args.mpath and args.continue_training):
            train_loader = gen_matched_simple_data(data_train,len(data_train), batch_size=args.batchsize,shuffle=(train_sampler is None),
            sampler=train_sampler)
                
        val_loader = gen_matched_simple_data(data_val,len(data_val), batch_size=args.batchsize,shuffle=True,)
    else:
        from dataset_loader import zpr_loader
        if not args.mpath or (args.mpath and args.continue_training):
            data_train = zpr_loader(args.ipath,maxfiles=args.num_max_files) 
            train_sampler = torch.utils.data.distributed.DistributedSampler(data_train, shuffle=True)
            
        data_val = zpr_loader(args.vpath,maxfiles=args.num_max_files)
        val_sampler = None
        if not args.mpath or (args.mpath and args.continue_training):
            
            train_loader = gen_matched_simple_data(data_train,len(data_train), batch_size=args.batchsize,shuffle=(train_sampler is None),
            sampler=train_sampler)
            
        val_loader = gen_matched_simple_data(data_val, len(data_val), batch_size=args.batchsize,shuffle=(val_sampler is None),
             sampler=val_sampler)
        return train_loader, val_loader
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
        
        model = models.Transformer(args,"transformer",_softmax,_sigmoid, args.sv,pretrain = args.pretrain)
    else:
        raise ValueError("Don't understand model ", args.model)
    return model

def load_train_objs():
    model = load_model()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    return model, optimizer





def main(save_every: int, total_epochs: int, batch_size: int, loss):
    ddp_setup()
    
    model, optimizer = load_train_objs()
    outdir = f"./{args.opath}/{model.name.replace(' ','_')}"
    outdir = utils.makedir(outdir,args.continue_training)
    train_loader, val_loader = load_data()
    trainer = PreTrainer(model, train_loader,val_loader, optimizer, save_every,outdir,loss,total_epochs, args)
    trainer.train(total_epochs)
#     if int(os.environ["LOCAL_RANK"]) == 0:
#         trainer.run_inference(args.plot_text,val_loader)
    if args.run_captum:
        from captum.attr import IntegratedGradients
        model.eval()
        torch.manual_seed(123)
        np.random.seed(123)
        baseline = torch.zeros(1,particleDataTrain.shape[1],particleDataTrain.shape[2]).to(device)
        baselineSV = torch.zeros(1,vertexDataTrain.shape[1],vertexDataTrain.shape[2]).to(device)
        baselineE  = torch.zeros(1,singletonFeatureDataTrain.shape[1]).to(device)
        inputs = torch.rand(1,particleDataTrain.shape[1],particleDataTrain.shape[2]).to(device)
        inputsSV = torch.rand(1,vertexDataTrain.shape[1],vertexDataTrain.shape[2]).to(device)
        inputsE = torch.rand(1,singletonFeatureDataTrain.shape[1]).to(device)
     
        ig = IntegratedGradients(model)
        os.system("mkdir -p "+outdir+"/captum/")
        if IN_SV_event:
            attributions, delta = ig.attribute((inputs,inputsSV,inputsE,), (baseline,baselineSV,baselineE), target=3, return_convergence_delta=True)
            np.savez(outdir+"/captum/qcd_score.npz",pf=attributions[0].cpu().detach().numpy(),sv=attributions[1].cpu().detach().numpy(),event=attributions[2].cpu().detach().numpy())
            attributions, delta = ig.attribute((inputs,inputsSV,inputsE,), (baseline,baselineSV,baselineE), target=2, return_convergence_delta=True)
            np.savez(outdir+"/captum/qq_score.npz",pf=attributions[0].cpu().detach().numpy(),sv=attributions[1].cpu().detach().numpy(),event=attributions[2].cpu().detach().numpy())
            attributions, delta = ig.attribute((inputs,inputsSV,inputsE,), (baseline,baselineSV,baselineE), target=1, return_convergence_delta=True)
            np.savez(outdir+"/captum/cc_score.npz",pf=attributions[0].cpu().detach().numpy(),sv=attributions[1].cpu().detach().numpy(),event=attributions[2].cpu().detach().numpy())
            attributions, delta = ig.attribute((inputs,inputsSV,inputsE,), (baseline,baselineSV,baselineE), target=0, return_convergence_delta=True)
            np.savez(outdir+"/captum/bb_score.npz",pf=attributions[0].cpu().detach().numpy(),sv=attributions[1].cpu().detach().numpy(),event=attributions[2].cpu().detach().numpy())
    destroy_process_group()
    

if __name__ == "__main__":
    n_particle_features = 6
    n_particles = args.nparts
    n_vertex_features = 13
    n_vertex = 5
    batchsize = args.batchsize
    n_epochs = args.nepochs
    
    print("Running with %i particle features, %i particles, %i vertex features, %i vertices, %i batchsize, %i epochs"%(n_particle_features,n_particles,n_vertex_features,n_vertex,batchsize,n_epochs))
    print("Loss: ", args.loss)
    
    _sigmoid=False
    _softmax=True
    if args.loss == 'bce':
        loss = nn.BCELoss()
    elif args.loss == 'categorical':
        loss = nn.CrossEntropyLoss()
        _sigmoid=False
        _softmax=False
    elif args.loss == 'all_vs_QCD':
        loss = losses.all_vs_QCD
        _sigmoid=True
        _softmax=False
    elif args.loss == 'jsd':
        loss = losses.jsd
    elif args.loss == 'disco':
        loss = losses.disco
    elif args.loss == 'disco_all_vs_QCD':
        loss = losses.disco_all_vs_QCD
        _sigmoid=True
        _softmax=False
    else:
        raise NameError("Don't understand loss")
    
    if ('jsd' in args.loss or 'disco' in args.loss) and not args.LAMBDA_ADV:
        raise ValueError("must provide lambda_adv for adversarial")
   
    main(1, args.nepochs, args.batchsize,loss)