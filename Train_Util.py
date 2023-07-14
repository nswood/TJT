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

import contrastive_losses

def load_matching_state_dict(model, state_dict_path):
    state_dict = torch.load(state_dict_path)
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict,strict=False)
    
class Trainer:
    def __init__(
        self,
        model,
        train_data,
        val_data,
        optimizer,
        save_every,
        outdir, 
        loss,
        max_epochs,
        args
        
    ):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.args = args
        self.outdir = outdir
        self.loss = loss
        self.loss_vals_training = np.zeros(max_epochs)
        self.loss_vals_validation = np.zeros(max_epochs)
        self.acc_vals_training = np.zeros(max_epochs)
        self.acc_vals_validation = np.zeros(max_epochs)
        if args.mname == None:
            self.name = self.model.name
        else:
            self.name = args.mname
        
#         if os.path.exists(snapshot_path):
#             print("Loading snapshot")
#             self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id],find_unused_parameters=True)
    
            

#     def _load_snapshot(self):
#         loc = f"cuda:{self.gpu_id}"
#         snapshot = torch.load(snapshot_path, map_location=loc)
#         self.model.load_state_dict(snapshot["MODEL_STATE"])
#         self.epochs_run = snapshot["EPOCHS_RUN"]
#         print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
    def _run_epoch_val(self, epoch):
        b_sz = len(next(iter(self.val_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.val_data)}")
        loss_validation, acc_validation = [], []
        self.model.train(False)
        for istep, (x_pf, x_sv, jet_features, jet_truthlabel) in enumerate(tqdm(self.val_data)):
                lv, av = self._run_batch_val(istep,x_pf, x_sv, jet_features, jet_truthlabel)
                loss_validation.append(lv)
                acc_validation.append(av)
        epoch_val_loss = np.mean(loss_validation)
        epoch_val_acc  = np.mean(acc_validation)
        self.loss_vals_validation[epoch] = epoch_val_loss
        self.acc_vals_validation[epoch]  = epoch_val_acc
        
    def _run_batch_val(self, istep, x_pf, x_sv, jet_features, jet_truthlabel):
        self.model.train(False)
        accuracy = Accuracy().to(self.gpu_id)
        if 'all_vs_QCD' in self.args.loss:
            jet_truthlabel = jet_truthlabel[:,:-1]

        #if (self.args.test_run and istep>10 ): break
        x_pf = torch.nan_to_num(x_pf,nan=0.,posinf=0.,neginf=0.)
        x_sv = torch.nan_to_num(x_sv,nan=0.,posinf=0.,neginf=0.)
        for param in self.model.parameters():
            param.grad = None
        
        if not self.args.load_gpu:
            x_pf = x_pf.to(self.gpu_id)
            x_sv = x_sv.to(self.gpu_id)
            jet_features = jet_features.to(self.gpu_id)
            jet_truthlabel = jet_truthlabel.to(self.gpu_id)
        if self.args.sv:
            output = self.model(x_pf,x_sv)
        elif self.args.hybrid:
            output,cspace = self.model(x_pf)
        else:
            output = self.model(x_pf)
        if self.args.hybrid:
            loss_fn = contrastive_losses.SimCLRLoss()
            cspace = torch.unsqueeze(cspace, dim= 1)
            contrastive_l = loss_fn.forward2(cspace, torch.argmax(jet_truthlabel, dim=1))
        #sys.exit(1)
        mass = jet_features[:,utils._singleton_labels.index('zpr_fj_msd')]
        if self.args.loss == 'jsd':
            l = self.loss(output, jet_truthlabel, one_hots[istep*batchSize:(istep+1)*batchSize].to(self.gpu_id), n_massbins=n_massbins, LAMBDA_ADV=self.args.LAMBDA_ADV)
        elif 'disco' in self.args.loss:
            #print(mass[:10])
            l = self.loss(output, jet_truthlabel, mass, LAMBDA_ADV=self.args.LAMBDA_ADV,)
        else:
            if self.args.ptweight:
                l = self.loss(output, jet_truthlabel)
                l_weighted = l * jet_features[:,0]/500.0
                l = torch.mean(l_weighted)
            elif self.args.weight_hist_QCD:
                l = self.loss(output, jet_truthlabel)
                scaling = self.get_scaling_weights(jet_features[:,0],self.weights, self.adj_min).to(self.gpu_id)
                l = torch.mean(l*scaling)
            else:
                l = self.loss(output, jet_truthlabel)
            
            
        if self.args.disco_reg:
            l = l + losses.disco(output, jet_truthlabel, mass, LAMBDA_ADV=self.args.LAMBDA_ADV)
            
        if  self.args.hybrid:
            l = l + 0.01*contrastive_l

        return l.item(),accuracy(output,torch.argmax(jet_truthlabel.squeeze(), dim=1)).cpu().detach().numpy()
    def _run_batch_train(self, istep, x_pf, x_sv, jet_features, jet_truthlabel):
        self.model.train(True)
        accuracy = Accuracy().to(self.gpu_id)
        if 'all_vs_QCD' in self.args.loss:
            jet_truthlabel = jet_truthlabel[:,:-1]

        #if (self.args.test_run and istep>10 ): break
        x_pf = torch.nan_to_num(x_pf,nan=0.,posinf=0.,neginf=0.)
        x_sv = torch.nan_to_num(x_sv,nan=0.,posinf=0.,neginf=0.)
        for param in self.model.parameters():
            param.grad = None
        self.optimizer.zero_grad()
        if not self.args.load_gpu:
            x_pf = x_pf.to(self.gpu_id)
            x_sv = x_sv.to(self.gpu_id)
            jet_features = jet_features.to(self.gpu_id)
            jet_truthlabel = jet_truthlabel.to(self.gpu_id)
        if self.args.sv:
            output = self.model(x_pf,x_sv)
        elif self.args.hybrid:
            output,cspace = self.model(x_pf)
        else:
            output = self.model(x_pf)
        #sys.exit(1)
        mass = jet_features[:,utils._singleton_labels.index('zpr_fj_msd')]
        if  self.args.hybrid:
            loss_fn = contrastive_losses.SimCLRLoss()
            cspace = torch.unsqueeze(cspace, dim= 1)
            contrastive_l = loss_fn.forward2(cspace, torch.argmax(jet_truthlabel, dim=1))
        if self.args.loss == 'jsd':
            l = self.loss(output, jet_truthlabel, one_hots[istep*batchSize:(istep+1)*batchSize].to(self.gpu_id), n_massbins=n_massbins, LAMBDA_ADV=self.args.LAMBDA_ADV)
        elif 'disco' in self.args.loss:
            #print(mass[:10])
            l = self.loss(output, jet_truthlabel, mass, LAMBDA_ADV=self.args.LAMBDA_ADV,)
        elif self.args.weight_hist_QCD:
            l = self.loss(output, jet_truthlabel)
            scaling = self.get_scaling_weights(jet_features[:,0],self.weights, self.adj_min).to(self.gpu_id)
            l = torch.mean(l*scaling)
        else:
            l = self.loss(output, jet_truthlabel)
        #print(istep,l.item())
        if self.args.disco_reg:
                l = l + losses.disco(output, jet_truthlabel, mass, LAMBDA_ADV=self.args.LAMBDA_ADV)
        if  self.args.hybrid:
            l = l + 0.01*contrastive_l
        l.backward()
        self.optimizer.step()

        torch.cuda.empty_cache()
        return l.item(),accuracy(output,torch.argmax(jet_truthlabel.squeeze(), dim=1)).cpu().detach().numpy()

    
    def _run_epoch_train(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        loss_training, acc_training = [], []
        loss_validation, acc_validation = [], []
        self.model.train(True)
        for istep, (x_pf, x_sv, jet_features, jet_truthlabel) in enumerate(tqdm(self.train_data)):
                lt, at = self._run_batch_train(istep,x_pf, x_sv, jet_features, jet_truthlabel)
                
                loss_training.append(lt)
                acc_training.append(at)
                
        
        epoch_train_loss = np.mean(loss_training)
        epoch_train_acc  = np.mean(acc_training)
        
        self.loss_vals_training[epoch] = epoch_train_loss
        self.acc_vals_training[epoch]  = epoch_train_acc
    def _save_snapshot(self, epoch):
        
        torch.save(self.model.state_dict(), 
                "{}/epoch_{}_{}_loss_{}_{}_acc_{}_{}.pth".format(self.outdir,epoch,self.name.replace(' ','_'),round(self.loss_vals_training[epoch],4),round(self.loss_vals_validation[epoch],4),round(self.acc_vals_training[epoch],4),round(self.acc_vals_validation[epoch],4))
            )
        print(f" Training snapshot saved")
    def get_weight(self,val,weight,adj_min):
        return weight[int((val-adj_min - (val % 50))/50)]
    def get_scaling_weights(self, data, weights,adj_min):
        scaling_weights = torch.ones(data.size(0))  # Initialize all weights as 1

        for i in range(data.size(0)):

            scaling_weights[i] = self.get_weight(data[i], weights,adj_min)

        return scaling_weights

    def train(self, max_epochs: int):
        self.model.train(True)
        np.random.seed(max_epochs)
        random.seed(max_epochs)
        
        if self.args.weight_hist_QCD:
            print('weighting QCD by pT distribution')
           
            jet_features = self.train_data.dataset.data_jetfeatures[:,0]
            labels = self.train_data.dataset.data_truthlabel
            
            
            qcd_pt = jet_features
            qcd_min,qcd_max = qcd_pt.min(),qcd_pt.max()
            adj_min = qcd_min-qcd_min % 50 
            self.adj_min = adj_min
            adj_max = qcd_max+(50-qcd_max %50)
            bins =np.linspace(adj_min,adj_max ,int((adj_max-adj_min)/50)+1)
            counts, bins = np.histogram(qcd_pt, bins = bins)
            weights = np.max(counts)/counts
            for i in range(len(weights)):
                    if weights[i] < 0.2:
                        weights[i] = 0.2
                    elif weights[i] > 10:
                        weights[i] = 10
            self.weights = torch.tensor(weights)
            
        
        print('training')
        
#         if self.args.distributed: 
#             train_loader.sampler.set_epoch(nepochs)
          
        
        if self.args.fix_weights:
            print('Freezing Weights')
            #Freeze whole model
            for param in self.model.parameters():
                param.requires_grad = False
                
            for name, param in self.model.named_parameters():
                #Unfreeze final layers    
                
                if name in ['module.final_embedder.1.weight',
                            'module.final_embedder.1.bias',
                            'module.final_embedder.2.weight',
                            'module.final_embedder.2.bias']:
                    
                    param.requires_grad = True
        #filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.args.lr)
            
        model_dir = self.outdir
        os.system("mkdir -p ./"+model_dir)
        n_massbins = 20
        start_epoch = 0
        if self.args.continue_training:
            self.model.load_state_dict(torch.load(self.args.mpath))
            start_epoch = self.args.mpath.split("/")[-1].split("epoch_")[-1].split("_")[0]
            start_epoch = int(start_epoch) + 1
            print(f"Continuing training from epoch {start_epoch}...")
            
        else:
            start_epoch = 1
            if not self.args.prepath == None:
                print('loaded pretrained model')
                
                load_matching_state_dict(self.model,self.args.prepath)
                
        print('in train')
        end_epoch = max_epochs
        for epoch in range(start_epoch,max_epochs):
            
            if not self.args.fix_weights_duration == None:
                if epoch == self.args.fix_weights_duration:
                    print('Unfreezing weights')
                    for param in self.model.parameters():
                        param.requires_grad = True
                    self.optimizer = optim.Adam(self.model.parameters(), lr = self.args.lr)

            self._run_epoch_train(epoch)
            self._run_epoch_val(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
                
        torch.cuda.empty_cache()
        utils.plot_loss(self.loss_vals_training,self.loss_vals_validation,model_dir)
    
    def run_inference(self, training_text,val_loader):
        with torch.no_grad():
            print("Running predictions on test data")
            predictions = []
            testingLabels = []
            testingSingletons = []
            batch_size = 1000
    
            for istep, (x_pf, x_sv, jet_features, jet_truthlabel) in enumerate(tqdm(val_loader)):
                #model.eval()
                x_pf = x_pf.to(self.gpu_id)
                x_sv = x_sv.to(self.gpu_id)
                jet_features = jet_features.to(self.gpu_id)
                jet_truthlabel = jet_truthlabel.to(self.gpu_id)
                if 'all_vs_QCD' in self.args.loss:
                    jet_truthlabel = jet_truthlabel[:,:-1]      
                if (self.args.test_run and istep>10 ): break
                if self.args.sv:
                    predictions.append(nn.Softmax(dim=1)(self.model(x_pf,x_sv)).cpu().detach().numpy())
                else:
                    predictions.append(nn.Softmax(dim=1)(self.model(x_pf)).cpu().detach().numpy())
                testingLabels.append(jet_truthlabel.cpu().detach().numpy())
                testingSingletons.append(jet_features.cpu().detach().numpy())
                torch.cuda.empty_cache()
                #break
            rand_x_pf = torch.randn(1,100,13).to(self.gpu_id)
        
        if self.args.sv:
            rand_x_sv = torch.rand(10,5,16).to(self.gpu_id)
#             torch.onnx.export(self.model.module,
#                   (rand_x_pf,rand_x_sv),
#                   f"{self.outdir}/"+self.args.mname+".onnx",
#                   export_params=True,
#                   opset_version=12,
#                   do_constant_folding=True,
#                   input_names=['pf','sv'],
#                   output_names=["outputs"],
#                   dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}},
#             )
        else: 
#             print(self.model)
#             print(self.model.module)
            torch.onnx.export(self.model.module,
                  rand_x_pf,
                  f"{self.outdir}/"+self.args.mname+".onnx",
                  export_params=True,
                  opset_version=12,
                  do_constant_folding=True,
                  input_names=['pf'],
                  output_names=["outputs"],
                  dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}},
            )
        self.outdir = self.outdir + '/plots'
        predictions = [item for sublist in predictions for item in sublist]
        testingLabels = [item for sublist in testingLabels for item in sublist]
        testingSingletons = [item for sublist in testingSingletons for item in sublist]
        predictions = np.array(predictions)#.astype(np.float32)
        testingLabels = np.array(testingLabels)
        testingSingletons = np.array(testingSingletons)
        
        if self.args.plot_features:
            print("Plotting all features. This might take a few minutes")
            utils.plot_features(singletonData,labels,utils._singleton_labels,args.opath)
            utils.plot_features(vertexData,labels,utils._SV_features_labels,args.opath,"SV")
            utils.plot_features(particleData,labels,utils._p_features_labels,args.opath,"Particle")
            utils.plot_features(singletonFeatureData,labels,utils._singleton_features_labels,args.opath)
            
        
    
        os.system("mkdir -p "+self.outdir)
        np.savez(self.outdir+"/predictions.npy", predictions=predictions, labels=testingLabels, singletons=testingSingletons)
        if 'all_vs_QCD' in self.args.loss:
            qcd_idxs = np.where(testingLabels.sum(axis=1)==0,True,False)
        else:
            qcd_idxs = testingLabels[:,-1].astype(bool)
            utils.plot_correlation(predictions[qcd_idxs,-1],testingSingletons[qcd_idxs,utils._singleton_labels.index("zpr_fj_msd")], "QCD output score","QCD jet $m_{SD}$ (GeV)", np.linspace(0,1,50),np.linspace(40,350,40),self.outdir, "qcd_vs_mass")
            utils.sculpting_curves(predictions[qcd_idxs,-1], testingSingletons[qcd_idxs,:], training_text, self.outdir, self.name, score="QCD", inverted=False)
        
        
        #Plotting ROC Over all pT ranges:
        utils.plot_roc_curve(testingLabels, predictions, training_text, self.outdir, self.name, all_vs_QCD="all_vs_QCD" in self.args.loss, QCD_only=False)
        for (pt_min,pt_max) in [[0,400],[400,600],[600,800],[800,1000],[1000,1200],[1200,5000]]: 
            
            mask = (testingSingletons[:, 0] < pt_max) & (testingSingletons[:, 0] > pt_min)
            utils.plot_roc_curve(testingLabels, predictions[mask], training_text, self.outdir, self.name+f'pt_{pt_min}_{pt_max}', all_vs_QCD="all_vs_QCD" in self.args.loss, QCD_only=False)
            
            '''EX of data usage: predictions[testLabels[:,itruth]>0,ilabel]
            Seems that predictions is Nx4 
            
            testingSingletons stores the jet_features
            pT is jet_feature[0]
            
            testingSingletons[:,0] < pt_max and testingSingletons[:,0] > pt_min should be the mask
            
            '''
        utils.plot_features(testingSingletons,testingLabels,utils._singleton_labels,self.outdir)
    
        if self.args.is_binary:
            
            prob_2prong = predictions[qcd_idxs,0]
            utils.sculpting_curves(prob_2prong, testingSingletons[qcd_idxs,:], training_text, self.outdir, self.name, score="Z\'",inverted=True)
    
        else:
            utils.plot_correlation(predictions[qcd_idxs,0],testingSingletons[qcd_idxs,utils._singleton_labels.index("zpr_fj_msd")], "bb vs QCD output score","QCD jet $m_{SD}$ (GeV)", np.linspace(0,1,100),np.linspace(40,350,40),self.outdir, "bb_vs_mass")
            utils.plot_correlation(predictions[qcd_idxs,1],testingSingletons[qcd_idxs,utils._singleton_labels.index("zpr_fj_msd")], "cc vs QCD output score","QCD jet $m_{SD}$ (GeV)", np.linspace(0,1,100),np.linspace(40,350,40),self.outdir, "cc_vs_mass")
            utils.plot_correlation(predictions[qcd_idxs,2],testingSingletons[qcd_idxs,utils._singleton_labels.index("zpr_fj_msd")], "qq vs QCD output score","QCD jet $m_{SD}$ (GeV)", np.linspace(0,1,100),np.linspace(40,350,40),self.outdir, "qq_vs_mass")
            prob_bb = predictions[qcd_idxs,0]
            utils.sculpting_curves(prob_bb, testingSingletons[qcd_idxs,:], training_text, self.outdir, self.name, score="bb",inverted=True)
            prob_cc = predictions[qcd_idxs,1]
            utils.sculpting_curves(prob_cc, testingSingletons[qcd_idxs,:], training_text, self.outdir, self.name, score="cc",inverted=True)
            prob_qq = predictions[qcd_idxs,2]
            utils.sculpting_curves(prob_qq, testingSingletons[qcd_idxs,:], training_text, self.outdir, self.name, score="qq",inverted=True)
    
        predictionsPN = testingSingletons[:,[utils._singleton_labels.index("zpr_fj_particleNetMD_Xbb"), utils._singleton_labels.index("zpr_fj_particleNetMD_Xcc"), utils._singleton_labels.index("zpr_fj_particleNetMD_Xqq"), utils._singleton_labels.index("zpr_fj_particleNetMD_QCD")]]
        utils.plot_roc_curve(testingLabels, predictionsPN, self.args.plot_text, self.outdir, "particleNet-MD", all_vs_QCD=False,QCD_only=False)
        
        qcd_idxs = testingLabels[:,-1].astype(bool)
        prob_bb = predictionsPN[qcd_idxs,0]
        utils.sculpting_curves(prob_bb, testingSingletons[qcd_idxs,:],"ParticleNet-MD:bb score", self.outdir, "particleNet-MD-bb",inverted=True)
        prob_cc = predictionsPN[qcd_idxs,1]
        utils.sculpting_curves(prob_cc, testingSingletons[qcd_idxs,:], "ParticleNet-MD:cc score", self.outdir, "particleNet-MD-cc",inverted=True)
        prob_qq = predictionsPN[qcd_idxs,2]
        utils.sculpting_curves(prob_qq, testingSingletons[qcd_idxs,:], "ParticleNet-MD:qq score", self.outdir, "particleNet-MD-qq",inverted=True)
        prob_QCD = predictionsPN[qcd_idxs,3]
        utils.sculpting_curves(prob_QCD, testingSingletons[qcd_idxs,:], "ParticleNet-MD:QCD score", self.outdir, "particleNet-MD-QCD",inverted=False)
        testingLabelsPN = np.concatenate((np.expand_dims(np.sum(testingLabels[:,:-1],axis=1),-1),np.expand_dims(testingLabels[:,-1],-1)),axis=1)
        predictionsPN = np.concatenate((np.expand_dims(np.sum(predictionsPN[:,:-1],axis=1),-1),np.expand_dims(predictionsPN[:,-1],-1)),axis=1)
        utils.plot_roc_curve(testingLabelsPN, predictionsPN, self.args.plot_text, self.outdir, "particleNet-MD-2prong", all_vs_QCD=False, QCD_only=False)
        #sys.exit(1)
#         predictionsN2 = testingSingletons[:,[utils._singleton_labels.index("zpr_fj_n2b1")]]
#         predictionsN2 = (predictionsN2 - np.min(predictionsN2)) / ( np.max(predictionsN2) - np.min(predictionsN2))
#         predictionsN2 = np.concatenate((1-predictionsN2,predictionsN2),axis=1)
#         utils.plot_roc_curve(labels, predictionsN2, self.args.plot_text, self.outdir, "N2", all_vs_QCD=False,qcd_only=False)
#         qcd_idxs = labels[:,-1].astype(bool)
#         prob_QCD = predictionsN2[qcd_idxs,0]
#         utils.sculpting_curves(prob_QCD, testingSingletons[qcd_idxs,:],"N2:QCD score", self.outdir, "N2",inverted=True)
#         prob_N2 = predictionsN2[qcd_idxs,1]
#         utils.sculpting_curves(prob_N2, testingSingletons[qcd_idxs,:],"N2:2prong score", self.outdir, "N2",inverted=False)
        
        #sys.exit(1)