#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports basics
import os
import numpy as np
import h5py
import json
import setGPU
import sklearn
import corner
import scipy
import time
from tqdm import tqdm 
import argparse

# Imports neural net tools
import itertools
import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
import torch.nn.functional as F
from fast_soft_sort.pytorch_ops import soft_rank
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score,  auc
plt.ioff()

import matplotlib
matplotlib.use('Agg')

# In[ ]:


parser = argparse.ArgumentParser(description='Test.')
parser.add_argument('--weightcov',  action='store', default=1., type=float, help='Covariance f.')
parser.add_argument('--weightrepr', action='store', default=1., type=float, help='MSE (attractive term) weight.')
parser.add_argument('--weightstd',  action='store',  default=1., type=float, help='Variance (repulsive term) weight.')
parser.add_argument('--weightCorr1', action='store', default=0., type=float, help='Correlation weight.')
parser.add_argument('--weightCorr2', action='store', default=0., type=float, help='Anti-correlation weight.')
parser.add_argument('--nepochs', action='store', default=1, type=int, help='Number of training epochs.')
args = parser.parse_args()

print(args)
# Opens files and reads data

print("Extracting")
#fOne = np.load("/n/holyscratch01/iaifi_lab/jkrupa/10Mar22-MiniAODv2/18May22-morevars-v3_test/zpr_fj_msd/2017/total.npz")
fOne = np.load("/work/tier3/jkrupa/FlatSamples/total.npz")
#totalData = fOne["deepDoubleQ"][:]
#print(totalData.shape)


# In[ ]:

weightrepr = args.weightrepr
weightcov = args.weightcov #(most useful)
weightstd = args.weightstd
#weightclr = 1
weightCorr1 = args.weightCorr1 #(most useful in barlow)
weightCorr2 = args.weightCorr2 #(not really useful but could explore)
batchSize = 6000
n_Dim = 4
n_epochs = args.nepochs
CorrDim = 1
mod = "vicreg"
label='Contrastive'+'_n_epochs'+str(n_epochs)+'_ndim'+str(n_Dim)+'_batchSize'+str(batchSize) + '_weightrepr'+str(weightrepr) + '_weightcov'+str(weightcov) + '_weightstd'+str(weightstd) + '_weightCorr1'+str(weightCorr1) + '_weightCorr2'+str(weightCorr2)
modelName = "DNN_FlatSamples_flatratio_" + label + mod
outdir = '/home/tier3/jkrupa/public_html/zprlegacy/cl_oct13/' + modelName #everything will output here
try: 
    os.system("mkdir -p "+outdir) 
except OSError as error: 
    print(error)

# Sets controllable values

particlesConsidered = 150
particlesPostCut = 150
entriesPerParticle = 6
eventDataFeatures = ['jet_eta', 'jet_phi', 'jet_EhadOverEem', 'jet_mass', 'jet_pt', 
                 'jet_sdmass', 'ecfns_2_1', 'ecfns_3_2', 'N2']

_singletons=["zpr_fj_msd","zpr_fj_pt","zpr_fj_eta","zpr_fj_phi","zpr_fj_n2b1","zpr_fj_tau21","zpr_fj_particleNetMD_QCD", "zpr_fj_particleNetMD_Xbb", "zpr_fj_particleNetMD_Xcc", "zpr_fj_particleNetMD_Xqq","zpr_fj_nBHadrons","zpr_fj_nCHadrons", "zpr_genAK8Jet_mass","zpr_genAK8Jet_pt","zpr_genAK8Jet_eta","zpr_genAK8Jet_phi", "zpr_genAK8Jet_partonFlavour","zpr_genAK8Jet_hadronFlavour", "zpr_fj_nBtags","zpr_fj_nCtags","zpr_fj_nLtags",]

eventDataFeatures=["zpr_fj_jetNSecondaryVertices","zpr_fj_jetNTracks","zpr_fj_tau1_trackEtaRel_0","zpr_fj_tau1_trackEtaRel_1","zpr_fj_tau1_trackEtaRel_2","zpr_fj_tau2_trackEtaRel_0","zpr_fj_tau2_trackEtaRel_1","zpr_fj_tau2_trackEtaRel_3","zpr_fj_tau1_flightDistance2dSig","zpr_fj_tau2_flightDistance2dSig","zpr_fj_tau1_vertexDeltaR","zpr_fj_tau1_vertexEnergyRatio","zpr_fj_tau2_vertexEnergyRatio","zpr_fj_tau1_vertexMass","zpr_fj_tau2_vertexMass","zpr_fj_trackSip2dSigAboveBottom_0","zpr_fj_trackSip2dSigAboveBottom_1","zpr_fj_trackSip2dSigAboveCharm","zpr_fj_trackSip3dSig_0","zpr_fj_trackSip3dSig_0","zpr_fj_tau1_trackSip3dSig_1","zpr_fj_trackSip3dSig_1","zpr_fj_tau2_trackSip3dSig_0","zpr_fj_tau2_trackSip3dSig_1","zpr_fj_trackSip3dSig_2","zpr_fj_trackSip3dSig_3","zpr_fj_z_ratio"]

_p_features=["zpr_PF_ptrel","zpr_PF_etarel","zpr_PF_phirel","zpr_PF_dz","zpr_PF_d0","zpr_PF_pdgId"]
_SV_features=["zpr_SV_mass","zpr_SV_dlen","zpr_SV_dlenSig","zpr_SV_dxy","zpr_SV_dxySig","zpr_SV_chi2","zpr_SV_ptrel","zpr_SV_x","zpr_SV_y","zpr_SV_z","zpr_SV_pAngle","zpr_SV_etarel","zpr_SV_phirel"]


eventDataLength = len(eventDataFeatures)
decayTypeColumn = -1
datapoints = fOne["singletons"].shape[0]
trainingDataLength = int(datapoints*0.8)
validationDataLength = int(datapoints*0.1)


# In[ ]:


# Creates Training Data

print("Preparing Data")

particleDataLength = particlesConsidered * entriesPerParticle

np.random.seed(42)
#np.random.shuffle(totalData)

#trainingDataLength = int(datapoints*0.8)
#validationDataLength = int(datapoints*0.1)

#mask = [i>90 and i<110 for i in totalData[:, eventDataLength-1] ]
#totalData = totalData[mask]

labels = fOne["singletons"][:, -3:]
labels = np.sum(labels,axis=1)
labels[labels==2]=1
singletonData = fOne["singletons"][:]
particleData = fOne["p_features"][:]
eventData = fOne["singleton_features"][:]
jetMassData = fOne["singletons"][:, 0] #last entry in eventData (zero indexing)
print(jetMassData)
labels, singletonData, particleData, eventData, jetMassData = sklearn.utils.shuffle(labels, singletonData, particleData, eventData, jetMassData)

######### Training Data ###############
eventTrainingData = np.array(eventData[0:trainingDataLength])
jetMassTrainingData = np.array(jetMassData[0:trainingDataLength])


particleTrainingData = np.transpose(
    particleData[0:trainingDataLength, ].reshape(trainingDataLength, 
                                                 entriesPerParticle, 
                                                 particlesConsidered),
                                                 axes=(0, 2, 1))
#trainingLabels = np.array(labels[0:trainingDataLength]).reshape((-1,2))
trainingLabels = np.array([[i, 1-i] for i in labels[0:trainingDataLength]]).reshape((-1, 2))
print(trainingLabels)
print(particleTrainingData.shape)

########## Validation Data ##########
eventValidationData = np.array(eventData[trainingDataLength:trainingDataLength + validationDataLength])
jetMassValidationData = np.array(jetMassData[trainingDataLength:trainingDataLength + validationDataLength])
particleValidationData = np.transpose(
    particleData[trainingDataLength:trainingDataLength + validationDataLength, ].reshape(validationDataLength,
                                                                                         entriesPerParticle,
                                                                                         particlesConsidered),
                                                                                         axes=(0, 2, 1))
validationLabels = np.array([[i, 1-i] for i in labels[trainingDataLength:trainingDataLength + validationDataLength]]).reshape((-1, 2))
print(particleValidationData.shape)


########### Testing Data ############
eventTestData = np.array(eventData[trainingDataLength + validationDataLength:])
jetMassTestData = np.array(jetMassData[trainingDataLength + validationDataLength:])
particleTestData = np.transpose(particleData[trainingDataLength + validationDataLength:,].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerParticle, particlesConsidered),
                                axes=(0, 2, 1))
testLabels = np.array([[i, 1-i] for i in labels[trainingDataLength + validationDataLength:]]).reshape((-1, 2))
print("testLabels:",testLabels)
print('Selecting particlesPostCut')
particleTrainingData = particleTrainingData[:, :, :]
particleValidationData = particleValidationData[:, :, :]#particlesPostCut]
particleTestData = particleTestData[:,:, :]#particlesPostCut]

particlesConsidered = particlesPostCut

print(particleTrainingData.shape)

# In[ ]:


# Look at the data a bit!
# Jet mass for correlation
jetMassTrainingDataSig = jetMassTrainingData[trainingLabels[:,0].astype(bool)]
jetMassTrainingDataBkg = jetMassTrainingData[trainingLabels[:,1].astype(bool)]

fig,ax = plt.subplots()
ax.hist(jetMassTestData[testLabels[:,0].astype(bool)],alpha=0.7,label="Z'")
ax.hist(jetMassTestData[testLabels[:,1].astype(bool)],alpha=0.7,label="QCD")
plt.savefig(outdir+"/jet_mass_testing_data.png")
plt.savefig(outdir+"/jet_mass_testing_data.pdf")
print(jetMassTrainingDataSig.shape)
print(jetMassTrainingDataBkg.shape)



# In[ ]:


# Defines the interaction matrices
class GraphNetnoSV(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden, De=5, Do=6, softmax=False):
        super(GraphNetnoSV, self).__init__()
        self.hidden = int(hidden)
        self.P = params
        self.Nv = 0 
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Nt = self.N * self.Nv
        self.Ns = self.Nv * (self.Nv - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.S = 0
        self.n_targets = n_targets
        self.assign_matrices()
        self.softmax = softmax
           
        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden).cuda()
        self.fr2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3 = nn.Linear(int(self.hidden/2), self.De).cuda()
        self.fr1_pv = nn.Linear(self.S + self.P + self.Dr, self.hidden).cuda()
        self.fr2_pv = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3_pv = nn.Linear(int(self.hidden/2), self.De).cuda()
        
        self.fo1 = nn.Linear(self.P + self.Dx + (self.De), self.hidden).cuda()
        self.fo2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fo3 = nn.Linear(int(self.hidden/2), self.Do).cuda()
        
        self.fc_fixed = nn.Linear(self.Do, self.n_targets).cuda()
            
    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = (self.Rr).cuda()
        self.Rs = (self.Rs).cuda()

    def forward(self, x):
        ###PF Candidate - PF Candidate###
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
        B = nn.functional.relu(self.fr2(B))
        E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_pp = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        

        ####Final output matrix for particles###
        

        C = torch.cat([x, Ebar_pp], 1)
        del Ebar_pp
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + (self.De))))
        C = nn.functional.relu(self.fo2(C))
        O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C

        
        #Taking the sum of over each particle/vertex
        N = torch.sum(O, dim=1)
        del O
        
        ### Classification MLP ###

        N = self.fc_fixed(N)
        
        if self.softmax:
            N = nn.Softmax(dim=1)(N)
        
        return N
            
    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])
    
class DNN(nn.Module):
    def __init__(self, n_DimLatent, n_Inputs):
        super(DNN, self).__init__()
        #self.flat = torch.flatten()
        self.f0 = nn.Linear(n_Inputs, 1000).cuda()
        self.f0c = nn.Linear(1000, 400).cuda()
        self.f0d = nn.Linear(400, 200).cuda()
        self.f1 = nn.Linear(200, 100).cuda()
        self.f2 = nn.Linear(100, 50).cuda()
        self.f3 = nn.Linear(50, 10).cuda()
        self.f4 = nn.Linear(10, n_DimLatent).cuda()
        self.activation = torch.nn.ReLU()
        self.lastactivation = torch.nn.Softmax()
    def forward(self, x): 
        x = torch.flatten(x,start_dim=1)
        x = self.activation(self.f0(x))
        x = self.activation(self.f0c(x))
        x = self.activation(self.f0d(x))
        x = self.activation(self.f1(x))
        x = self.activation(self.f2(x))
        x = self.activation(self.f3(x))
        x = self.f4(x)
        #x = self.lastactivation(self.f4(x))
        return(x)

class MLP(nn.Module):
    def __init__(self, n_inputs, n_targets):
        super(MLP, self).__init__()
        self.f1 = nn.Linear(n_inputs, n_inputs).cuda()
        self.f2 = nn.Linear(n_inputs, int(n_inputs/2)).cuda()
        self.f3 = nn.Linear(int(n_inputs/2), int(n_inputs/10)).cuda()
        self.f4 = nn.Linear(int(n_inputs/10), n_targets).cuda()
        self.activation = torch.nn.Softmax()
    def forward(self, x): 
        x = torch.flatten(x,start_dim=1)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return(self.activation(x))

class MLP2(nn.Module):
    def __init__(self, n_inputs, n_targets):
        super(MLP2, self).__init__()
        self.f1 = nn.Linear(n_inputs, n_inputs*10).cuda()
        #self.f2 = nn.Linear(n_inputs*10, n_inputs*5).cuda()
        #self.f3 = nn.Linear(n_inputs*5, n_inputs).cuda()
        #self.f4 = nn.Linear(n_inputs, int(n_inputs/2)).cuda()
        #self.f5 = nn.Linear(int(n_inputs/2), n_targets).cuda()
        self.activation = torch.nn.Softmax()
    def forward(self, x): 
        x = torch.flatten(x,start_dim=1)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        return(self.activation(x))

class Linear(nn.Module):
    def __init__(self, n_inputs, n_targets):
        super(Linear, self).__init__()
        self.f1 = nn.Linear(n_inputs, n_inputs*10).cuda()
        self.f2 = nn.Linear(n_inputs*10, n_inputs*5).cuda()
        self.f3 = nn.Linear(n_inputs*5, n_targets).cuda()
        self.activation = torch.nn.Sigmoid()
    def forward(self, x): 
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return(self.activation(x))


# In[ ]:


# Define losses 
class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = torch.device('cuda:0')

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        #self.device = (torch.device('cuda')if z_a.is_cuda else torch.device('cpu'))
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D, device=self.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()
        return loss

    
# return a flattened view of the off-diagonal elements of a square matrix
def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class VICRegLoss(torch.nn.Module):

    def __init__(self, lambda_param=1,mu_param=1,nu_param=20):
        super(VICRegLoss, self).__init__()
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.nu_param = nu_param
        #self.device = torch.device('cpu')

    def forward(self, x, y):
        self.device = (torch.device('cuda')if x.is_cuda else torch.device('cpu'))
        
        x_scale = x
        y_scale = y
        repr_loss = F.mse_loss(x_scale, y_scale)
        
        #x = torch.cat(FullGatherLayer.apply(x), dim=0)
        #y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x_scale = x_scale - x_scale.mean(dim=0)
        y_scale = y_scale - y_scale.mean(dim=0)
        N = x_scale.size(0)
        D = x_scale.size(1)
        
        std_x = torch.sqrt(x_scale.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y_scale.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        x_scale = x_scale/std_x
        y_scale = y_scale/std_y

        cov_x = (x_scale.T @ x_scale) / (N - 1)
        cov_y = (y_scale.T @ y_scale) / (N - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(D) + off_diagonal(cov_y).pow_(2).sum().div(D)

        #loss = (self.lambda_param * repr_loss + self.mu_param * std_loss+ self.nu_param * cov_loss)
        #print(repr_loss,cov_loss,std_loss)
        return repr_loss,cov_loss,std_loss
    
class CorrLoss(nn.Module):
    def __init__(self, corr=False,sort_tolerance=1.0,sort_reg='l2'):
        super(CorrLoss, self).__init__()
        self.tolerance = sort_tolerance
        self.reg       = sort_reg
        self.corr      = corr
        
    def spearman(self, pred, target):
        pred   = soft_rank(pred.cpu().reshape(1,-1),regularization=self.reg,regularization_strength=self.tolerance,)
        target = soft_rank(target.cpu().reshape(1,-1),regularization=self.reg,regularization_strength=self.tolerance,)
        #pred   = torchsort.soft_rank(pred.reshape(1,-1),regularization_strength=x)
        #target = torchsort.soft_rank(target.reshape(1,-1),regularization_strength=x)
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()
        ret = (pred * target).sum()
        if self.corr:
            return (1-ret)*(1-ret)
        else:
            return ret*ret 
    
    def forward(self, features, labels):
        return self.spearman(features,labels)


# In[ ]:


# Separate Encoder & Classifier training 

def train_encoder(encoder, batchSize, n_Dim, CorrDim, n_epochs, modelName, outdir,
                  particleTrainingData, particleValidationData, trainingLabels, jetMassTrainingData, jetMassValidationData,
                  weightrepr = 1, weightcov = 1, weightstd = 1, weightCorr1 = 0, weightCorr2 = 0):
    
    # Separating signal and bkg arrays
    particleTrainingDataSig = particleTrainingData[trainingLabels[:,0].astype(bool)]
    particleTrainingDataBkg = particleTrainingData[trainingLabels[:,1].astype(bool)]
    particleValidationDataSig = particleValidationData[validationLabels[:,0].astype(bool)]
    particleValidationDataBkg = particleValidationData[validationLabels[:,1].astype(bool)]
    particleTrainingLabelSig = trainingLabels[trainingLabels[:,0].astype(bool)]
    particleTrainingLabelBkg = trainingLabels[trainingLabels[:,1].astype(bool)]

    # Jet mass for correlation
    jetMassTrainingDataSig = jetMassTrainingData[trainingLabels[:,0].astype(bool)]
    jetMassTrainingDataBkg = jetMassTrainingData[trainingLabels[:,1].astype(bool)]
    jetMassValidationDataSig = jetMassValidationData[validationLabels[:,0].astype(bool)]
    jetMassValidationDataBkg = jetMassValidationData[validationLabels[:,1].astype(bool)]
    
    peaky_sig = False
    if peaky_sig: 
        mask_train = [mass<110 and mass>60 for mass in jetMassTrainingDataSig]
        mask_val = [mass<110 and mass>60 for mass in jetMassValidationDataSig]
        particleTrainingDataSig = particleTrainingDataSig[mask_train]
        particleTrainingLabelSig = particleTrainingLabelSig[mask_train]
        jetMassTrainingDataSig = jetMassTrainingDataSig[mask_train]
        particleValidationDataSig = particleValidationDataSig[mask_val]
        jetMassValidationDataSig = jetMassValidationDataSig[mask_val]
        
    try: 
        os.mkdir(outdir) 
    except OSError as error: 
        print(error)
    clr_criterion  = VICRegLoss(lambda_param=1,mu_param=1,nu_param=1)
    #clr_criterion = BarlowTwinsLoss()
    cor_criterion  = CorrLoss()
    acr_criterion  = CorrLoss(corr=True)

    optimizer = optim.Adam(encoder.parameters(), lr = 0.001)
    loss_vals_training = [] #np.zeros(n_epochs)
    loss_vals_validation = [] #np.zeros(n_epochs)

    final_epoch = 0
    l_val_best = 99999

    epoch_idx = 0 
    for m in range(n_epochs):
        print("Epoch %s\n" % m)
        #torch.cuda.empty_cache()
        final_epoch = m
        lst = []
        loss_val = []
        loss_training = []
        correct = []
        tic = time.perf_counter()

        particleTrainingDataSig, jetMassTrainingDataSig = sklearn.utils.shuffle(particleTrainingDataSig, jetMassTrainingDataSig)
        particleTrainingDataBkg, jetMassTrainingDataBkg = sklearn.utils.shuffle(particleTrainingDataBkg, jetMassTrainingDataBkg)
        particleValidationDataSig, jetMassValidationDataSig = sklearn.utils.shuffle(particleValidationDataSig,
                                                                                    jetMassValidationDataSig)
        particleValidationDataBkg, jetMassValidationDataBkg = sklearn.utils.shuffle(particleValidationDataBkg,
                                                                                    jetMassValidationDataBkg)

        totaldiv2 = min(len(particleTrainingDataSig), len(particleTrainingDataBkg))
        for i in tqdm(range(int(totaldiv2/batchSize))): 
            optimizer.zero_grad()

            # Define training events
            trainingvMassSig, SigSort = torch.FloatTensor(jetMassTrainingDataSig[i*batchSize:(i+1)*batchSize]).cuda().sort()
            #trainingvMassSig = torch.FloatTensor(jetMassTrainingDataSig[i*batchSize:(i+1)*batchSize]).cuda()
            trainingvMassBkg, BkgSort = torch.FloatTensor(jetMassTrainingDataBkg[i*batchSize:(i+1)*batchSize]).cuda().sort()
            #trainingvMassBkg = torch.FloatTensor(jetMassTrainingDataBkg[i*batchSize:(i+1)*batchSize]).cuda()
            trainingvSig = torch.FloatTensor(particleTrainingDataSig[i*batchSize:(i+1)*batchSize]).cuda()[SigSort]
            trainingvBkg = torch.FloatTensor(particleTrainingDataBkg[i*batchSize:(i+1)*batchSize]).cuda()[BkgSort]
            trainingv1 = torch.cat((trainingvSig[:int(batchSize/2)], 
                                    trainingvBkg[:int(batchSize/2)]))
            trainingv1_mass = torch.cat(( trainingvMassSig[:int(batchSize/2)], 
                                          trainingvMassBkg[:int(batchSize/2)]))
            trainingv2 = torch.cat((trainingvSig[int(batchSize/2):], 
                                    trainingvBkg[int(batchSize/2):]))
            trainingv2_mass = torch.cat((trainingvMassSig[int(batchSize/2):], 
                                         trainingvMassBkg[int(batchSize/2):]))
            # Calculate network output
            out1 = encoder(trainingv1)
            out2 = encoder(trainingv2)

            #VICReg Loss
            repr_loss, cov_loss, std_loss = clr_criterion(out1, out2)
            #l = weightclr*clr_criterion(out1, out2)

            l = weightrepr*repr_loss + weightcov*cov_loss + weightstd*std_loss
            # For Clara: these can be commented out if not in use to make things run faster
            # Anti-Correlation (actually correlation)
            for dim in range(CorrDim):
                l += weightCorr1*acr_criterion(trainingv1_mass, out1[:,dim])
                l += weightCorr1*acr_criterion(trainingv2_mass, out2[:,dim])
            # Correlation for rest of dimensions (anti-correlation)
            #for dim in range(1): 
            for dim in range(out1.shape[1]-CorrDim): 
                l += weightCorr2*(dim+1)*cor_criterion(out1[:,dim+CorrDim], trainingv1_mass)
                l += weightCorr2*(dim+1)*cor_criterion(out2[:,dim+CorrDim], trainingv2_mass)

            loss_training.append(l.item())
            l.backward()
            optimizer.step()
            loss_string = "Loss: %s" % "{0:.5f}".format(l.item())
            del trainingvSig, trainingvBkg, trainingv1_mass, trainingv2_mass, trainingv1, trainingv2, out1, out2
            torch.cuda.empty_cache()

        toc = time.perf_counter()
        print(f"Training done in {toc - tic:0.4f} seconds")
        tic = time.perf_counter()
        out1_totSig = np.empty((0,n_Dim))
        out1_totBkg = np.empty((0,n_Dim))

        trainingv1_mass_totSig = []
        trainingv1_mass_totBkg = []

        
        out_val_total_sig = []
        out_val_total_bkg = []
        out_val_mass_total_sig = []
        out_val_mass_total_bkg = []

        totaldiv2 = min(len(particleValidationDataSig), len(particleValidationDataBkg))
        for i in range(int(totaldiv2/batchSize)):
            torch.cuda.empty_cache()

            # Define validation events
            trainingvSig_val = torch.FloatTensor(particleValidationDataSig[i*batchSize:(i+1)*batchSize]).cuda()
            trainingvBkg_val = torch.FloatTensor(particleValidationDataBkg[i*batchSize:(i+1)*batchSize]).cuda()
            trainingvMassSig_val = torch.FloatTensor(jetMassValidationDataSig[i*batchSize:(i+1)*batchSize]).cuda()
            trainingvMassBkg_val = torch.FloatTensor(jetMassValidationDataBkg[i*batchSize:(i+1)*batchSize]).cuda()
            targetv_val = torch.FloatTensor(validationLabels[i*batchSize:(i+1)*batchSize]).cuda()
            trainingv1_val = torch.cat((trainingvSig_val[:int(batchSize/2)], trainingvBkg_val[:int(batchSize/2)]))
            trainingv2_val = torch.cat((trainingvSig_val[int(batchSize/2):], trainingvBkg_val[int(batchSize/2):]))
            trainingv1_val_mass = torch.cat((trainingvMassSig_val[:int(batchSize/2)], 
                                    trainingvMassBkg_val[:int(batchSize/2)]))
            trainingv2_val_mass = torch.cat((trainingvMassSig_val[int(batchSize/2):], 
                                    trainingvMassBkg_val[int(batchSize/2):]))


            # For use in making plots later in epoch
            out_val_total_sig.append(encoder(trainingvSig_val).cpu().detach().numpy())
            out_val_mass_total_sig.append(trainingvMassSig_val.cpu().detach().numpy())
            out_val_total_bkg.append(encoder(trainingvBkg_val).cpu().detach().numpy())
            out_val_mass_total_bkg.append(trainingvMassBkg_val.cpu().detach().numpy())

            # VICReg Loss
            out1_val = encoder(trainingv1_val)
            out2_val = encoder(trainingv2_val)
            repr_loss, cov_loss, std_loss = clr_criterion(out1_val, out2_val)
            #l_val = weightclr*clr_criterion(out1_val, out2_val)
            l_val = weightrepr*repr_loss + weightcov*cov_loss + weightstd*std_loss

            # For Clara: these can be commented out if not in use to make things run faster
            # AntiCorrelation
            for dim in range(CorrDim): 
                l_val += weightCorr1*acr_criterion(trainingv1_val_mass, out1_val[:,dim])
                l_val += weightCorr1*acr_criterion(trainingv2_val_mass, out2_val[:,dim])
            # Correlation for rest of dimensions
            #for dim in range(1):
            for dim in range(out1_val.shape[1]-CorrDim): 
                l_val += weightCorr2*(dim+1)*cor_criterion(out1_val[:,dim+CorrDim], trainingv1_val_mass)
                l_val += weightCorr2*(dim+1)*cor_criterion(out2_val[:,dim+CorrDim], trainingv2_val_mass)
            
            # Classical validation
            loss_val.append(l_val.item())

            del trainingvSig_val, trainingvBkg_val, trainingv1_val, trainingv2_val, out1_val, out2_val
            torch.cuda.empty_cache()
            
        out_val_total_sig = np.array(out_val_total_sig).reshape(-1, n_Dim)
        out_val_total_bkg = np.array(out_val_total_bkg).reshape(-1, n_Dim)
        out_val_mass_total_sig = np.array(out_val_mass_total_sig).reshape(-1, 1)
        out_val_mass_total_bkg = np.array(out_val_mass_total_bkg).reshape(-1, 1)
        
        fig,ax = plt.subplots()    
        plt.clf()
        fig, axs = plt.subplots(n_Dim,2, figsize=(10,n_Dim*10))

        axs[0,0].text(0.05,2.8, loss_text, transform=ax.transAxes)

        for dim in range(n_Dim): 
            outSig, massSig = out_val_total_sig[:, dim].copy(), out_val_mass_total_sig[:].copy()
            outSig -= np.mean(outSig)
            outSig /= np.std(outSig)
            massSig -= np.mean(massSig)
            massSig /= np.std(massSig)

            outBkg, massBkg = out_val_total_bkg[:, dim].copy(), out_val_mass_total_bkg[:].copy()
            outBkg -= np.mean(outBkg)
            outBkg /= np.std(outBkg)
            massBkg -= np.mean(massBkg)
            massBkg /= np.std(massBkg)

            outSig = outSig.reshape(-1)
            outBkg = outBkg.reshape(-1)
            massSig = massSig.reshape(-1)
            massBkg = massBkg.reshape(-1)
            
            axs[dim,0].text(0.8,1.03,f"Z' Corr:  {np.corrcoef(outSig, massSig)[0,1] : .4f}", transform=axs[dim,0].transAxes)
            axs[dim,0].hist2d(outSig, out_val_mass_total_sig.reshape(-1), bins=30, )
            axs[dim,1].text(0.8,1.03,f"QCD Corr: {np.corrcoef(outBkg, massBkg)[0,1] : .4f}", transform=axs[dim,1].transAxes)
            axs[dim,1].hist2d(outBkg, out_val_mass_total_bkg.reshape(-1), bins=30, )
            axs[dim,0].set_xlim([-3.,3.])
            axs[dim,1].set_xlim([-3.,3.])
            axs[dim,0].set_xlabel(f'Dimension {dim} output')
            axs[dim,1].set_xlabel(f'Dimension {dim} output')
            axs[dim,0].set_ylabel('Jet mass (GeV)')
        plt.legend(loc="best")
        plt.savefig(outdir+"/"+modelName+f"_contrastivefigIN_trainingDataset_epoch{m}.png")
        plt.savefig(outdir+"/"+modelName+f"_contrastivefigIN_trainingDataset_epoch{m}.pdf")

        try: 
            label_str = ["latent var %s"%str(i) for i in range(n_Dim)]
            label_str.append("mass")
            fig = corner.corner(np.concatenate((out_val_total_sig, out_val_mass_total_sig.reshape(-1, 1)), axis=1), color='red', labels=label_str)
            corner.corner(np.concatenate((out_val_total_bkg, out_val_mass_total_bkg.reshape(-1, 1)), axis=1), fig=fig, color='blue', labels=label_str)
            fig.text(0.45,1.05,loss_text, transform=ax.transAxes, fontsize=16)
            fig.savefig('%s/CornerPlot_%s.png'%(outdir,modelName))
            fig.savefig('%s/CornerPlot_%s.pdf'%(outdir,modelName))
        except: 
            # Can happen at beginning of trainings
            print('corner plot problems - might not plot? but also might plot?')

        # Calculate train/val loss over epoch
        toc = time.perf_counter()
        print(f"Evaluation done in {toc - tic:0.4f} seconds")
        l_val = np.mean(np.array(loss_val))

        print('\nValidation Loss: ', l_val)

        l_training = np.mean(np.array(loss_training))
        print('Training Loss: ', l_training)

        torch.save(encoder.state_dict(), '%s/encoder_%s_last.pth'%(outdir,modelName))
        if l_val < l_val_best:
            print("new best model")
            l_val_best = l_val
            torch.save(encoder.state_dict(), '%s/encoder_%s_best.pth'%(outdir,modelName))

        loss_vals_training.append(l_training)
        loss_vals_validation.append(l_val)

        epoch_idx += 1
        # Early stopping
        if m > 8 and all(loss_vals_validation[max(0, m - 8):m] > min(np.append(loss_vals_validation[0:max(0, m - 8)], 200))):
            print('Early Stopping...')
            print(loss_vals_training, '\n', np.diff(loss_vals_training))
            break
    plt.clf()
    fig,ax = plt.subplots()
    ax.plot(range(1,epoch_idx+1),loss_vals_training,label="Training")
    ax.plot(range(1,epoch_idx+1),loss_vals_validation,label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Encoder Loss")
    ax.legend(loc="upper right")
    ax.text(0.45,1.05,loss_text, transform=ax.transAxes, fontsize=16)
    plt.savefig(outdir+"/encoder_loss.png")
    plt.savefig(outdir+"/encoder_loss.pdf")

    print(loss_vals_training, '\n', np.diff(loss_vals_training))
    
    print('DONE with ENCODER training')
    return encoder

def train_classifier(classifier, encoder, batchSize, n_Dim, CorrDim, n_epochs, modelName, outdir, 
                    particleTrainingData, particleValidationData, trainingLabels, jetMassTrainingData, jetMassValidationData):   
    
    # Separating signal and bkg arrays
    particleTrainingDataSig = particleTrainingData[trainingLabels[:,0].astype(bool)]
    particleTrainingDataBkg = particleTrainingData[trainingLabels[:,1].astype(bool)]
    particleValidationDataSig = particleValidationData[validationLabels[:,0].astype(bool)]
    particleValidationDataBkg = particleValidationData[validationLabels[:,1].astype(bool)]
    particleTrainingLabelSig = trainingLabels[trainingLabels[:,0].astype(bool)]
    particleTrainingLabelBkg = trainingLabels[trainingLabels[:,1].astype(bool)]

    # Jet mass for correlation
    jetMassTrainingDataSig = jetMassTrainingData[trainingLabels[:,0].astype(bool)]
    jetMassTrainingDataBkg = jetMassTrainingData[trainingLabels[:,1].astype(bool)]
    jetMassValidationDataSig = jetMassValidationData[validationLabels[:,0].astype(bool)]
    jetMassValidationDataBkg = jetMassValidationData[validationLabels[:,1].astype(bool)]
    
    
    loss = nn.BCELoss(reduction='mean')  
    optimizer = optim.Adam(classifier.parameters(), lr = 0.001)

    loss_vals_training = [] #np.zeros(n_epochs)
    loss_vals_validation = [] # np.zeros(n_epochs)
    acc_vals_training = np.zeros(n_epochs)
    acc_vals_validation = np.zeros(n_epochs)
    
    final_epoch = 0
    l_val_best = 99999
    epoch_idx = 0 
    for m in range(n_epochs):
        print("Epoch %s\n" % m)
        tic = time.perf_counter()
        final_epoch = m
        lst = []
        loss_val = []
        loss_training = []
        correct = []
        tic = time.perf_counter()

        totaldiv2 = min(len(particleTrainingDataSig), len(particleTrainingDataBkg))
        for i in tqdm(range(int(totaldiv2/batchSize))): 
            
            optimizer.zero_grad()

            ######### train classifier #########
            trainingv = torch.FloatTensor(particleTrainingData[i*batchSize:(i+1)*batchSize]).cuda()
            targetv = torch.FloatTensor(trainingLabels[i*batchSize:(i+1)*batchSize]).cuda()
            #print(trainingv,encoder(trainingv))
            #print(encoder(trainingv)[:, 1].reshape(batchSize,1))
            outC = classifier(encoder(trainingv)[:, CorrDim:].reshape(batchSize,n_Dim-CorrDim))
            l = loss(outC, targetv)

            loss_training.append(l.item())
            l.backward()
            optimizer.step()

            loss_string = "Loss: %s" % "{0:.5f}".format(l.item())
            del trainingv, targetv
            torch.cuda.empty_cache()

        toc = time.perf_counter()
        print(f"Training done in {toc - tic:0.4f} seconds")
        tic = time.perf_counter()

        totaldiv2 = min(len(particleValidationDataSig), len(particleValidationDataBkg))
        for i in range(int(totaldiv2/batchSize)): 
            torch.cuda.empty_cache()

            # Classifier 
            targetv_val = torch.FloatTensor(validationLabels[i*batchSize:(i+1)*batchSize]).cuda()
            trainingv_val = torch.FloatTensor(particleValidationData[i*batchSize:(i+1)*batchSize]).cuda()
            #out = classifier(encoder(trainingv_val)[:, 1])
            out = classifier(encoder(trainingv_val)[:, CorrDim:].reshape(batchSize,n_Dim-CorrDim))

            l_val = loss(out, targetv_val)
            lst.append(out.cpu().data.numpy())
            loss_val.append(l_val.item())
            correct.append(targetv_val.cpu())

            del trainingv_val, targetv_val
            torch.cuda.empty_cache()


        toc = time.perf_counter()
        print(f"Evaluation done in {toc - tic:0.4f} seconds")
        l_val = np.mean(np.array(loss_val))

        print('\nValidation Loss: ', l_val)

        l_training = np.mean(np.array(loss_training))
        print('Training Loss: ', l_training)
        
        predicted = np.concatenate(lst)
        val_targetv = np.concatenate(correct)
        acc_vals_validation[m] = accuracy_score(val_targetv[:,0],predicted[:,0]>0.5)
        print("Validation Accuracy: ", acc_vals_validation[m])
        
        torch.save(classifier.state_dict(), '%s/classifier_%s_last.pth'%(outdir,modelName))
        if l_val < l_val_best:
            print("new best model")
            l_val_best = l_val
            torch.save(classifier.state_dict(), '%s/classifier_%s_best.pth'%(outdir,modelName))
        loss_vals_training.append(l_training)
        loss_vals_validation.append(l_val)
        epoch_idx += 1
        if m > 25 and all(loss_vals_validation[max(0, m - 8):m] > min(np.append(loss_vals_validation[0:max(0, m - 8)], 200))):
            print('Early Stopping...')
            print(loss_vals_training, '\n', np.diff(loss_vals_training))
            break
    plt.clf()
    fig,ax = plt.subplots()
    ax.plot(range(1,epoch_idx+1),loss_vals_training,label="Training")
    ax.plot(range(1,epoch_idx+1),loss_vals_validation,label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Classifier Loss")
    ax.legend(loc="upper right")
    ax.text(0.45,1.05,loss_text, transform=ax.transAxes, fontsize=16)
    plt.savefig(outdir+"/classifier_loss.png")
    plt.savefig(outdir+"/classifier_loss.pdf")

    print(loss_vals_training, '\n', np.diff(loss_vals_training))
    print('DONE with CLASSIFIER training')
    
    return classifier

import matplotlib.ticker as plticker
def eval_classifier(classifier, encoder, loss_params_text, outdir):  
    testv = torch.FloatTensor(particleTestData).cuda()

    #enc = 
    predictions = classifier((encoder(testv)         [:, CorrDim:].reshape(-1,n_Dim-CorrDim))).cpu().detach().numpy()
    testData = singletonData[trainingDataLength + validationDataLength: ,]
 
    fig,ax = plt.subplots()
    ax.hist(predictions[testLabels[:,0]==1][:,0], alpha=0.7, bins=20, label="Z'", density=True)
    ax.hist(predictions[testLabels[:,1]==1][:,0], alpha=0.7, bins=20, label="QCD", density=True)
    ax.set_xlabel("Network Output",ha='right', x=1.0, fontsize=16)
    ax.set_ylabel(r'Normalized scale ({})'.format('QCD'), ha='right', y=1.0, fontsize=16)
    ax.legend()
    ax.text(0.45,1.05,loss_text, transform=ax.transAxes, fontsize=16)
    plt.savefig('%s/model_output.png'%(outdir))
    plt.savefig('%s/model_output.pdf'%(outdir))

    fpr, tpr, threshold = roc_curve(np.array(testLabels)[:,1].reshape(-1), np.array(predictions)[:,1].reshape(-1))
    np.savez(outdir+"/rocvals",fpr=fpr,tpr=tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2.5, label="{}, AUC = {:.1f} %".format('ZprimeAtoqq IN',auc(fpr,tpr)*100))
    #plt.title('ROC Curve')
    plt.xlabel('FPR',ha='right', x=1.0, fontsize=24)
    ax.set_ylabel(r'Normalized scale ({})'.format('QCD'), ha='right', y=1.0, fontsize=24)
    plt.legend()
    plt.text(0.45,1.05,loss_text, transform=ax.transAxes, fontsize=16)
    plt.savefig('%s/%s_model_ROC.png'%(outdir,modelName))
    plt.savefig('%s/%s_model_ROC.pdf'%(outdir,modelName))

    sculpt_vars = ['jet_sdmass', 'jet_pT', 'jet_eta', 'jet_phi']  
    #sculpt_vars = ['jet_eta', "jet_phi","jet_EhadOverEem","jet_mass", 'jet_pT', 'jet_sdmass']
    for i in range(len(sculpt_vars)):
        
        # Calculate sculpt_var distribution after cuts
        hist, edges = np.histogram(predictions[testLabels[:,1] == 1][:,0], bins=np.linspace(0.,1.,100),density=True)
        #hist, edges = np.histogram(outputs[y_torch[:,1].cpu().detach().numpy()==1][:,1].cpu().detach().numpy(), bins=np.linspace(0.,1.,100),density=True)
        cdf = np.cumsum(hist)*(edges[1]-edges[0])
        
        pctls = [0.,0.25,0.5,0.7,0.9,0.95,0.99]
        cuts = np.searchsorted(cdf,pctls)
        fig, ax = plt.subplots(figsize=(10,10))

        m_torch = testData[:, i]
        qcd_idxs = testLabels[:,1].astype(bool)

        qcd_inclusive, _ = np.histogram(m_torch[(qcd_idxs)], density=True)
        for c,p in zip(cuts,pctls):
            passing_idxs = predictions[:,0] > edges[c]
            hist, bin_edges = np.histogram(
                m_torch[(qcd_idxs&passing_idxs)], 
            )
            N_passing = float(np.sum(hist))
            qcd_passing = np.divide(hist,[N_passing])
            jsd = scipy.spatial.distance.jensenshannon(qcd_passing, qcd_inclusive)

            bins_centers = 0.5*(bin_edges[1:]+bin_edges[:-1])
            ax.plot(
                bins_centers, 
                qcd_passing,
                label = f"{(1-p)*100:.0f}% ({int(N_passing)}) JSD={0 if jsd is np.nan else jsd:.2f}"
            )

        
        if sculpt_vars[i] == 'jet_sdmass':
            ax.set_xlabel(r'$\mathrm{Jet\ m_{SD}\ (GeV)}$', ha='right', x=1.0, fontsize=24)
        elif sculpt_vars[i] == 'jet_pT':
            ax.set_xlabel(r'$\mathrm{Jet\ pT\ (GeV)}$', ha='right', x=1.0, fontsize=24)
        elif sculpt_vars[i] == 'jet_eta':
            ax.set_xlabel(r'$\mathrm{Jet\ \eta\ (GeV)}$', ha='right', x=1.0, fontsize=24)
        elif sculpt_vars[i] == 'jet_phi':
            ax.set_xlabel(r'$\mathrm{Jet\ \phi\ (GeV)}$', ha='right', x=1.0, fontsize=24)
        else: 
            ax.set_xlabel(sculpt_vars[i], ha='right', x=1.0, fontsize=16)
        ax.set_ylabel(r'Normalized scale ({})'.format('QCD'), ha='right', y=1.0, fontsize=24)
        import matplotlib.ticker as plticker
        #ax.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
        #ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=10))
        #ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
        
        pt_range = [200., 1500.]
        mass_range = [40., 350.]
        
        if sculpt_vars[i] == 'jet_sdmass':
            ax.set_xlim(mass_range[0], mass_range[1])
            ax.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
            ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=10))
            ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
        elif sculpt_vars[i] == 'jet_pT':
            ax.set_xlim(pt_range[0], pt_range[1])
            ax.xaxis.set_major_locator(plticker.MultipleLocator(base=200))
            ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=50))
            ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
        else: 
            ax.xaxis.set_major_locator(plticker.MultipleLocator(base=2))
            ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))
            ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
        
        ax.set_ylim(0, 0.30)
        ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)#, labelleft=False )
        ax.tick_params(direction='in', axis='both', which='minor' , length=6)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')    
        #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax.grid(which='major', alpha=0.9, linestyle='dotted')
        plt.legend(loc="best", fontsize=13)
        
        
        leg = ax.text(0.03, 0.88, ""+str(int(round((min(pt_range)))))+" $\mathrm{<\ Jet\ p_T\ <}$ "+str(int(round((max(pt_range)))))+" GeV" \
              + "\n "+str(int(round((min(mass_range)))))+" $\mathrm{<\ Jet\ m_{SD}\ <}$ "+str(int(round((max(mass_range)))))+" GeV"
                      + "\n Sculpted Sample"
                  , fontsize=16, transform=ax.transAxes) #borderpad=1, frameon=False, loc='upper left', fontsize=16,          )
        #leg._legend_box.align = "left"
        
        #ax.set_xlabel(sculpt_vars[i])
        #ax.set_ylabel("a.u.")
        #ax.text(0.05,1.03,"QCD jets", transform=ax.transAxes)
        ax.text(0.6,1.03,loss_text, transform=ax.transAxes, fontsize=24)
        plt.savefig(outdir+"/sculptingQCD_%s.png"%(sculpt_vars[i]))
        plt.savefig(outdir+"/sculptingQCD_%s.pdf"%(sculpt_vars[i]))
        #plt.show()


# In[ ]:



fig, ax = plt.subplots()
ax.hist(jetMassTrainingDataSig, alpha=0.7, bins=np.linspace(40,350,30),label="Z'")
ax.hist(jetMassTrainingDataBkg, alpha=0.7, bins=np.linspace(40,350,30),label="QCD")
#plt.hist(jetMassValidationDataSig, density=True, alpha=0.5)
#plt.hist(jetMassValidationDataBkg, density=True, alpha=0.5)
ax.legend(loc="best")
ax.set_xlabel("$\mathrm{Jet\ m_{SD}}$" , ha='right', x=1.0, fontsize=16)
ax.set_ylabel(r'Normalized scale ({})'.format('QCD'), ha='right', y=1.0, fontsize=16)

import matplotlib.ticker as plticker
ax.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=10))
ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))

plt.savefig(outdir+"/jet_msd.png")
plt.savefig(outdir+"/jet_msd.pdf")

#$\mathrm{Jet\ m_{SD}\ (GeV)}$

loss_text = 'lambda_cor=%s, lambdacorr1=%s, lambdacorr2=%s'%(weightcov, weightCorr1, weightCorr2)
loss_text = '$\lambda_{MSE}$=%.1f, $\lambda_{STD}$=%.1f, $\lambda_{cov}=$%s, $\lambda_{corr}=$%s'%(weightrepr, weightstd, weightcov, weightCorr1)#, weightCorr2)
#loss_text = "Contrastive Training "
#encoder = GraphNetnoSV(particlesPostCut, n_Dim, entriesPerParticle, 15,
#                      De=5,
#                       Do=6, softmax=False)

encoder = DNN(n_Dim, particlesPostCut*entriesPerParticle)


classifier = Linear(n_Dim-CorrDim,2)
#def encoder(x):
#   return x
encoder = train_encoder(encoder, batchSize, n_Dim, CorrDim, n_epochs, modelName, outdir, 
            particleTrainingData, particleValidationData, trainingLabels, jetMassTrainingData, jetMassValidationData,
            weightrepr, weightcov, weightstd, weightCorr1, weightCorr2)

classifier = train_classifier(classifier, encoder, batchSize, n_Dim, CorrDim, n_epochs, modelName, outdir, 
            particleTrainingData, particleValidationData, trainingLabels, jetMassTrainingData, jetMassValidationData)

eval_classifier(classifier, encoder, loss_text, outdir)




# In[ ]:


#encoder = 1
#classifier = MLP(200, n_Dim)

#classifier = train_classifier(classifier, encoder, batchSize, n_Dim, CorrDim, n_epochs, modelName, outdir, 
#            particleTrainingData, particleValidationData, trainingLabels, jetMassTrainingData, jetMassValidationData)

#eval_classifier(classifier, encoder, loss_text)


# In[ ]:


#eval_classifier(classifier, encoder, loss_text)

