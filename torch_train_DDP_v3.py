# Imports basics
import os
import numpy as np
import h5py
import json
#import setGPU
import sklearn
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

#DPP implementation
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn',force=True)
except RuntimeError:
    pass

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
    ('--lr_policy'), ('--grad_acc', p.INT),
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

p.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
args = p.parse_args()

args.nparts = 100


def main():
#     utils.init_distributed_mode(args)
    print(args)
    
    device = torch.device(args.device)
    
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,world_size=args.world_size, rank=args.rank)
    
    if args.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
     
    assert(args.model)
    if not args.make_PN:
        assert(args.loss)
    
    if not args.is_binary and args.make_N2:
        raise ValueError("need binary for N2 plots")
    if args.opath:
        os.system("mkdir -p ./"+args.opath)
    
    
    #srcDir = '/work/tier3/jkrupa/FlatSamples/data//30Oct22-MiniAODv2/30Oct22/zpr_fj_msd/2017/' 
    #srcDir = '/work/tier3/jkrupa/FlatSamples/data/'
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
    # convert training file
    #data = h5py.File(os.path.join(srcDir, 'tot.h5'),'r')
    
    if False:
        print("Removing NaNs...")
        x = np.where(~np.isfinite(vertexData).all(axis=1))
        y = np.where(~np.isfinite(particleData).all(axis=1))
        z = np.where(~np.isfinite(singletonFeatureData).all(axis=1))
        nan_mask = np.ones(len(particleData),dtype=bool)
        nan_mask[x[0]] = False 
        nan_mask[y[0]] = False 
        nan_mask[z[0]] = False
        print("# events with NaNs (removed): ", np.sum((nan_mask==False).astype(int))) 
        particleData = particleData[nan_mask]
        vertexData = vertexData[nan_mask]
        singletonData = singletonData[nan_mask]
        labels = labels[nan_mask]
        singletonFeatureData = singletonFeatureData[nan_mask]
        
    
    if args.plot_features:
        print("Plotting all features. This might take a few minutes")
        utils.plot_features(singletonData,labels,utils._singleton_labels,args.opath)
        utils.plot_features(vertexData,labels,utils._SV_features_labels,args.opath,"SV")
        utils.plot_features(particleData,labels,utils._p_features_labels,args.opath,"Particle")
        utils.plot_features(singletonFeatureData,labels,utils._singleton_features_labels,args.opath)
    
    
    def run_inference(opath, plot_text, modelName, figpath, model):#, particleDataTest, labelsTest, singletonDataTest, svTestingData=None, eventTestingData=None, pfMaskTestingData=None, svMaskTestingData=None):    
        model.load_state_dict(torch.load(args.mpath))
        eval_classifier(model, plot_text, modelName, figpath, )#particleDataTest, labelsTest, singletonDataTest,svTestingData=vertexDataTest,eventTestingData=eventTestingData,pfMaskTestingData=pfMaskTestingData,svMaskTestingData=svMaskTestingData,)
        return
    
    def train_classifier(classifier, loss, batchSize, nepochs, modelName, outdir,train_loader
        #particleTrainingData, particleValidationData,  trainingLabels, validationLabels,
        #jetMassTrainingData=None, jetMassValidationData=None,
        #encoder=None,n_Dim=None, CorrDim=None, 
        #svTrainingData=None, svValidationData=None,
        #eventTrainingData=None, eventValidationData=None,
        #maskpfTrain=None, maskpfVal=None, masksvTrain=None, masksvVal=None, 
        ):
#         dist.init_process_group("nccl")
#         rank = dist.get_rank()
#         print(f"Start running basic DDP example on rank {rank}.")
#         device_id = rank % torch.cuda.device_count()
#         model = model.to(device_id)
#         model = DDP(model, device_ids=[device_id])
        np.random.seed(epoch)
        random.seed(epoch)
        
        optimizer = optim.Adam(classifier.parameters(), lr = 0.001)
    
        if args.distributed:
            train_sampler.set_epoch(epoch)
            
        train_loader = train_loader.cuda()
        loss_vals_training = np.zeros(nepochs)
        loss_vals_validation = np.zeros(nepochs)
        acc_vals_training = np.zeros(nepochs)
        acc_vals_validation = np.zeros(nepochs)   
    
        accuracy = Accuracy().to(device_id)
        model_dir = outdir
        os.system("mkdir -p ./"+model_dir)
        n_massbins = 20
    
        if args.continue_training:
            model.load_state_dict(torch.load(args.mpath))
            start_epoch = args.mpath.split("/")[-1].split("epoch_")[-1].split("_")[0]
            start_epoch = int(start_epoch) + 1
            print(f"Continuing training from epoch {start_epoch}...")
        else:
            start_epoch = 1
        end_epoch = nepochs
        for iepoch in range(start_epoch,end_epoch):
            loss_training, acc_training = [], []
            loss_validation, acc_validation = [], []
            print(f'Training Epoch {iepoch} on {len(train_loader.dataset)} jets')
            model.train(True)
            for istep, (x_pf, x_sv, jet_features, jet_truthlabel) in enumerate(tqdm(train_loader)):
                
                if 'all_vs_QCD' in args.loss:
                    jet_truthlabel = jet_truthlabel[:,:-1]
          
                if (args.test_run and istep>10 ): break
                x_pf = torch.nan_to_num(x_pf,nan=0.,posinf=0.,neginf=0.)
                x_sv = torch.nan_to_num(x_sv,nan=0.,posinf=0.,neginf=0.)
                for param in model.parameters():
                    param.grad = None
                #optimizer.zero_grad()
                if not args.load_gpu:
                    x_pf = x_pf.to(device_id)
                    x_sv = x_sv.to(device_id)
                    jet_features = jet_features.to(device_id)
                    jet_truthlabel = jet_truthlabel.to(device_id)
                if args.sv:
                    output = model(x_pf,x_sv)
                else:
                    output = model(x_pf)
                #sys.exit(1)
                mass = jet_features[:,utils._singleton_labels.index('zpr_fj_msd')]
                if args.loss == 'jsd':
                    l = loss(output, jet_truthlabel, one_hots[istep*batchSize:(istep+1)*batchSize].to(device_id), n_massbins=n_massbins, LAMBDA_ADV=args.LAMBDA_ADV)
                elif 'disco' in args.loss:
                    #print(mass[:10])
                    l = loss(output, jet_truthlabel, mass, LAMBDA_ADV=args.LAMBDA_ADV,)
                else:
                    l = loss(output, jet_truthlabel)
                #print(istep,l.item())
                loss_training.append(l.item())
                acc_training.append(accuracy(output,torch.argmax(jet_truthlabel.squeeze(), dim=1)).cpu().detach().numpy())
    
                l.backward()
                optimizer.step()
    
                torch.cuda.empty_cache()
                #del , output
            print(f'Validating Epoch {iepoch} on {len(val_loader.dataset)} jets')
            model.train(False)
            if args.rank == 0:
                for istep, (x_pf_val, x_sv_val, jet_features_val, jet_truthlabel_val) in enumerate(tqdm(val_loader)):
                    #model.eval()
                    if 'all_vs_QCD' in args.loss:
                        jet_truthlabel_val = jet_truthlabel_val[:,:-1]      
                    if (args.test_run and istep>10 ): break
                    x_pf_val = torch.nan_to_num(x_pf_val,nan=0.,posinf=0.,neginf=0.)
                    x_sv_val = torch.nan_to_num(x_sv_val,nan=0.,posinf=0.,neginf=0.)
                    #x_sv = torch.nan_to_num(x_pf,nan=0.,posinf=0.,neginf=0.)

                    if not args.load_gpu:
                        x_pf_val = x_pf_val.to(device_id)
                        x_sv_val = x_sv_val.to(device_id)
                        jet_features_val = jet_features_val.to(device_id)
                        jet_truthlabel_val = jet_truthlabel_val.to(device_id)
                    if args.sv:
                        output_val = model(x_pf_val,x_sv_val)
                    else: 
                        output_val = model(x_pf_val)
                    mass = jet_features_val[:,utils._singleton_labels.index('zpr_fj_msd')]
                    if 'disco' in args.loss:
                        l_val = loss(output_val, jet_truthlabel_val, mass, LAMBDA_ADV=args.LAMBDA_ADV,)
                    else:
                        l_val = loss(output_val, jet_truthlabel_val)
                    #print(istep,l_val.item())
                    loss_validation.append(l_val.item())
                    acc_validation.append(accuracy(output_val,torch.argmax(jet_truthlabel_val.squeeze(), dim=1)).cpu().detach().numpy())

                    torch.cuda.empty_cache()
                    #break
    
            epoch_val_loss = np.mean(loss_validation)
            epoch_val_acc  = np.mean(acc_validation)
            epoch_train_loss = np.mean(loss_training)
            epoch_train_acc  = np.mean(acc_training)
    
            print("Epoch %i of %i"%(iepoch,nepochs))
            print("\tTraining:\tloss=%.5f, acc=%.4f"%(epoch_train_loss, epoch_train_acc))
            print("\tValidation:\tloss=%.5f, acc=%.4f"%(epoch_val_loss, epoch_val_acc))
    
            loss_vals_validation[iepoch] = epoch_val_loss
            acc_vals_validation[iepoch]  = epoch_val_acc
            loss_vals_training[iepoch] = epoch_train_loss
            acc_vals_training[iepoch]  = epoch_train_acc
    
            torch.save(classifier.state_dict(), 
                "{}/epoch_{}_{}_loss_{}_{}_acc_{}_{}.pth".format(model_dir,iepoch,modelName.replace(' ','_'),round(loss_vals_training[iepoch],4),round(loss_vals_validation[iepoch],4),round(acc_vals_training[iepoch],4),round(acc_vals_validation[iepoch],4))
            )
    
            #del valInputs, valLabels
            torch.cuda.empty_cache()
            #sys.exit(1)
    
            #epoch_patience = 20
            #if iepoch > epoch_patience and all(loss_vals_validation[max(0, iepoch - epoch_patience):iepoch] > min(np.append(loss_vals_validation[0:max(0, iepoch - epoch_patience)], 200))):
            #    print('Early Stopping...')
    
            #    utils.plot_loss(loss_vals_training,loss_vals_validation,model_dir)
            #    break
            #elif iepoch == nepochs-1:
            utils.plot_loss(loss_vals_training,loss_vals_validation,model_dir)
    
        return classifier
    #END OF TRAINING
    
    def eval_classifier(classifier, training_text, modelName, outdir,val_loader
                        #particleTestingData, testingLabels, testingSingletons,
                        #svTestingData=None,eventTestingData=None,encoder=None,pfMaskTestingData=None,svMaskTestingData=None):
        ):
        
        
        val_loader = val_loader.cuda()
        #classifier.eval()  
        with torch.no_grad():
            print("Running predictions on test data")
            predictions = []
            testingLabels = []
            testingSingletons = []
            batch_size = 1000
    
            for istep, (x_pf, x_sv, jet_features, jet_truthlabel) in enumerate(tqdm(val_loader)):
                #model.eval()
                if 'all_vs_QCD' in args.loss:
                    jet_truthlabel = jet_truthlabel[:,:-1]      
                if (args.test_run and istep>10 ): break
                if args.sv:
                    predictions.append(nn.Softmax(dim=1)(classifier(x_pf,x_sv)).cpu().detach().numpy())
                else:
                    predictions.append(nn.Softmax(dim=1)(classifier(x_pf)).cpu().detach().numpy())
                testingLabels.append(jet_truthlabel.cpu().detach().numpy())
                testingSingletons.append(jet_features.cpu().detach().numpy())
                torch.cuda.empty_cache()
                #break
        predictions = [item for sublist in predictions for item in sublist]
        testingLabels = [item for sublist in testingLabels for item in sublist]
        testingSingletons = [item for sublist in testingSingletons for item in sublist]
        predictions = np.array(predictions)#.astype(np.float32)
        testingLabels = np.array(testingLabels)
        testingSingletons = np.array(testingSingletons)
    
        os.system("mkdir -p "+outdir)
        np.savez(outdir+"/predictions.npy", predictions=predictions, labels=testingLabels, singletons=testingSingletons)
        if 'all_vs_QCD' in args.loss:
            qcd_idxs = np.where(testingLabels.sum(axis=1)==0,True,False)
        else:
            qcd_idxs = testingLabels[:,-1].astype(bool)
            utils.plot_correlation(predictions[qcd_idxs,-1],testingSingletons[qcd_idxs,utils._singleton_labels.index("zpr_fj_msd")], "QCD output score","QCD jet $m_{SD}$ (GeV)", np.linspace(0,1,50),np.linspace(40,350,40),outdir, "qcd_vs_mass")
            utils.sculpting_curves(predictions[qcd_idxs,-1], testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="QCD", inverted=False)
    
        utils.plot_roc_curve(testingLabels, predictions, training_text, outdir, modelName, all_vs_QCD="all_vs_QCD" in args.loss, QCD_only=False)
        utils.plot_features(testingSingletons,testingLabels,utils._singleton_labels,outdir)
    
        if args.is_binary:
            
            prob_2prong = predictions[qcd_idxs,0]
            utils.sculpting_curves(prob_2prong, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="Z\'",inverted=True)
    
        else:
            utils.plot_correlation(predictions[qcd_idxs,0],testingSingletons[qcd_idxs,0], "bb vs QCD output score","QCD jet $m_{SD}$ (GeV)", np.linspace(0,1,100),np.linspace(40,350,40),outdir, "bb_vs_mass")
            utils.plot_correlation(predictions[qcd_idxs,1],testingSingletons[qcd_idxs,0], "cc vs QCD output score","QCD jet $m_{SD}$ (GeV)", np.linspace(0,1,100),np.linspace(40,350,40),outdir, "cc_vs_mass")
            utils.plot_correlation(predictions[qcd_idxs,2],testingSingletons[qcd_idxs,0], "qq vs QCD output score","QCD jet $m_{SD}$ (GeV)", np.linspace(0,1,100),np.linspace(40,350,40),outdir, "qq_vs_mass")
            prob_bb = predictions[qcd_idxs,0]
            utils.sculpting_curves(prob_bb, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="bb",inverted=True)
            prob_cc = predictions[qcd_idxs,1]
            utils.sculpting_curves(prob_cc, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="cc",inverted=True)
            prob_qq = predictions[qcd_idxs,2]
            utils.sculpting_curves(prob_qq, testingSingletons[qcd_idxs,:], training_text, outdir, modelName, score="qq",inverted=True)
    
        predictionsPN = testingSingletons[:,[utils._singleton_labels.index("zpr_fj_particleNetMD_Xbb"), utils._singleton_labels.index("zpr_fj_particleNetMD_Xcc"), utils._singleton_labels.index("zpr_fj_particleNetMD_Xqq"), utils._singleton_labels.index("zpr_fj_particleNetMD_QCD")]]
        utils.plot_roc_curve(testingLabels, predictionsPN, args.plot_text, outdir, "particleNet-MD", all_vs_QCD=False,QCD_only=False)
        
        qcd_idxs = testingLabels[:,-1].astype(bool)
        prob_bb = predictionsPN[qcd_idxs,0]
        utils.sculpting_curves(prob_bb, testingSingletons[qcd_idxs,:],"ParticleNet-MD:bb score", outdir, "particleNet-MD-bb",inverted=True)
        prob_cc = predictionsPN[qcd_idxs,1]
        utils.sculpting_curves(prob_cc, testingSingletons[qcd_idxs,:], "ParticleNet-MD:cc score", outdir, "particleNet-MD-cc",inverted=True)
        prob_qq = predictionsPN[qcd_idxs,2]
        utils.sculpting_curves(prob_qq, testingSingletons[qcd_idxs,:], "ParticleNet-MD:qq score", outdir, "particleNet-MD-qq",inverted=True)
        prob_QCD = predictionsPN[qcd_idxs,3]
        utils.sculpting_curves(prob_QCD, testingSingletons[qcd_idxs,:], "ParticleNet-MD:QCD score", outdir, "particleNet-MD-QCD",inverted=False)
        testingLabelsPN = np.concatenate((np.expand_dims(np.sum(testingLabels[:,:-1],axis=1),-1),np.expand_dims(testingLabels[:,-1],-1)),axis=1)
        predictionsPN = np.concatenate((np.expand_dims(np.sum(predictionsPN[:,:-1],axis=1),-1),np.expand_dims(predictionsPN[:,-1],-1)),axis=1)
        utils.plot_roc_curve(testingLabelsPN, predictionsPN, args.plot_text, outdir, "particleNet-MD-2prong", all_vs_QCD=False, QCD_only=False)
        sys.exit(1)
        predictionsN2 = testingSingletons[:,[utils._singleton_labels.index("zpr_fj_n2b1")]]
        predictionsN2 = (predictionsN2 - np.min(predictionsN2)) / ( np.max(predictionsN2) - np.min(predictionsN2))
        predictionsN2 = np.concatenate((1-predictionsN2,predictionsN2),axis=1)
        utils.plot_roc_curve(labels, predictionsN2, args.plot_text, outdir, "N2", all_vs_QCD=False,qcd_only=False)
        qcd_idxs = labels[:,-1].astype(bool)
        prob_QCD = predictionsN2[qcd_idxs,0]
        utils.sculpting_curves(prob_QCD, testingSingletons[qcd_idxs,:],"N2:QCD score", outdir, "N2",inverted=True)
        prob_N2 = predictionsN2[qcd_idxs,1]
        utils.sculpting_curves(prob_N2, testingSingletons[qcd_idxs,:],"N2:2prong score", outdir, "N2",inverted=False)
        
        sys.exit(1)
        
        #END OF EVAL
        
        
    maskpfTrain = None
    maskpfVal = None
    maskpfTest = None
    masksvTrain = None
    masksvVal = None
    masksvTest = None
    
    if args.model =='DNN':
        model = models.DNN("DNN",particleDataTrain.shape[1]*particleDataTrain.shape[2],labelsTrain.shape[1]).to(device_id)
    
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
    
        model = models.Transformer(args,"transformer",_softmax,_sigmoid, args.sv)
    else:
        raise ValueError("Don't understand model ", args.model) 
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
    if args.sv:
        summary(model,[(100,13),(5,16)])
    else:
        summary(model,(100,13))
    #print(model)
#     if args.resume:
#         pass
    
      
    outdir = f"./{args.opath}/{model.module.name.replace(' ','_')}"
    outdir = utils.makedir(outdir,args.continue_training)
    
    #GET DATA
    from torch.utils.data import DataLoader
    if args.load_gpu:
        from dataset_loader_gpu import zpr_loader
        if not args.mpath or (args.mpath and args.continue_training):
            data_train = zpr_loader(args.ipath,maxfiles=args.num_max_files) 
            train_sampler = torch.utils.data.distributed.DistributedSampler(data_train, shuffle=True)

        data_val = zpr_loader(args.vpath,maxfiles=args.num_max_files)
        val_sampler = None
        if not args.mpath or (args.mpath and args.continue_training):
            train_loader = DataLoader(data_train, batch_size=args.batchsize,shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
            
        val_loader = DataLoader(data_val, batch_size=args.batchsize,shuffle=True,)
    else:
        from dataset_loader import zpr_loader
        if not args.mpath or (args.mpath and args.continue_training):
            data_train = zpr_loader(args.ipath,maxfiles=args.num_max_files) 
            train_sampler = torch.utils.data.distributed.DistributedSampler(data_train, shuffle=True)
        data_val = zpr_loader(args.vpath,maxfiles=args.num_max_files)
        val_sampler = None
        if not args.mpath or (args.mpath and args.continue_training):
            
            train_loader = DataLoader(data_train, batch_size=args.batchsize,shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
            
        val_loader = DataLoader(data_val, batch_size=args.batchsize,shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)
    
    
    #TRAINING NOW
    torch.backends.cudnn.benchmark = True
    if args.mpath:
        if args.continue_training:
            #models = sorted(glob.glob(outdir+"/models/"),key = lambda x:datetime.strptime(x[0], '%d-%m-%Y'))
            #print(models)
            #sys.exit(1)
            model = train_classifier(model, loss, batchsize, n_epochs, model.module.name, "/".join(args.mpath.split("/")[:-1],train_loader),
            )
        else:
            run_inference(args.mpath, args.plot_text, model.module.name, args.mpath+"_plots",val_loader,
                      model, #particleDataTest, labelsTest, singletonDataTest, svTestingData=vertexDataTest, eventTestingData=singletonFeatureDataTest,
                      #pfMaskTestingData=maskpfTest,svMaskTestingData=masksvTest,
            )
    
    else: 
        model = train_classifier(model, loss, batchsize, n_epochs, model.module.name, outdir+"/models/", train_loader
                                 #particleDataTrain, particleDataVal, labelsTrain, labelsVal, jetMassTrainingData=singletonDataTrain[:,0],
                                 #jetMassValidationData=singletonDataVal[:,0],
                                 #svTrainingData=vertexDataTrain, svValidationData=vertexDataVal,
                                 #eventTrainingData=singletonFeatureDataTrain, eventValidationData=singletonFeatureDataVal,
                                 #maskpfTrain=maskpfTrain, maskpfVal=maskpfVal, masksvTrain=masksvTrain, masksvVal=masksvVal,
        )
        eval_classifier(model, args.plot_text, model.module.name, outdir+"/plots/",val_loader )#particleDataTest, labelsTest, singletonDataTest, svTestingData=vertexDataTest, eventTestingData=singletonFeatureDataTest,pfMaskTestingData=maskpfTest,svMaskTestingData=masksvTest,) 
    
    if args.run_captum:
        from captum.attr import IntegratedGradients
        model.eval()
        torch.manual_seed(123)
        np.random.seed(123)
        baseline = torch.zeros(1,particleDataTrain.shape[1],particleDataTrain.shape[2]).to(device_id)
        baselineSV = torch.zeros(1,vertexDataTrain.shape[1],vertexDataTrain.shape[2]).to(device_id)
        baselineE  = torch.zeros(1,singletonFeatureDataTrain.shape[1]).to(device_id)
        inputs = torch.rand(1,particleDataTrain.shape[1],particleDataTrain.shape[2]).to(device_id)
        inputsSV = torch.rand(1,vertexDataTrain.shape[1],vertexDataTrain.shape[2]).to(device_id)
        inputsE = torch.rand(1,singletonFeatureDataTrain.shape[1]).to(device_id)
     
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
    
    
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    main() 
