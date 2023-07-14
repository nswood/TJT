import numpy as np
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DiscoCorr(nn.Module):
    def __init__(self,background_only=False,anti=False,background_label=1,power=2):
        self.backonly = background_only
        self.background_label = background_label
        self.power = power
        self.anti = anti

    def distance_corr(self,var_1,var_2,normedweight,power=1):
        xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
        yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
        amat = (xx-yy).abs()
        del xx,yy

        amatavg = torch.mean(amat*normedweight,dim=1)
        Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))-amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))+torch.mean(amatavg*normedweight)
        del amat

        xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
        yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
        bmat = (xx-yy).abs()
        del xx,yy

        bmatavg = torch.mean(bmat*normedweight,dim=1)
        Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
          -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
          +torch.mean(bmatavg*normedweight)
        del bmat

        ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
        AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
        BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)
        del Bmat, Amat

        if(power==1):
            dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
        elif(power==2):
            dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
        else:
            dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power

        return dCorr

    def __call__(self,pred,x_biased,weights=None):
        xweights = torch.ones_like(pred)
        disco = self.distance_corr(x_biased,pred,normedweight=xweights,power=self.power)
        if self.anti:
            disco = 1-disco
        return disco

def disco_all_vs_QCD(output, target, mass, LAMBDA_ADV=10.):
    disco = DiscoCorr()
    qcd_idxs = torch.where(torch.sum(target,1)==0,True,False)
    mass_loss = 0 
    for iO in range(output.shape[1]):
        mass_loss += disco(output[qcd_idxs,iO],mass[qcd_idxs])
    '''
    if output.shape[1] == 4: 
        mass_loss = LAMBDA_ADV*(disco(output[qcd_idxs,0], mass[qcd_idxs]) + disco(output[qcd_idxs,1], mass[qcd_idxs]) + disco(output[qcd_idxs,2], mass[qcd_idxs]) + disco(output[qcd_idxs,3], mass[qcd_idxs]))
    elif output.shape[1] == 2:
        mass_loss = LAMBDA_ADV*(disco(output[qcd_idxs,0], mass[qcd_idxs]) + disco(output[qcd_idxs,1], mass[qcd_idxs]))
    '''
    return all_vs_QCD(output,target) + LAMBDA_ADV*mass_loss

def disco(output, target, mass, LAMBDA_ADV=10.,):
    disco = DiscoCorr()
    crossentropy = nn.CrossEntropyLoss()
    #crossentropy = nn.BCELoss()
    perf_loss = crossentropy(output,target)
    qcd_idxs = target[:,-1].to(torch.bool)
    mass_loss = 0 
    #print(f"crossentropy: {perf_loss}")
    for iO in range(0,4): #output.shape[1]):
        #print(f"Disco along axis {iO}",disco(output[qcd_idxs,iO],mass[qcd_idxs]))
        #mass_loss += LAMBDA_ADV*disco(output[qcd_idxs,iO],mass[qcd_idxs])
        mass_loss += LAMBDA_ADV*disco(output[:,iO],mass)
        print(output[:,iO])
        print(mass)
        print(mass_loss)
    #    print(f"mass decorr: {mass_loss}")
    #print("perf_loss + mass_loss",perf_loss + mass_loss)
    '''
    if output.shape[1] == 4: 
        mass_loss = LAMBDA_ADV*(disco(output[qcd_idxs,0], mass[qcd_idxs]) + disco(output[qcd_idxs,1], mass[qcd_idxs]) + disco(output[qcd_idxs,2], mass[qcd_idxs]) + disco(output[qcd_idxs,3], mass[qcd_idxs]))
    elif output.shape[1] == 2:
        mass_loss = LAMBDA_ADV*(disco(output[qcd_idxs,0], mass[qcd_idxs]) + disco(output[qcd_idxs,1], mass[qcd_idxs]))
    '''
#     return perf_loss + mass_loss
    
    return mass_loss



def adversarial():
    #https://stackoverflow.com/questions/71049941/create-one-hot-encoding-for-values-of-histogram-bins
    one_hots = nn.functional.one_hot(one_hots, num_classes=-1).to(torch.float32)



    #mass histogram for true b events weighted by b,qcd prob 
    hist_alltag_b = torch.mm(torch.transpose(one_hots,0,1), torch.mm(torch.diag(torch.where(target[:,0]==True,1.0,0.0)),output))
    #mass histogram for true b events weighted by qcd prob
    hist_qcdtag_b = hist_alltag_b[:,-1]/torch.sum(hist_alltag_b[:,-1],axis=0)
    #mass histogram for true b events weighted by b prob
    hist_btag_b   = hist_alltag_b[:,0]/torch.sum(hist_alltag_b[:,0],axis=0)
    #average of true b histogram
    hist_average_b = (hist_btag_b + hist_qcdtag_b)/2.0

    #mass histogram for true qcd events weighted by b,qcd prob 
    hist_alltag_qcd = torch.mm(torch.transpose(one_hots,0,1), torch.mm(torch.diag(torch.where(target[:,-1]==True,1.0,0.0)),output))
    #mass histogram for true qcd events weighted by qcd prob 
    hist_qcdtag_qcd = hist_alltag_qcd[:,-1]/torch.sum(hist_alltag_qcd[:,-1],axis=0)
    #mass histogram for true qcd events weighted by b prob 
    hist_btag_qcd = hist_alltag_qcd[:,0]/torch.sum(hist_alltag_qcd[:,0],axis=0)
    #average of true qcd histogram
    hist_average_qcd = (hist_btag_qcd + hist_qcdtag_qcd)/2.0
    bce_loss = nn.functional.binary_cross_entropy(output,target)
    
    return bce_loss \
         + LAMBDA_ADV*(torch.nn.functional.kl_div(hist_qcdtag_b,hist_average_b) + torch.nn.functional.kl_div(hist_btag_b,hist_average_b))/2.\
         + LAMBDA_ADV*(torch.nn.functional.kl_div(hist_btag_qcd,hist_average_qcd) + torch.nn.functional.kl_div(hist_qcdtag_qcd,hist_average_qcd))/2.\

def all_vs_QCD(output, target):

    qcd_idxs = torch.where(torch.sum(target,1)==0,True,False)
    mask_bb = (target[:,0] == 1) | qcd_idxs
    mask_cc = (target[:,1] == 1) | qcd_idxs
    mask_qq = (target[:,2] == 1) | qcd_idxs
    #print(output[mask_bb].float(), target[mask_bb].float())
    #print(output[mask_cc].float(), target[mask_cc].float())
    #print(output[mask_qq].float(), target[mask_qq].float())

    #sys.exit(1)
    loss = nn.functional.binary_cross_entropy(output[mask_bb].float(), target[mask_bb].float()) + nn.functional.binary_cross_entropy(output[mask_cc].float(), target[mask_cc].float()) + nn.functional.binary_cross_entropy(output[mask_qq].float(), target[mask_qq].float()) 
    return loss

import torch
import torch.nn as nn

import torch
import torch.nn.functional as F

def margin_triplet_loss(embeddings, labels, margin=0.2):
    # Compute pairwise distances between embeddings
    labels = torch.sum(labels[:,0:3], dim =1)
    
    pairwise_distances = torch.cdist(embeddings, embeddings, p=2)
    
    # Initialize variables for storing triplet loss components
    triplet_loss = 0.0
    num_triplets = 0
    
    # Iterate over each embedding and label
    for i in range(len(embeddings)):
        anchor_embedding = embeddings[i]
        anchor_label = labels[i]
        
        # Select positive pairs with the same label as the anchor
        positive_mask = (labels == anchor_label).float()
        positive_distances = pairwise_distances[i] * positive_mask
        
        # Find the hardest positive sample (maximum distance)
        hardest_positive_distance = positive_distances.max()
        
        # Select negative pairs with different labels from the anchor
        negative_mask = (labels != anchor_label).float()
        negative_distances = pairwise_distances[i] * negative_mask
        
        # Find the hardest negative sample (minimum distance)
        hardest_negative_distance = negative_distances.min()
        
        # Compute the triplet loss component for the anchor
        triplet_loss += F.relu(hardest_positive_distance - hardest_negative_distance + margin)
        num_triplets += 1
        
    # Average the triplet loss over the number of triplets
    if num_triplets > 0:
        triplet_loss /= num_triplets
    
    return triplet_loss


def ContrastiveLoss(self, output1, output2, label,margin=1.0):
    euclidean_distance = nn.functional.pairwise_distance(output1, output2)
    loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                   (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive

