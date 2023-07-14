from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import torch


'''
gen_matched_data(data, n)
inputs: 
- data, a zpr data loader generated to load jet data
- n, the amount of data you want to use to pretrain. n MUST be <= len(data)

returns:
- torch Dataloader object to load pretraining pairs of jets.
    - 50% of jet pairs will be matched correctly
    - 50% matched incorrectly (uniform incorrect distribution over all possible incorrect labels)

'''
#specify n the number of jet matchings to prepare, must be <= size of dataset
#
# It will make 1/2 correct matchings and 1/2 incorrect (equal probability to be any incorrect combination
#
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import torch


def gen_matched_data(data, n, batch_size, shuffle,sampler = None):
    indices = torch.randperm(len(data))[:n]
    dt = data[indices]
    x_pf, x_sv, jet_features, jet_truthlabel = dt
    
    mask1 = torch.squeeze(torch.nonzero(torch.all(jet_truthlabel == torch.tensor([1., 0., 0., 0.]), dim=1)))
    mask2 = torch.squeeze(torch.nonzero(torch.all(jet_truthlabel == torch.tensor([0., 1., 0., 0.]), dim=1)))
    mask3 = torch.squeeze(torch.nonzero(torch.all(jet_truthlabel == torch.tensor([0., 0., 1., 0.]), dim=1)))
    mask4 = torch.squeeze(torch.nonzero(torch.all(jet_truthlabel == torch.tensor([0., 0., 0., 1.]), dim=1)))
    masks = [mask1,mask2,mask3,mask4]


    pre_train_jets = []
    truth = []
    pre_train_sv = []
    pre_train_jet_features = []
    for v in range(n):
        index = torch.randint(0, n, (1,)).item()
        cur_label = jet_truthlabel[index]
        cur_jet = x_pf[index]
        cur_sv = x_sv[index]
        if v < n/2:
            #Get the same class of jets
            cur_mask = masks[torch.squeeze(torch.nonzero(cur_label)).item()]
            exclude_index = index
            num_elements = cur_mask.numel() - 1
            random_index = torch.randint(0, num_elements, (1,)).item()
            if random_index >= exclude_index:
                random_index += 1
            random_element = cur_mask[random_index]
            truth.append(0)
        else:
            #Get different classes of jets
            exclude_index = torch.squeeze(torch.nonzero(cur_label)).item()
            num_elements = 3
            random_index = torch.randint(0, num_elements, (1,)).item()
            if random_index >= exclude_index:
                random_index += 1
            cur_mask = masks[random_index]
            rand_mask_index= torch.randint(0, cur_mask.numel(), (1,)).item()
            random_element = cur_mask[rand_mask_index]
            truth.append(1)
        matched_jets = [x_pf[index], x_pf[random_element]]
        matched_sv = [x_sv[index], x_sv[random_element]]
        matched_jet_features = [jet_features[index], jet_features[random_element]]
        pre_train_jets.append(matched_jets)
        pre_train_sv.append(matched_sv)
        pre_train_jet_features.append(matched_jet_features)
    # Assuming you have a list of 5000 pairs of tensors
    pairs = pre_train_jets
    # Combine the tensors into a single tensor
    combined_matched_jets = torch.stack([torch.stack(pair, dim=0) for pair in pairs], dim=0)
    
    
    pairs = pre_train_sv
    # Combine the tensors into a single tensor
    combined_matched_sv = torch.stack([torch.stack(pair, dim=0) for pair in pairs], dim=0)
    
    pairs = pre_train_jet_features
    # Combine the tensors into a single tensor
    combined_matched_jet_features = torch.stack([torch.stack(pair, dim=0) for pair in pairs], dim=0)
    
    truth = torch.tensor(truth)
    permutation = torch.randperm(n)

    combined_matched_jets = combined_matched_jets[permutation]
    truth = truth[permutation]
    truth = torch.unsqueeze(truth, dim = 1).float()
    combined_matched_sv = combined_matched_sv[permutation]
    if sampler == None:
            loader = DataLoader(TensorDataset(combined_matched_jets,combined_matched_sv,combined_matched_jet_features,truth),batch_size=batch_size, shuffle=shuffle)
    else: 
        loader = DataLoader(TensorDataset(combined_matched_jets,combined_matched_sv,combined_matched_jet_features,truth),batch_size=batch_size, shuffle=shuffle,sampler = sampler)
    
    return loader

def gen_matched_simple_data(data, n, batch_size, shuffle,sampler = None):
    indices = torch.randperm(len(data))[:n]
    dt = data[indices]
    x_pf, x_sv, jet_features, jet_truthlabel = dt
    jet_truthlabel = torch.hstack((torch.unsqueeze(jet_truthlabel[:,0:3].sum(dim = 1),dim=1), torch.unsqueeze(jet_truthlabel[:,3],dim=1)))
    mask1 = torch.squeeze(torch.nonzero(torch.all(jet_truthlabel == torch.tensor([1., 0.]), dim=1)))
    mask2 = torch.squeeze(torch.nonzero(torch.all(jet_truthlabel == torch.tensor([0., 1.]), dim=1)))
    masks = [mask1,mask2]

    pre_train_jets = []
    truth = []
    pre_train_sv = []
    pre_train_jet_features = []
    for v in range(n):
        index = torch.randint(0, n, (1,)).item()
        cur_label = jet_truthlabel[index]
        cur_jet = x_pf[index]
        cur_sv = x_sv[index]
        if v < n/2:
            #Get the same class of jets
            cur_mask = masks[torch.squeeze(torch.nonzero(cur_label)).item()]
            exclude_index = index
            num_elements = cur_mask.numel() - 1
            random_index = torch.randint(0, num_elements, (1,)).item()
            if random_index >= exclude_index:
                random_index += 1
            random_element = cur_mask[random_index]
            truth.append(0)
        else:
            #Get different classes of jets
            exclude_index = torch.squeeze(torch.nonzero(cur_label)).item()
            cur_mask = masks[(1+exclude_index)%2]
            rand_mask_index= torch.randint(0, cur_mask.numel(), (1,)).item()
            random_element = cur_mask[rand_mask_index]
            truth.append(1)
        matched_jets = [x_pf[index], x_pf[random_element]]
        matched_sv = [x_sv[index], x_sv[random_element]]
        matched_jet_features = [jet_features[index], jet_features[random_element]]
        pre_train_jets.append(matched_jets)
        pre_train_sv.append(matched_sv)
        pre_train_jet_features.append(matched_jet_features)
    # Assuming you have a list of 5000 pairs of tensors
    pairs = pre_train_jets
    # Combine the tensors into a single tensor
    combined_matched_jets = torch.stack([torch.stack(pair, dim=0) for pair in pairs], dim=0)


    pairs = pre_train_sv
    # Combine the tensors into a single tensor
    combined_matched_sv = torch.stack([torch.stack(pair, dim=0) for pair in pairs], dim=0)

    pairs = pre_train_jet_features
    # Combine the tensors into a single tensor
    combined_matched_jet_features = torch.stack([torch.stack(pair, dim=0) for pair in pairs], dim=0)
    truth = torch.tensor(truth)
    truth = torch.unsqueeze(truth, dim = 1).float()
    permutation = torch.randperm(n)

    combined_matched_jets = combined_matched_jets[permutation]
    truth = truth[permutation]
    combined_matched_sv = combined_matched_sv[permutation]

    if sampler == None:
            loader = DataLoader(TensorDataset(combined_matched_jets,combined_matched_sv,combined_matched_jet_features,truth),batch_size=batch_size, shuffle=shuffle)
    else: 
        loader = DataLoader(TensorDataset(combined_matched_jets,combined_matched_sv,combined_matched_jet_features,truth),batch_size=batch_size, shuffle=shuffle,sampler = sampler)
    return loader