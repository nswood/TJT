import numpy as np
import h5py
import glob
import tqdm
import torch
import random
from torch.utils.data.dataset import Dataset  # For custom datasets


class zpr_loader(Dataset):
    def __init__(self, raw_paths, small_feature = False, pf_size = 13,sv_size= 16,  qcd_only=True, transform=None,maxfiles=None,small_QCD= False):
        #super(zpr_loader, self).__init__(raw_paths)
        self.raw_paths = sorted(glob.glob(raw_paths+'*h5'))[:maxfiles]
       
        self.small_feature =small_feature
        self.pf_size = pf_size
        self.sv_size = sv_size
        self.small_QCD = small_QCD
        self.fill_data()
        
        
    def calculate_offsets(self):
        for path in self.raw_paths:
            
            with h5py.File(path, 'r') as f:
                self.strides.append(f['features'].shape[0])
        self.strides = np.cumsum(self.strides)
        
    def fill_data(self):
        
        self.data_features = []
        self.data_sv_features = [] 
        self.data_jetfeatures = []
        self.data_truthlabel = [] 
        
        for fi,path in enumerate(tqdm.tqdm(self.raw_paths)):
            with h5py.File(path, 'r') as f:
                
                 tmp_features = f['features'][()].astype(np.float32)
                 tmp_sv_features = f['features_SV'][()].astype(np.float32)
                 tmp_jetfeatures = f['jet_features'][()].astype(np.float32)
                 tmp_truthlabel = f['jet_truthlabel'][()]
                 self.data_features.append(tmp_features)
                 self.data_sv_features.append(tmp_sv_features)
                 self.data_jetfeatures.append(tmp_jetfeatures)
                 self.data_truthlabel.append(tmp_truthlabel)
                 #if fi == 0:
                 #    self.data_features = tmp_features
                 #    self.data_sv_features = tmp_sv_features
                 #    self.data_jetfeatures = tmp_jetfeatures
                 #    self.data_truthlabel = tmp_truthlabel
                 #else:
                 #    self.data_features = np.concatenate((self.data_features,tmp_features))
                 #    self.data_sv_features = np.concatenate((self.data_sv_features,tmp_sv_features))
                 #    self.data_jetfeatures = np.concatenate((self.data_jetfeatures,tmp_jetfeatures))
                 #    self.data_truthlabel = np.concatenate((self.data_truthlabel,tmp_truthlabel))

        self.data_features = [item for sublist in self.data_features for item in sublist]
        self.data_sv_features = [item for sublist in self.data_sv_features for item in sublist]
        self.data_jetfeatures = [item for sublist in self.data_jetfeatures for item in sublist]
        self.data_truthlabel = [item for sublist in self.data_truthlabel for item in sublist]

        self.data_features = np.array(self.data_features)
        self.data_sv_features = np.array(self.data_sv_features)
        self.data_jetfeatures = np.array(self.data_jetfeatures)
        self.data_truthlabel = np.array(self.data_truthlabel)

        print("self.data_features.shape",self.data_features.shape)
     
        self.data_features = torch.FloatTensor(self.data_features)
        self.data_sv_features = torch.FloatTensor(self.data_sv_features)
        self.data_jetfeatures = torch.FloatTensor(self.data_jetfeatures)
        self.data_truthlabel = torch.FloatTensor(self.data_truthlabel)
        
        if self.small_QCD:
            mask = torch.eq(self.data_truthlabel, torch.tensor([0., 0., 0., 1.]))
            rows_equal_to_target = torch.all(mask, dim=1)
            true_indices = torch.nonzero(rows_equal_to_target).squeeze()
            random.shuffle(true_indices.tolist())
            num_switch = len(true_indices) *1// 3
            rows_equal_to_target[true_indices[:num_switch]] = False
            
            
            self.data_truthlabel = self.data_truthlabel[~rows_equal_to_target]
            self.data_features = self.data_features[~rows_equal_to_target]
            self.data_sv_features = self.data_sv_features[~rows_equal_to_target]
            self.data_jetfeatures = self.data_jetfeatures[~rows_equal_to_target]
        
        if self.small_feature:
            self.data_features = self.data_features[:,:,0:self.pf_size]
            self.data_sv_features = self.data_sv_features[:,:,0:self.sv_size]
    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.h5')))
        return raw_files

    @property
    def processed_file_names(self):
        return []

    def __len__(self):
        return self.data_jetfeatures.shape[0]#self.strides[-1]

    def __getitem__(self, idx):
        x_pf = self.data_features[idx,:,:]
        x_sv = self.data_sv_features[idx,:,:]
        x_jet = self.data_jetfeatures[idx,:]
        y = self.data_truthlabel[idx]
        return x_pf, x_sv, x_jet, y
