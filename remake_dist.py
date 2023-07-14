# Imports basics

import numpy as np
import pandas as pd
import h5py
import json
import uproot
import os,sys

# Defines important variables

particle_num = 50
file_num_sig = 1
file_num_bkg = 1 
fill_factor = 1
pt_range = [400., 1000.]
#pt_range = [40., 250.]
mass_range = [40., 250.]
dR_limit = 0.8
signal_list = ['flat_qq']
background_list = ['QCD_HT700to1000', 'QCD_HT_1000to1500', 'QCD_HT_1500to2000', 'QCD_HT_2000toInf']

output_name = "data/FullQCD_FullSig_Zqq_noFill_dRlimit08_50particlesordered_genMatched50_ECF.h5"
output_name_flatratio = "data/FullQCD_FullSig_Zqq_noFill_dRlimit08_50particlesordered_genMatched50_ECF_flatratio.h5"

# Opens json files for signal and background

with open("pf.json") as jsonfile:
    payload = json.load(jsonfile)
    weight = payload['weight']
    features_track = payload['features_track']
    conversion_track = payload['conversion_track']
    features_tower = payload['features_tower']
    conversion_tower = payload['conversion_tower']
    ss = payload['ss_vars']
    gen = payload['gen_vars']
    N2feat = payload['N2_vars']

# Creates the column names of the final data frame

part_features = []
for iVar in features_track:
    for i0 in range(particle_num):
        part_features.append(iVar + str(i0))

columns = ss + weight + N2feat + ['N2'] + part_features + ['label']

# Unnests a pandas dataframe

def unnest(df, explode):
    """unnest(DataFrame,List)"""
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx

    return df1.join(df.drop(explode, 1), how='left')

def phi_kernel(a): 
    return (a + np.pi) % (2 * np.pi) - np.pi

def deltaphi_kernel(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi

def dR(df): 
    df_dR = np.sqrt(np.add(np.square(np.subtract(df["jet_eta"], df["gen_eta"])), np.square(deltaphi_kernel(df["jet_phi"],df["gen_phi"]))))
    return df_dR

def genMatch(df):
    abs_max = 50 #defines maximum jet-mass - gen-mass
    genMatched_df = df[np.absolute(df["jet_mass"]-df["gen_mass"]) < abs_max]
    
    return genMatched_df
    

# Makes a data set where the distribution of the background across mass and pT is similar to that of the signal
def remake_fillsig(iFiles_sig, iFiles_bkg, iFile_out):
    """remake(list[array(nxm),...], list[array(nxs),...], str)"""

    # Creates the signal data frame
    df_sig_to_concat = []
    for sig in iFiles_sig:
        file_list = os.listdir(payload['samples'][sig])
        file_num_sig_min = min(len(file_list), file_num_sig)
        if len(file_list) < file_num_sig: print('LACKING SIGNAL SAMPLES TO FILL')
        for i in range(file_num_sig_min):
            print(file_list[i])
            print(str(i) + " out of " + str(file_num_sig_min))
            data_set = payload['samples'][sig]+file_list[i]
            arr_sig_to_concat_temp = []
            file1 = uproot.open(data_set)
            tree = file1['tree']
            branches = tree.arrays()
            event_num = len(branches['jet_pt'])
            df_sig_tower = pd.DataFrame({column: list(branches[conversion_tower[column]]) for column in features_tower})
            df_sig_tower = unnest(df_sig_tower, features_tower)
            df_sig_track = pd.DataFrame({column: list(branches[conversion_track[column]]) for column in features_track})
            df_sig_track = unnest(df_sig_track, features_track)
            for event in range(event_num):
                df_sig_temp = pd.concat([df_sig_track.loc[event], df_sig_tower.loc[event]], sort=False).fillna(0)
                df_sig_temp = df_sig_temp.sort_values("pt", ascending=False).head(particle_num)
                arr_sig_temp = df_sig_temp.values.flatten('F')
                arr_sig_to_concat_temp.append(arr_sig_temp)
            arr_sig_temp = np.vstack(arr_sig_to_concat_temp)
            df_sig_temp = pd.DataFrame(arr_sig_temp, columns=part_features)
            for column in ss + weight + gen:
                df_sig_temp[column] = np.array(branches[column]).reshape(-1, 1)
            df_sig_temp['label'] = 1
            df_sig_temp = genMatch(df_sig_temp)
            dR_col = dR(df_sig_temp).values.reshape(-1, 1)
            df_sig_temp = df_sig_temp[columns]
            pt_col = df_sig_temp[weight[0]].values.reshape(-1, 1)
            mass_col = df_sig_temp[weight[1]].values.reshape(-1, 1)
            pt0_col = df_sig_temp['pt0'].values.reshape(-1, 1)
            df_sig_temp = df_sig_temp[np.logical_and(np.logical_and(np.logical_and(np.greater(pt_col, pt_range[0]), 
                                                                                   np.less(pt_col, pt_range[1])), 
                                                                    np.logical_and(np.greater(mass_col, mass_range[0]), 
                                                                                   np.less(mass_col, mass_range[1]))),
                                                                    np.logical_and(np.less(dR_col, dR_limit), 
                                                                                   np.greater(pt0_col, 0)))]
            df_sig_to_concat.append(df_sig_temp)
    df_sig = pd.concat(df_sig_to_concat)

    # Calculates the distribution of the signal

    sig_hist, _x, _y = np.histogram2d(df_sig[weight[0]], df_sig[weight[1]], bins=20,
                                      range=np.array([pt_range, mass_range]))
    
    # Creates the background data frame
    
    df_remade_bkg = pd.DataFrame(columns=columns)
    df_bkg_to_concat = []
    for bkg in iFiles_bkg:
        file_list = os.listdir(payload['samples'][bkg])
        file_num_bkg_min = min(len(file_list), file_num_bkg)
        if len(file_list) < file_num_bkg: print('LACKING BACKGROUND SAMPLES TO FILL')
        for i in range(file_num_bkg_min):
            print(str(i) + " out of " + str(file_num_bkg_min))
            print(file_list[i])
            data_set = payload['samples'][bkg]+file_list[i]
            arr_bkg_to_concat_temp = []
            file1 = uproot.open(data_set)
            tree = file1['tree']
            branches = tree.arrays()
            event_num = len(branches['jet_pt'])
            df_bkg_tower = pd.DataFrame({column: list(branches[conversion_tower[column]]) for column in features_tower})
            df_bkg_tower = unnest(df_bkg_tower, features_tower)
            df_bkg_track = pd.DataFrame({column: list(branches[conversion_track[column]]) for column in features_track})
            df_bkg_track = unnest(df_bkg_track, features_track)
            for event in range(event_num):
                df_bkg_temp = pd.concat([df_bkg_track.loc[event], df_bkg_tower.loc[event]], sort=False).fillna(0)
                df_bkg_temp = df_bkg_temp.sort_values("pt", ascending=False).head(particle_num)
                arr_bkg_temp = df_bkg_temp.values.flatten('F')
                arr_bkg_to_concat_temp.append(arr_bkg_temp)
            arr_bkg_temp = np.vstack(arr_bkg_to_concat_temp)
            df_bkg_temp = pd.DataFrame(arr_bkg_temp, columns=part_features)
            for column in ss + weight:
                df_bkg_temp[column] = np.array(branches[column]).reshape(-1, 1)
            df_bkg_temp['label'] = 0
            df_bkg_temp = df_bkg_temp[columns]
            df_bkg_to_concat.append(df_bkg_temp)
    df_bkg = pd.concat(df_bkg_to_concat)

    # Adds background based on signal distribution until fill factor is reached

    for ix in range(len(_x) - 1):
        for iy in range(len(_y) - 1):
            new_df_bkg = df_bkg[((df_bkg[weight[0]] >= _x[ix]) & (df_bkg[weight[0]] < _x[ix + 1]) & (
                                df_bkg[weight[1]] >= _y[iy]) & (df_bkg[weight[1]] < _y[iy + 1]))]
            df_remade_bkg = pd.concat([df_remade_bkg, new_df_bkg.sample(n=min(int(int(sig_hist[ix, iy]) * fill_factor), len(new_df_bkg)))], ignore_index=True)

    # Shows fill factor per bin

    bkg_hist, _, _ = np.histogram2d(df_remade_bkg[weight[0]], df_remade_bkg[weight[1]], bins=20,
                                    range=np.array([pt_range, mass_range]))
    print(np.nan_to_num(np.divide(bkg_hist, sig_hist)))
    
    # Merges data frames

    merged_df = pd.concat([df_sig, df_remade_bkg]).astype('float32')

    # Creates output file

    merged_df = merged_df[columns]
    final_df = merged_df[~(np.sum(np.isinf(merged_df.values), axis=1) > 0)]
    arr = final_df.values

    # Open HDF5 file and write dataset

    h5_file = h5py.File(iFile_out, 'w')
    h5_file.create_dataset('deepDoubleQ', data=arr, compression='lzf')
    h5_file.close()
    del h5_file

# Makes a data set where the distribution of the signal across mass and pT is similar to that of the background
def remake_fillbkg(iFiles_sig, iFiles_bkg, iFile_out):
    """remake(list[array(nxm),...], list[array(nxs),...], str)"""

    # Creates the background data frame
    
    df_remade_bkg = pd.DataFrame(columns=columns)
    df_bkg_to_concat = []
    for bkg in iFiles_bkg:
        file_list = os.listdir(payload['samples'][bkg])
        file_num_bkg_min = min(len(file_list), file_num_bkg)
        if len(file_list) < file_num_bkg: print('LACKING BACKGROUND SAMPLES TO FILL')
        for i in range(file_num_bkg_min):
            print(str(i) + " out of " + str(file_num_bkg_min))
            print(file_list[i])
            data_set = payload['samples'][bkg]+file_list[i]
            arr_bkg_to_concat_temp = []
            file1 = uproot.open(data_set)
            tree = file1['tree']
            branches = tree.arrays()
            print(len(branches))
            #branches = branches[:1000]
            branches = branches[:int(len(branches))]
            event_num = int(len(branches['jet_pt']))
            df_bkg_tower = pd.DataFrame({column: list(branches[conversion_tower[column]]) for column in features_tower})
            df_bkg_tower = unnest(df_bkg_tower, features_tower)
            df_bkg_track = pd.DataFrame({column: list(branches[conversion_track[column]]) for column in features_track})
            df_bkg_track = unnest(df_bkg_track, features_track)
            for event in range(event_num):
                df_bkg_temp = pd.concat([df_bkg_track.loc[event], df_bkg_tower.loc[event]], sort=False).fillna(0)
                df_bkg_temp = df_bkg_temp.sort_values("pt", ascending=False).head(particle_num)
                arr_bkg_temp = df_bkg_temp.values.flatten('F')
                arr_bkg_to_concat_temp.append(arr_bkg_temp)
            arr_bkg_temp = np.vstack(arr_bkg_to_concat_temp)
            df_bkg_temp = pd.DataFrame(arr_bkg_temp, columns=part_features)
            for column in ss + weight + N2feat:
                df_bkg_temp[column] = np.array(branches[column]).reshape(-1, 1)
            df_bkg_temp['label'] = 0
            df_bkg_temp['N2'] = df_bkg_temp[N2feat[1]]/(df_bkg_temp[N2feat[0]]*df_bkg_temp[N2feat[0]])
            print(df_bkg_temp[N2feat[1]])
            print(df_bkg_temp[N2feat[0]])
            print(df_bkg_temp['N2'])
            df_bkg_temp = df_bkg_temp[columns]
            pt_col = df_bkg_temp[weight[0]].values.reshape(-1, 1)
            mass_col = df_bkg_temp[weight[1]].values.reshape(-1, 1)
            pt0_col = df_bkg_temp['pt0'].values.reshape(-1, 1)
            df_bkg_temp = df_bkg_temp[np.logical_and(np.logical_and(np.logical_and(np.greater(pt_col, pt_range[0]), 
                                                                                   np.less(pt_col, pt_range[1])), 
                                                                    np.logical_and(np.greater(mass_col, mass_range[0]), 
                                                                                   np.less(mass_col, mass_range[1]))),
                                                                                   np.greater(pt0_col, 0))]
            df_bkg_to_concat.append(df_bkg_temp)
    df_bkg = pd.concat(df_bkg_to_concat)
    
    bkg_hist, _x, _y = np.histogram2d(df_bkg[weight[0]], df_bkg[weight[1]], bins=20,
                                      range=np.array([pt_range, mass_range]))
    
    print('background hist:')
    
    # Creates the signal data frame
    df_remade_sig_flatratio = pd.DataFrame(columns=columns)
    df_remade_sig = pd.DataFrame(columns=columns)
    df_sig_to_concat = []
    for sig in iFiles_sig:
        file_list = os.listdir(payload['samples'][sig])
        print(len(file_list))
        file_num_sig_min = min(len(file_list), file_num_sig)
        if len(file_list) < file_num_sig: print('LACKING SIGNAL SAMPLES TO FILL')
        for i in range(file_num_sig_min):
            print(file_list[i])
            print(str(i) + " out of " + str(file_num_sig_min))
            data_set = payload['samples'][sig]+file_list[i]
            arr_sig_to_concat_temp = []
            file1 = uproot.open(data_set)
            tree = file1['tree']
            branches = tree.arrays()
            print(len(branches))
            #branches = branches[:1000]
            branches = branches[:int(len(branches))]
            event_num = int(len(branches['jet_pt']))
            df_sig_tower = pd.DataFrame({column: list(branches[conversion_tower[column]]) for column in features_tower})
            df_sig_tower = unnest(df_sig_tower, features_tower)
            df_sig_track = pd.DataFrame({column: list(branches[conversion_track[column]]) for column in features_track})
            df_sig_track = unnest(df_sig_track, features_track)
            for event in range(event_num):
                df_sig_temp = pd.concat([df_sig_track.loc[event], df_sig_tower.loc[event]], sort=False).fillna(0)
                df_sig_temp = df_sig_temp.sort_values("pt", ascending=False).head(particle_num)
                arr_sig_temp = df_sig_temp.values.flatten('F')
                arr_sig_to_concat_temp.append(arr_sig_temp)
            arr_sig_temp = np.vstack(arr_sig_to_concat_temp)
            df_sig_temp = pd.DataFrame(arr_sig_temp, columns=part_features)
            for column in ss + weight + gen + N2feat:
                df_sig_temp[column] = np.array(branches[column]).reshape(-1, 1)
            df_sig_temp['label'] = 1
            df_sig_temp['N2'] = df_sig_temp[N2feat[1]]/(df_sig_temp[N2feat[0]]*df_sig_temp[N2feat[0]])
            df_sig_temp = genMatch(df_sig_temp)
            dR_col = dR(df_sig_temp).values.reshape(-1, 1)
            df_sig_temp = df_sig_temp[columns]
            pt_col = df_sig_temp[weight[0]].values.reshape(-1, 1)
            mass_col = df_sig_temp[weight[1]].values.reshape(-1, 1)
            pt0_col = df_sig_temp['pt0'].values.reshape(-1, 1)
            df_sig_temp = df_sig_temp[np.logical_and(np.logical_and(np.logical_and(np.greater(pt_col, pt_range[0]), 
                                                                                   np.less(pt_col, pt_range[1])), 
                                                                    np.logical_and(np.greater(mass_col, mass_range[0]), 
                                                                                   np.less(mass_col, mass_range[1]))),
                                                                    np.logical_and(np.less(dR_col, dR_limit), 
                                                                                   np.greater(pt0_col, 0)))]
            df_sig_to_concat.append(df_sig_temp)
    df_sig = pd.concat(df_sig_to_concat)
    

    # Adds signal based on background distribution until fill factor is reached
     
    fill_completely = True
    
    for ix in range(len(_x) - 1):
        for iy in range(len(_y) - 1):
            new_df_sig = df_sig[((df_sig[weight[0]] >= _x[ix]) & (df_sig[weight[0]] < _x[ix + 1]) & (
                                df_sig[weight[1]] >= _y[iy]) & (df_sig[weight[1]] < _y[iy + 1]))]
            
            new_df_bkg = df_bkg[((df_bkg[weight[0]] >= _x[ix]) & (df_bkg[weight[0]] < _x[ix + 1]) & (
                                df_bkg[weight[1]] >= _y[iy]) & (df_bkg[weight[1]] < _y[iy + 1]))]
            
            if True:
                fill_num = len(new_df_bkg)
                if len(new_df_sig) == 0: 
                    fill_num = 0
                df_remade_sig_flatratio = pd.concat([df_remade_sig_flatratio, 
                                           new_df_sig.sample(n=fill_num, replace=True)], 
                                           ignore_index=True)    
            else:    
                df_remade_sig_flatratio = pd.concat([df_remade_sig_flatratio, new_df_sig.sample(n=min(int(int(bkg_hist[ix, iy]) * fill_factor),
                                                                                  len(new_df_sig)))], ignore_index=True)

    
    df_remade_sig = df_sig
    
    # Shows fill factor per bin

    sig_hist, _, _ = np.histogram2d(df_remade_sig[weight[0]], df_remade_sig[weight[1]], bins=20,
                                    range=np.array([pt_range, mass_range]))
    
    print('signal hist:')
    print(sig_hist)
    
    print('ratio:')
    print(np.nan_to_num(np.divide(bkg_hist, sig_hist)))
   
    # Merges data frames

    merged_df = pd.concat([df_bkg, df_remade_sig]).astype('float32')
    merged_df_flatratio = pd.concat([df_bkg, df_remade_sig_flatratio]).astype('float32')
    
    # Creates output file

    merged_df = merged_df[columns]
    merged_df_flatratio = merged_df_flatratio[columns]
    final_df = merged_df[~(np.sum(np.isinf(merged_df.values), axis=1) > 0)]
    final_df_flatratio = merged_df_flatratio[~(np.sum(np.isinf(merged_df_flatratio.values), axis=1) > 0)]
    print(list(final_df.columns))
    arr = final_df.values
    arr_flatratio = final_df_flatratio.values
    print(arr.shape)

    # Open HDF5 file and write dataset

    h5_file = h5py.File(iFile_out, 'w')
    h5_file.create_dataset('deepDoubleQ', data=arr, compression='lzf')
    h5_file.close()
    
    output_name_flatratio = "data/FullQCD_FullSig_Zqq_noFill_dRlimit08_50particlesordered_genMatched50_ECF_flatratio.h5"
    h5_file = h5py.File(output_name_flatratio, 'w')
    h5_file.create_dataset('deepDoubleQ', data=arr_flatratio, compression='lzf')
    h5_file.close()
    del h5_file
    
    
# Remakes data sets
#remake_fillsig(signal_list, background_list, output_name)
remake_fillbkg(signal_list, background_list, output_name)