import numpy as np
import h5py

# Define reweighting procedure

pt_range = [400., 1000.]
mass_range = [40., 250.]

def weights_1dsdmass(data, reweight_to='sig'):
    
    if reweight_to == 'sig': 
        indep = data[data[:,-1]==1]
        reweighted = data[data[:,-1]==0]
    
    elif reweight_to == 'bkg': 
        reweighted = data[data[:,-1]==1]
        indep = data[data[:,-1]==0]
    
    else:
        print('not a valid reweighting')

    nBins = 15
    indep_hist, indep_msd_bins = np.histogram(indep[:,5],bins=nBins,range=(mass_range[0],mass_range[1]),density=True)
    reweighted_hist, reweighted_msd_bins = np.histogram(reweighted[:,5],bins=nBins,range=(mass_range[0],mass_range[1]),density=True)

    weight_surface = np.divide(indep_hist,reweighted_hist)
    
    msd_idxs = np.digitize(reweighted[:,5], bins=np.linspace(mass_range[0], mass_range[1], num=nBins), right=True)
    msd_idxs[msd_idxs>=nBins] = msd_idxs[nBins-1]

    weights = weight_surface[msd_idxs-1]
    weights_reweighted = np.nan_to_num(weights, nan = 1., posinf = 1., neginf = 1.)
    weights_indep = np.ones(len(indep))
    
    weights = np.concatenate((weights_indep, weights_reweighted))
    data = np.concatenate((indep, reweighted))
    
    return(weights, data)

def weights_fillbkg(data):

    sig = data[data[:,-1]==1]
    bkg = data[data[:,-1]==0]

    nBins = 15
    sig_hist, sig_msd_bins, sig_pt_bins = np.histogram2d(sig[:,5],sig[:,4],bins=nBins,range=[(mass_range[0],mass_range[1]), (pt_range[0],pt_range[1])],density=True)
    bkg_hist, bkg_msd_bins, bkg_pt_bins = np.histogram2d(bkg[:,5],bkg[:,4],bins=nBins,range=((mass_range[0],mass_range[1]), (pt_range[0],pt_range[1])),density=True)

    weight_surface = np.divide(bkg_hist,sig_hist)
    
    msd_idxs = np.digitize(sig[:,5], bins=np.linspace(mass_range[0], mass_range[1], num=nBins), right=True)
    pt_idxs = np.digitize(sig[:,4], bins=np.linspace(pt_range[0], pt_range[1], num=nBins), right=True)

    msd_idxs[msd_idxs>=nBins] = msd_idxs[nBins-1]
    pt_idxs[pt_idxs>=nBins] = pt_idxs[nBins-1]

    weights = weight_surface[msd_idxs-1,pt_idxs-1]
    weights_sig = np.nan_to_num(weights, nan = 1., posinf = 1., neginf = 1.)
    weights_bkg = np.ones(len(bkg))
    
    weights = np.concatenate((weights_sig, weights_bkg))
    data = np.concatenate((sig, bkg))
    
    return(weights, data)


def weights_fillsig(data):

    sig = data[data[:,-1]==1]
    bkg = data[data[:,-1]==0]

    nBins = 10
    sig_hist, sig_msd_bins, sig_pt_bins = np.histogram2d(sig[:,5],sig[:,4],bins=nBins,range=[(mass_range[0],mass_range[1]), (pt_range[0],pt_range[1])],density=True)
    bkg_hist, bkg_msd_bins, bkg_pt_bins = np.histogram2d(bkg[:,5],bkg[:,4],bins=nBins,range=((mass_range[0],mass_range[1]), (pt_range[0],pt_range[1])),density=True)
    
    print(sig_hist)
    print(bkg_hist)
 
    weight_surface = np.divide(sig_hist,bkg_hist)
    
    msd_idxs = np.digitize(bkg[:,5], bins=np.linspace(mass_range[0], mass_range[1], num=nBins), right=True)
    pt_idxs = np.digitize(bkg[:,4], bins=np.linspace(pt_range[0], pt_range[1], num=nBins), right=True)

    msd_idxs[msd_idxs>=nBins] = msd_idxs[nBins-1]
    pt_idxs[pt_idxs>=nBins] = pt_idxs[nBins-1]

    weights = weight_surface[msd_idxs-1,pt_idxs-1]
    weights_bkg = np.nan_to_num(weights, nan = 1., posinf = 1., neginf = 1.)/10
    weights_sig = np.ones(len(sig))
    
    weights = np.concatenate((weights_sig, weights_bkg))
    data = np.concatenate((sig, bkg))
    
    return(weights, data)

def weights_fillflat(data):

    sig = data[data[:,-1]==1]
    bkg = data[data[:,-1]==0]

    nBins = 15
    sig_hist, sig_msd_bins, sig_pt_bins = np.histogram2d(sig[:,5],sig[:,4],bins=nBins,range=[(mass_range[0],mass_range[1]), (pt_range[0],pt_range[1])],density=True)
    bkg_hist, bkg_msd_bins, bkg_pt_bins = np.histogram2d(bkg[:,5],bkg[:,4],bins=nBins,range=((mass_range[0],mass_range[1]), (pt_range[0],pt_range[1])),density=True)

    weight_surface = np.divide(1,bkg_hist)
    
    msd_idxs = np.digitize(bkg[:,5], bins=np.linspace(mass_range[0], mass_range[1], num=nBins), right=True)
    pt_idxs = np.digitize(bkg[:,4], bins=np.linspace(pt_range[0], pt_range[1], num=nBins), right=True)

    msd_idxs[msd_idxs>=nBins] = msd_idxs[nBins-1]
    pt_idxs[pt_idxs>=nBins] = pt_idxs[nBins-1]

    weights = weight_surface[msd_idxs-1,pt_idxs-1]
    weights_bkg = np.nan_to_num(weights, nan = 1., posinf = 1., neginf = 1.)
    
    
    weight_surface = np.divide(1,sig_hist)
    
    msd_idxs = np.digitize(sig[:,5], bins=np.linspace(mass_range[0], mass_range[1], num=nBins), right=True)
    pt_idxs = np.digitize(sig[:,4], bins=np.linspace(pt_range[0], pt_range[1], num=nBins), right=True)

    msd_idxs[msd_idxs>=nBins] = msd_idxs[nBins-1]
    pt_idxs[pt_idxs>=nBins] = pt_idxs[nBins-1]

    weights = weight_surface[msd_idxs-1,pt_idxs-1]
    weights_sig = np.nan_to_num(weights, nan = 1., posinf = 1., neginf = 1.)
    
    weights = np.concatenate((weights_sig, weights_bkg))
    data = np.concatenate((sig, bkg))
    
    return(weights, data)
