import numpy as np
import h5py
import keras.backend as K
import tensorflow as tf
import json

def phi_kernel(a): 
    return (a + np.pi) % (2 * np.pi) - np.pi

print("Extracting")

fOne = h5py.File("data/FullQCD_FullSig_Zqq_fillfactor1_pTsdmassfilling_dRlimit08_50particlesordered_bkgFill_genMatched50.h5", 'r')
totalData = fOne["deepDoubleQ"][:]
print(fOne["deepDoubleQ"])
print(totalData.shape)

# Sets controllable values                                                                                                         
print('filtering out zero pt events')
#totalData = totalData[np.array([event[6]>0 for event in totalData])]
print('done')


particlesConsidered = 50
entriesPerParticle = 4

eventDataLength = 6

decayTypeColumn = -1

testDataLength = int(len(totalData)*0.1)
trainingDataLength = int(len(totalData)*0.8)

unnormalize_pT = False

validationDataLength = int(len(totalData)*0.1)
particleDataLength = particlesConsidered * entriesPerParticle

labels = totalData[:, decayTypeColumn:]
particleData = totalData[:, eventDataLength:particleDataLength + eventDataLength]

if unnormalize_pT:
    print('UNNORMALIZING pT')
    for i in range(50):
        particleData[:, i] = np.multiply(particleData[:, i], totalData[:, 4])

    print('Done')

pt = particleData[:, :100]
eta = particleData[:, 100:200]
phi = particleData[:, 200:300]
phi = [[phi_kernel(phik) for phik in event] for event in phi]
charge = particleData[:, 300:400]


'''
pt_charged = [[particleData[i, j] for j in range(100) if abs(particleData[i, j+300]) == 1] for i in range(len(particleData))]
pt_neutral =  [[particleData[i, j] for j in range(100) if particleData[i, j+300] == 0] for i in range(len(particleData))]
eta_charged = [[particleData[i, j+100] for j in range(100) if abs(particleData[i, j+300]) == 1] for i in range(len(particleData))]
eta_neutral =  [[particleData[i, j+100] for j in range(100) if particleData[i, j+300] == 0] for i in range(len(particleData))]
phi_charged = [[particleData[i, j+200] for j in range(100) if abs(particleData[i, j+300]) == 1] for i in range(len(particleData))]
phi_neutral =  [[particleData[i, j+200] for j in range(100) if particleData[i, j+300] == 0] for i in range(len(particleData))]


import matplotlib.pyplot as plt
plt.figure()
plt.title('pT particles')
plt.hist(np.array(pt_charged).flatten(), bins = 20, alpha=0.5, label='charged')
plt.hist(np.array(pt_neutral).flatten(), bins = 20, alpha=0.5, label='neutral')
plt.legend(loc='upper right')
plt.savefig('data/hist/pt_particle_hist.jpg')

plt.figure()
plt.title('eta particles')
plt.hist(np.array(eta_charged).flatten(), bins = 20, alpha=0.5, label='charged')
plt.hist(np.array(eta_neutral).flatten(), bins = 20, alpha=0.5, label='neutral')
plt.legend(loc='upper right')
plt.savefig('data/hist/eta_particle_hist.jpg')

plt.figure()
plt.title('phi particles')
plt.hist(np.array(phi_charged).flatten(), bins = 20, alpha=0.5, label='charged')
plt.hist(np.array(phi_neutral).flatten(), bins = 20, alpha=0.5, label='neutral')
plt.legend(loc='upper right')
plt.savefig('data/hist/phi_particle_hist.jpg')
'''

pt_sig = [particleData[i, :100] for i in range(len(particleData)) if labels[i] == 1]
pt_bkg = [particleData[i, :100] for i in range(len(particleData)) if labels[i] == 0]
eta_sig = [particleData[i, 100:200] for i in range(len(particleData)) if labels[i] == 1]
eta_bkg = [particleData[i, 100:200] for i in range(len(particleData)) if labels[i] == 0]
phi_sig = [particleData[i, 200:300] for i in range(len(particleData)) if labels[i] == 1]
phi_bkg = [particleData[i, 200:300] for i in range(len(particleData)) if labels[i] == 0]
charge_sig = [particleData[i, 300:400] for i in range(len(particleData)) if labels[i] == 1]
charge_bkg = [particleData[i, 300:400] for i in range(len(particleData)) if labels[i] == 0]

jet_eta_sig = [totalData[i, 0] for i in range(len(totalData)) if labels[i] == 1] 
jet_phi_sig = [totalData[i, 1] for i in range(len(totalData)) if labels[i] == 1]
jet_EhadOverEem_sig = [totalData[i, 2] for i in range(len(totalData)) if labels[i] == 1]
jet_mass_sig = [totalData[i, 3] for i in range(len(totalData)) if labels[i] == 1]
jet_pT_sig = [totalData[i, 4] for i in range(len(totalData)) if labels[i] == 1]
jet_sdmass_sig = [totalData[i, 5] for i in range(len(totalData)) if labels[i] == 1]

jet_eta_bkg = [totalData[i, 0] for i in range(len(totalData)) if labels[i] == 0]
jet_phi_bkg = [totalData[i, 1] for i in range(len(totalData)) if labels[i] == 0] 
jet_EhadOverEem_bkg = [totalData[i, 2] for i in range(len(totalData)) if labels[i] == 0]
jet_mass_bkg = [totalData[i, 3] for i in range(len(totalData)) if labels[i] == 0]
jet_pT_bkg = [totalData[i, 4] for i in range(len(totalData)) if labels[i] == 0]
jet_sdmass_bkg = [totalData[i, 5] for i in range(len(totalData)) if labels[i] == 0]    

import matplotlib.pyplot as plt

'''
for i in range(100): 
    plt.figure()
    plt.title('pT particle #' + str(i))
    plt.hist(np.array(pt_sig)[:, i], bins = 60, alpha=0.5, label='sig')
    plt.hist(np.array(pt_bkg)[:, i], bins = 60, alpha=0.5, label='bkg')
    plt.legend(loc='upper right')
    plt.savefig('data/hist/pt_hist_particle' + str(i) + '.jpg')


for i in range(100):
    plt.figure()
    plt.title('eta particle #' + str(i))
    plt.hist(np.array(eta_sig)[:, i], bins = 20, alpha=0.5, label='sig')
    plt.hist(np.array(eta_bkg)[:, i], bins = 20, alpha=0.5, label='bkg')
    plt.legend(loc='upper right')                                                                                                                                                                                                                                            
    plt.savefig('data/hist/eta_hist_particle' + str(i) + '.jpg')

for i in range(100):
    plt.figure()
    plt.title('phi particle #' + str(i))
    plt.hist(np.array(phi_sig)[:, i], bins = 20, alpha=0.5, label='sig')
    plt.hist(np.array(phi_bkg)[:, i], bins = 20, alpha=0.5, label='bkg')
    plt.legend(loc='upper right')
    plt.savefig('data/hist/phi_hist_particle' + str(i) + '.jpg')

plt.figure()
plt.title('pT sig')
plt.hist(np.array(pt_sig).flatten(), bins = 20)
plt.savefig('data/hist/pt_sig_hist.jpg')

plt.figure()
plt.title('pT bkg')
plt.hist(np.array(pt_bkg).flatten(), bins = 20)
plt.savefig('data/hist/pt_bkg_hist.jpg')

plt.figure()
plt.title('pT')
plt.hist(np.array(pt_sig).flatten(), bins = 20, range=[0, 0.15], alpha=0.5, label='sig')
plt.hist(np.array(pt_bkg).flatten(), bins = 20, range=[0, 0.15], alpha=0.5, label='bkg')
plt.legend(loc='upper right') 
plt.savefig('data/hist/pt_dual_hist.jpg')
print(np.histogram(np.array(pt_sig).flatten(), bins = 20, range=[0, 0.15]))
print(np.histogram(np.array(pt_sig).flatten(), bins = 20, range=[0, 0.15]))

plt.figure()
plt.title('eta sig')
plt.hist(np.array(eta_sig).flatten(), bins = 20)
plt.savefig('data/hist/eta_sig_hist.jpg')

plt.figure()
plt.title('eta_bkg')
plt.hist(np.array(eta_bkg).flatten(), bins = 20)
plt.savefig('data/hist/eta_bkg_hist.jpg')

plt.figure()
plt.title('eta')
plt.hist(np.array(eta_sig).flatten(), bins = 20, alpha=0.5, label='sig')
plt.hist(np.array(eta_bkg).flatten(), bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/eta_dual_hist.jpg')

plt.figure()
plt.title('phi sig')
plt.hist(np.array(phi_sig).flatten(), bins = 20)
plt.savefig('data/hist/phi_sig_hist.jpg')

plt.figure()
plt.title('phi bkg')
plt.hist(np.array(phi_bkg).flatten(), bins = 20)
plt.savefig('data/hist/phi_bkg_hist.jpg')

plt.figure()
plt.title('phi')
plt.hist([(i+ np.pi) % (2 * np.pi) - np.pi for i in np.array(phi_sig).flatten()], bins = 20, alpha=0.5, label='sig')
plt.hist([(i+ np.pi) % (2 * np.pi) - np.pi for i in np.array(phi_bkg).flatten()], bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/phi_dual_hist.jpg')


plt.figure()
plt.title('charge sig')
plt.hist(np.array(charge_sig).flatten(), bins = 4)
plt.savefig('data/hist/charge_sig_hist.jpg')

plt.figure()
plt.title('charge bkg')
plt.hist(np.array(charge_bkg).flatten(), bins = 4)
plt.savefig('data/hist/charge_bkg_hist.jpg')


plt.figure()
plt.title('charge')
plt.hist(np.array(charge_sig).flatten(), bins = 4, alpha=0.5, label='sig')
plt.hist(np.array(charge_bkg).flatten(), bins = 4, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/charge_dual_hist.jpg')
'''

plt.figure()
plt.title('jet_eta')
plt.hist(np.array(jet_eta_sig).flatten(), bins = 20, alpha=0.5, label='sig')
plt.hist(np.array(jet_eta_bkg).flatten(), bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/jet_eta_dual_hist.jpg')

plt.figure()
plt.title('jet_mass')
plt.hist(np.array(jet_mass_sig).flatten(), bins = 20, alpha=0.5, label='sig')
plt.hist(np.array(jet_mass_bkg).flatten(), bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/jet_mass_dual_hist.jpg')

plt.figure()
plt.title('jet_phi')
plt.hist(np.array(jet_phi_sig).flatten(), bins = 20, alpha=0.5, label='sig')
plt.hist(np.array(jet_phi_bkg).flatten(), bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/jet_phi_dual_hist.jpg')

plt.figure()
plt.title('jet_sdmass')
plt.hist(np.array(jet_sdmass_sig).flatten(), bins = 20, alpha=0.5, label='sig')
plt.hist(np.array(jet_sdmass_bkg).flatten(), bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/jet_sdmass_dual_hist.jpg')

plt.figure()
plt.title('jet_pT')
plt.hist(np.array(jet_pT_sig).flatten(), bins = 20, alpha=0.5, label='sig')
plt.hist(np.array(jet_pT_bkg).flatten(), bins = 20, alpha=0.5, label='bkg')
plt.legend(loc='upper right')
plt.savefig('data/hist/jet_pT_dual_hist.jpg')


plt.figure()
plt.title('pT, mass signal histogram')
plt.hist2d(np.array(jet_pT_sig).flatten(), np.array(jet_sdmass_sig).flatten())
plt.colorbar()
plt.xlabel('jet_pT')
plt.ylabel('jet_mass')
plt.savefig('data/hist/jet_pT_mass_hist2d_sig.jpg')


plt.figure()
plt.title('pT, mass bkg histogram')
plt.hist2d(np.array(jet_pT_bkg).flatten(), np.array(jet_sdmass_bkg).flatten())
plt.colorbar()
plt.xlabel('jet_pT')
plt.ylabel('jet_mass')
plt.savefig('data/hist/jet_pT_mass_hist2d_bkg.jpg')
 


