import numpy as np
import h5py
import keras.backend as K
import tensorflow as tf
import json

# Imports neural net tools

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv1D, Lambda, Dot, Flatten
from keras.models import Model


print("Extracting")

fOne = h5py.File("data/FullQCD_FullSig_Zqq_fillfactor1_pTsdmassfilling_dRlimit08_50particlesordered_bkgFill_genMatched50.h5", 'r')
totalData = fOne["deepDoubleQ"][:]
print(totalData[:, 0])
print(totalData[:, 1])

modelName = "IN_FlatSamples_FullQCDFullSig_50particles_pTsdmassfillsig1dmass_dRlimit08_genMatched50"

# Sets controllable values

particlesConsidered = 50
particlesPostCut = 50
entriesPerParticle = 4

eventDataLength = 6

decayTypeColumn = -1

testDataLength = int(len(totalData)*0.1)
trainingDataLength = int(len(totalData)*0.8)

unnormalize_pT = False
charged_particles_only = False
removeLowMultiplicity = False
includeeventData = False
onepad = False

validationDataLength = int(len(totalData)*0.1)
particleDataLength = particlesConsidered * entriesPerParticle

np.random.seed(42)
np.random.shuffle(totalData)

labels = totalData[:, decayTypeColumn:]
particleData = totalData[:, eventDataLength:particleDataLength + eventDataLength]
eventData = totalData[:, :eventDataLength]

if unnormalize_pT:
    print('UNNORMALIZING pT')
    for i in range(50):
        particleData[:, i] = np.multiply(particleData[:, i], totalData[:, 4])

    print('Done')

testData = totalData[trainingDataLength + validationDataLength:, ]
particleTestData = np.transpose(particleData[trainingDataLength + validationDataLength:, ].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerParticle, particlesConsidered),
                                axes=(0, 2, 1))
testLabels = np.array(labels[trainingDataLength + validationDataLength:])
eventTestData = np.array(eventData[trainingDataLength + validationDataLength:])

if charged_particles_only:
    selected_events = np.array([i for i in range(len(particleTestData)) if np.sum(np.abs(particleTestData[i,:,3])) >= 1])
    particleTestData = particleTestData[selected_events]
    testLabels = testLabels[selected_events]
    print('Selecting only charged particles')
    zeros = np.zeros((50, 4))
    particleTestData = np.array([np.concatenate(([particle for particle in event if abs(particle[3]) == 1], zeros))[:50] for event in particleTestData], dtype='float32')

elif removeLowMultiplicity:
    print('Removing low multiplicity events')
    selected_events = np.array([i for i in range(len(particleTestData)) if np.sum(np.abs(particleTestData[i,:,3])) == 50])
    particleTestData = particleTestData[selected_events]
    testLabels = testLabels[selected_events]
    eventTestData = eventTestData[selected_events]
particleTestData = particleTestData[:, :particlesPostCut]


if onepad:
    print('1-PADDING')
    testmissing = np.array([int(50-np.sum(np.abs(particleTestData[i,:,3]))) for i in range(len(particleTestData))])
    for i in range(len(particleTestData)):
        for j in range(testmissing[i]):
            if int(np.sum(np.abs(particleTestData[i, -(j+1), :]) == 0)):
                particleTestData[i, -(j+1), 3] = 1


# Defines the interaction matrices
particlesConsidered = particlesPostCut
# Defines the recieving matrix for particles
RR = []
for i in range(particlesConsidered):
    row = []
    for j in range(particlesConsidered * (particlesConsidered - 1)):
        if j in range(i * (particlesConsidered - 1), (i + 1) * (particlesConsidered - 1)):
            row.append(1.0)
        else:
            row.append(0.0)
    RR.append(row)
RR = np.array(RR)
RR = np.float32(RR)
RRT = np.transpose(RR)

# Defines the sending matrix for particles
RST = []
for i in range(particlesConsidered):
    for j in range(particlesConsidered):
        row = []
        for k in range(particlesConsidered):
            if k == j:
                row.append(1.0)
            else:
                row.append(0.0)
        RST.append(row)
rowsToRemove = []
for i in range(particlesConsidered):
    rowsToRemove.append(i * (particlesConsidered + 1))
RST = np.array(RST)
RST = np.float32(RST)
RST = np.delete(RST, rowsToRemove, 0)
RS = np.transpose(RST)


# Creates and trains the neural net                                                                                                                                                                                                                         

# Particle data interaction NN                                                                                                                                                                                                                              
inputParticle = Input(shape=(particlesConsidered, entriesPerParticle), name="inputParticle")
inputEvent = Input(shape=(eventDataLength), name="inputEvent")

XdotRR = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RR, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRR")(inputParticle)
XdotRS = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RS, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRS")(inputParticle)
Bpp = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bpp")([XdotRR, XdotRS])

convOneParticle = Conv1D(80, kernel_size=1, activation="relu", name="convOneParticle")(Bpp)
convTwoParticle = Conv1D(50, kernel_size=1, activation="relu", name="convTwoParticle")(convOneParticle)
convThreeParticle = Conv1D(30, kernel_size=1, activation="relu", name="convThreeParticle")(convTwoParticle)

Epp = BatchNormalization(momentum=0.6, name="Epp")(convThreeParticle)

# Combined prediction NN                                                                                                                                                                                                                                    
EppBar = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RRT, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="EppBar")(Epp)
C = Lambda(lambda listOfTensors: tf.concat((listOfTensors[0], listOfTensors[1]), axis=2), name="C")(
    [inputParticle, EppBar])

convPredictOne = Conv1D(80, kernel_size=1, activation="relu", name="convPredictOne")(C)
convPredictTwo = Conv1D(50, kernel_size=1, activation="relu", name="convPredictTwo")(convPredictOne)

O = Conv1D(24, kernel_size=1, activation="relu", name="O")(convPredictTwo)

# Calculate output                                                                                                                                                                                                                                          
OBar = Lambda(lambda tensor: K.sum(tensor, axis=1), name="OBar")(O)

if includeeventData:
    OBar_withEvent = Lambda(lambda tensorListEvent: tf.concat((tensorListEvent[0], tensorListEvent[1]), axis=1), name="events")([OBar, inputEvent])
    denseEndOne = Dense(60, activation="relu", name="denseEndOne")(OBar_withEvent)

else:
    denseEndOne = Dense(60, activation="relu", name="denseEndOne")(OBar)

normEndOne = BatchNormalization(momentum=0.6, name="normEndOne")(denseEndOne)
denseEndTwo = Dense(30, activation="relu", name="denseEndTwo")(normEndOne)
denseEndThree = Dense(10, activation="relu", name="denseEndThree")(denseEndTwo)
output = Dense(1, activation="sigmoid", name="output")(denseEndThree)

print("Compiling")

def DNN(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = Flatten()(inputs)
    x = Dense(int(25), activation='relu')(x)
    x = Dense(int(10), activation='relu')(x)
    x = Dense(int(5), activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)
    return model

if includeeventData:
    model = Model(inputs=[inputParticle, inputEvent], outputs=[output])

else:
    model = Model(inputs=[inputParticle], outputs=[output])

#model = DNN(particleTestData)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

model.load_weights("./data/"+modelName+".h5")
print("Predicting")

if includeeventData:
    predictions = model.predict([particleTestData, eventTestData])
else: 
    predictions = model.predict([particleTestData])

    

predictions = [[i[0], 1-i[0]] for i in predictions]
testLabels = np.array([[i[0], 1-i[0]] for i in testLabels])
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc, accuracy_score

print(predictions)
print(testLabels)

fpr, tpr, threshold = roc_curve(np.array(testLabels).reshape(-1), np.array(predictions).reshape(-1))
plt.plot(fpr, tpr, lw=2.5, label="{}, AUC = {:.1f} %".format('ZprimeAtoqq IN',auc(fpr,tpr)*100))
plt.title('ROC Curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.savefig('./data/{}/model_ROC.jpg'.format(modelName))


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]

fpr, tpr, threshold = roc_curve(np.array(testLabels).reshape(-1), np.array(predictions).reshape(-1))
cuts = {}
for wp in [0.01, 0.03, 0.05, 1.0]:#0.1, 0.5, 1.0]: # % mistag rate
    idx, val = find_nearest(fpr, wp)
    cuts[str(wp)] = threshold[idx] # threshold for deep double-b corresponding to ~1% mistag rate
        
f, ax = plt.subplots(figsize=(10,10))

print('sculpting')
sculpt_vars = ['jet_eta', "jet_phi","jet_EhadOverEem","jet_mass", 'jet_pT', 'jet_sdmass']
for i in range(len(sculpt_vars)):
    f, ax = plt.subplots(figsize=(10,10))

    for wp, cut in reversed(sorted(cuts.items())):
        predictions = np.array(predictions)
        ctdf = np.array([testData[pred, i] for pred in range(len(predictions)) if predictions[pred,0] > cut])
        weight = np.array([testLabels[pred, 1] for pred in range(len(predictions)) if predictions[pred,0] > cut])
        
        if str(wp)=='1.0':
            ax.hist(ctdf.flatten(), weights = weight/np.sum(weight), lw=2,
                        histtype='step',label='No tagging applied ({} Events)'.format(len(ctdf.flatten())), bins = 15)
        else:
            ax.hist(ctdf.flatten(), weights = weight/np.sum(weight), lw=2,
                        histtype='step',label='{}%  mistagging rate ({} Events)'.format(float(wp)*100., len(ctdf.flatten())), bins=15)

    ax.set_xlabel(sculpt_vars[i])
    ax.set_ylabel('Normalized Scale QCD')
    ax.set_title(sculpt_vars[i] + ' 50 Particle Sculpting Sig Fill 1D Reweight sdmass') 
    ax.legend() 

    f.savefig('data/{}/sculpting_'.format(modelName) + sculpt_vars[i] + '.jpg')
    
print(predictions)
print(testLabels)
print('ptBinning')
hist, pt_bins = np.histogram(testData[:, 4], bins=15)
for pt_bin in range(len(pt_bins)): 
    
    testDataNew = []
    testLabelsNew = []
    predictionsNew = []
    for i in range(len(testData)): 
        if testData[i, 4] < pt_bins[pt_bin + 1]: 
            if testData[i, 4] > pt_bins[pt_bin]:
                testDataNew.append(testData[i])
                testLabelsNew.append(testLabels[i])
                predictionsNew.append(predictions[i])
    
    testDataNew = np.array(testDataNew)
    testLabelsNew = np.array(testLabelsNew)
    predictionsNew = np.array(predictionsNew)
    
    print(testDataNew)
    print(testLabelsNew)
    print(predictionsNew)
    
    fpr, tpr, threshold = roc_curve(np.array(testLabelsNew).reshape(-1), np.array(predictionsNew).reshape(-1))
    cuts = {}
    for wp in [0.01, 0.03, 0.05, 1.0]:#0.1, 0.5, 1.0]: # % mistag rate
        idx, val = find_nearest(fpr, wp)
        cuts[str(wp)] = threshold[idx] # threshold for deep double-b corresponding to ~1% mistag rate
    
    for i in range(len(sculpt_vars)):
        f, ax = plt.subplots(figsize=(10,10))

        for wp, cut in reversed(sorted(cuts.items())):
            predictions = np.array(predictions)
            ctdf = np.array([testDataNew[pred, i] for pred in range(len(predictionsNew)) if predictionsNew[pred,0] > cut])
            weight = np.array([testLabelsNew[pred, 1] for pred in range(len(predictionsNew)) if predictionsNew[pred,0] > cut])

            if str(wp)=='1.0':
                ax.hist(ctdf.flatten(), weights = weight/np.sum(weight), lw=2,
                            histtype='step',label='No tagging applied ({} Events)'.format(len(ctdf.flatten())), bins = 20)
            else:
                ax.hist(ctdf.flatten(), weights = weight/np.sum(weight), lw=2,
                            histtype='step',label='{}%  mistagging rate ({} Events)'.format(float(wp)*100., len(ctdf.flatten())), bins=20)

        ax.set_xlabel(sculpt_vars[i])
        ax.set_ylabel('Normalized Scale QCD')
        ax.set_title(sculpt_vars[i] + ' 50 Particle Sculpting Sig Fill 1D Reweight sdmass') 
        ax.legend() 

        f.savefig('data/{}/sculpting_PtBin{}-{}_'.format(modelName, pt_bins[pt_bin],pt_bins[pt_bin + 1]) + sculpt_vars[i] + '.jpg')


