# Imports basics

import numpy as np
import h5py
import keras.backend as K
import tensorflow as tf
import json
import setGPU

# Imports neural net tools

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv1D, Lambda, Dot, Flatten
from keras.models import Model

# Opens files and reads data

print("Extracting")

fOne = h5py.File("data/FullQCD_FullSig_Zqq_fillfactor1_pTsdmassfilling_dRlimit08_50particlesordered_sigFill_genMatched50.h5", 'r')
totalData = fOne["deepDoubleQ"][:]
print(totalData.shape)

# Sets controllable values

particlesConsidered = 100
particlesPostCut = 100
entriesPerParticle = 4

eventDataLength = 6

decayTypeColumn = -1

trainingDataLength = int(len(totalData)*0.8)

validationDataLength = int(len(totalData)*0.1)

numberOfEpochs = 100
batchSize = 512

unnormalize_pT = False
charged_particles_only = False
removeLowMultiplicity = False 
includeeventData = False
onepad = False

modelName = "IN_FlatSamples_EighthQCDEighthSig_50particles_pTsdmassfilling_dRlimit08"

# Creates Training Data

print("Preparing Data")

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

eventTrainingData = np.array(eventData[0:trainingDataLength])
particleTrainingData = np.transpose(
    particleData[0:trainingDataLength, ].reshape(trainingDataLength, entriesPerParticle, particlesConsidered),
    axes=(0, 2, 1))
trainingLabels = np.array(labels[0:trainingDataLength])


if charged_particles_only: 
    selected_events = np.array([i for i in range(len(particleTrainingData)) if np.sum(np.abs(particleTrainingData[i,:,3])) >= 1])
    particleTrainingData = particleTrainingData[selected_events]
    trainingLabels = trainingLabels[selected_events]
    TrainingData = TrainingData[selected_events]
    print('Selecting only charged particles')
    zeros = np.zeros((50, 4))
    particleTrainingData = np.array([np.concatenate(([particle for particle in event if abs(particle[3]) == 1], zeros))[:50] for event in particleTrainingData], dtype='float32')
elif removeLowMultiplicity: 
    print('Removing low multiplicity events')
    selected_events = np.array([i for i in range(len(particleTrainingData)) if np.sum(np.abs(particleTrainingData[i,:,3])) == 50])               
    particleTrainingData = particleTrainingData[selected_events]
    trainingLabels = trainingLabels[selected_events]
    eventTrainingData = eventTrainingData[selected_events]

eventValidationData = np.array(eventData[trainingDataLength:trainingDataLength + validationDataLength])
particleValidationData = np.transpose(
    particleData[trainingDataLength:trainingDataLength + validationDataLength, ].reshape(validationDataLength,
                                                                                         entriesPerParticle,
                                                                                         particlesConsidered),
    axes=(0, 2, 1))
validationLabels = np.array(labels[trainingDataLength:trainingDataLength + validationDataLength])

if charged_particles_only:
    selected_events = np.array([i for i in range(len(particleValidationData)) if np.sum(np.abs(particleValidationData[i,:,3])) >= 1])
    particleValidationData = particleValidationData[selected_events]
    validationLabels = validationLabels[selected_events]
    print('Selecting only charged particles')
    zeros = np.zeros((50, 4))
    particleValidationData = np.array([np.concatenate(([particle for particle in event if abs(particle[3]) == 1], zeros))[:50] for event in particleValidationData], dtype='float32')

elif removeLowMultiplicity:
    print('Removing low multiplicity events')
    selected_events = np.array([i for i in range(len(particleValidationData)) if np.sum(np.abs(particleValidationData[i,:,3])) == 50])
    particleValidationData = particleValidationData[selected_events]
    validationLabels = validationLabels[selected_events]
    eventValidationData = eventValidationData[selected_events]

particleTestData = np.transpose(particleData[trainingDataLength + validationDataLength:, ].reshape(
    len(particleData) - trainingDataLength - validationDataLength, entriesPerParticle, particlesConsidered),
                                axes=(0, 2, 1))
testLabels = np.array(labels[trainingDataLength + validationDataLength:])

print('Selecting particlesPostCut')
particleTrainingData = particleTrainingData[:, :particlesPostCut]
particleValidationData = particleValidationData[:, :particlesPostCut]

if onepad: 
    print('1-PADDING')
    trainingmissing = np.array([int(50-np.sum(np.abs(particleTrainingData[i,:,3]))) for i in range(len(particleTrainingData))])
    validationmissing = np.array([int(50-np.sum(np.abs(particleValidationData[i,:,3]))) for i in range(len(particleValidationData))])
    for i in range(len(particleTrainingData)): 
        for j in range(trainingmissing[i]):
            if int(np.sum(np.abs(particleTrainingData[i, -(j+1), :]) == 0)):
                particleTrainingData[i, -(j+1), 3] = 1
            
    for i in range(len(particleValidationData)):
        for j in range(validationmissing[i]):
            if np.sum(np.abs(particleValidationData[i, -(j+1), :] == 0)):
                 particleValidationData[i, -(j+1), 3] = 1

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

#model = DNN(particleTrainingData)
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

print('Calculating')

modelCallbacks = [EarlyStopping(patience=10),
                  ModelCheckpoint(filepath="./data/"+modelName+".h5", save_weights_only=True,
                                  save_best_only=True)]

if includeeventData:
    history = model.fit([particleTrainingData, eventTrainingData], trainingLabels, epochs=numberOfEpochs, batch_size=batchSize,
                    callbacks=modelCallbacks,
                    validation_data=([particleValidationData, eventValidationData], validationLabels))

else: 
    history = model.fit([particleTrainingData], trainingLabels, epochs=numberOfEpochs, batch_size=batchSize,
                    callbacks=modelCallbacks,
                    validation_data=([particleValidationData], validationLabels))

with open("./data/"+modelName+"history.json", "w") as f:
    json.dump(history.history,f)

print("Loading weights")

model.load_weights("./data/"+modelName+".h5")

model.save("./data/"+modelName+"model")

print("Predicting")

#predictions = model.predict([particleTestData])
