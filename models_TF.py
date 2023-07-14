import keras.backend as K
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv1D, Lambda, Dot, Flatten
from keras.models import Model

# X is the particle-feature matrix, Y is HLF-data (optional)
# Creates Particle-Particle interactions but needs modification for SV-Particle if you need that
def IN_TF(X, Y=None): 
    
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
    particlesConsidered = X.shape[1]
    entriesPerParticle = X.shape[2]
    inputParticle = Input(shape=(particlesConsidered, entriesPerParticle), name="inputParticle")
    
    if Y is not None: 
        eventDataLength = Y.shape[1]
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

    if Y is not None:
        OBar_withEvent = Lambda(lambda tensorListEvent: tf.concat((tensorListEvent[0], tensorListEvent[1]), axis=1), name="events")([OBar, inputEvent])
        denseEndOne = Dense(60, activation="relu", name="denseEndOne")(OBar_withEvent)

    else: 
        denseEndOne = Dense(60, activation="relu", name="denseEndOne")(OBar)

    normEndOne = BatchNormalization(momentum=0.6, name="normEndOne")(denseEndOne)
    denseEndTwo = Dense(30, activation="relu", name="denseEndTwo")(normEndOne)
    denseEndThree = Dense(10, activation="relu", name="denseEndThree")(denseEndTwo)
    output = Dense(1, activation="sigmoid", name="output")(denseEndThree)

    if Y is not None:
        model = Model(inputs=[inputParticle, inputEvent], outputs=[output])

    else: 
        model = Model(inputs=[inputParticle], outputs=[output])
    
    return model
        
        