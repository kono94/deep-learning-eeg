import random
random.seed(123)
import numpy as np
from numpy.random import seed
seed(123)
import tensorflow as tf
tf.random.set_seed(123)
import sys, getopt, time, os
from Model import createCNN, plotAndSaveHistory, loadModel, predict
from Util import visualizeSpectrogram
from Mode import Mode
from PreProcessor import applyFilters, preProcess
from NetworkType import NetworkType

# setting random number generators to ensure some results
### Starting parameter
def help():
    print('-t --train filePath')
    print('-p --predict filePath')
    print('-l --live')
    print('-m --model modelPath')


###  Train related variables ###
# 80% training, 20% validation data
trainSplit = 0.8
# number of samples to work through before updating the internal model parameters
batchSize = 512
# number of passes over the entire dataset
nrOfEpochs = 8
# window size of the generated spectrograms
spectroWindowSize = 128
# defines the grade of overlapping between spectrogram windows
# 8 timestamps will generate one pixel in the output spectrogram
spectroWindowShift = 8


### Global Parameters ###
# use saved up numpy arrays to skip preprocessing and test different network configurations
useCachedNumpyArrays = True
# network to train with, see ENUM "NetworkType"
networkToUse = NetworkType.CNN_PROPOSED_MASTER_THESIS

channelsToUse = [2,3,5]
numberOfChannels = len(channelsToUse)
# defines the labels to which data crops should be buit
# currently hard coded to only work with "left" and "right"
classes = ["left", "right"]
numberOfClasses = len(classes)
# used for Data Augmentation (generate more data to learn from)
# size of one data crop. Input for the CNN
# for example: 500 equals 2 seconds because frequency is 250 for the openBCI helmet
cropWindowSize = 512
# amount of timestamps to shift the crop window
# less means more data augmentation (more crop windows) but more similar data
# more means less data augmentation but more distinct data
cropWindowShift = 5

# frequency of the OpenBCI helmet
fs = 250
# defines which channels to use data from
# 0 = Fp1 (left forehead)
# 1 = Fp2 (right forehead)
# 2 = C3 (left side motor cortex)
# 3 = C4 (right side motor cortex)
# 4 = F3
# 5 = Cz (center motor cortex)
# 6 = F4
# 7 = Fz

### Starting parameters ###
# used to determine the mode (learning from dataset, predicting dataset or make
# live predictions from datastream)
try:
    opts, args = getopt.getopt(sys.argv[1:],'hw:t:p:m:l', ['train', 'predict', 'model', 'live'])
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

mode = None
filePath = None
modelPath = None
for opt, arg in opts:
    if opt in ('-t', '--train'):
        mode = Mode.TRAIN
        filePath = arg
    elif opt in ('-p', '--predict'):
        mode = Mode.PREDICT
        filePath = arg
    elif opt in ('-l', '--live'):
        mode = Mode.LIVE
    elif opt in ('-m', '--model'):
        modelPath = arg
    else:
        help()
        raise ValueError("Invalid start parameter")

if mode is Mode.PREDICT and modelPath is None:
     raise ValueError("No modelPath (-m, --model) given")


def loadData():
    # get raw eeg volt data
    print("Loading in EEG data...")
    X = np.loadtxt(filePath, usecols=channelsToUse, delimiter=",", skiprows=1)
    # get labels
    print("Loading in labels...")
    Y = np.loadtxt(filePath, usecols=[8],dtype=np.str, delimiter=",", skiprows=1)
    return X,Y

model = createCNN(networkToUse, numberOfChannels, cropWindowSize, spectroWindowSize, spectroWindowShift, numberOfClasses)

if mode is Mode.TRAIN:
    if os.path.exists("data/mX.npy") and useCachedNumpyArrays:
        mX = np.load("data/mX.npy")
        mY = np.load("data/mY.npy")
    else:
        X, Y = loadData()
        (mX, mY) = preProcess(X, Y, numberOfChannels, numberOfClasses, cropWindowSize, cropWindowShift, fs)
        np.save("data/mX", mX)
        np.save("data/mY", mY)

    # visualize some spectrograms
    # visualizeSpectrogram(mX, spectroWindowSize, spectroWindowShift, fs, nrOfSpectrogram=1)

    print("X input shape: " , np.shape(mX))
    print("Label input shape: ", np.shape(mY))


    # is this really necesary or does keras shuffle the data every epoch?
    # assuming that shuffle=True in keras.fit is shuffling correctly
    rng_state = np.random.get_state()
    np.random.shuffle(mX)
    np.random.set_state(rng_state)
    np.random.shuffle(mY)
    history = model.fit(mX, mY, shuffle=True, validation_split=(1 - trainSplit), batch_size=batchSize, epochs=nrOfEpochs)
    model.summary()
    model.save('models/model_' + str(int(round(time.time()))) +  '.h5')
    # saving acc and loss graph as png
    plotAndSaveHistory(history)
    # predicting classes for 1% of the dataset mX
    # predict(model, mX, mY, 0.01)
elif mode is Mode.PREDICT:
   X, Y = loadData()
   (mX, mY) = preProcess(X, Y, numberOfChannels, numberOfClasses, cropWindowSize, cropWindowShift, fs)
   # visuals some spectrograms
   visualizeSpectrogram(mX, spectroWindowSize, spectroWindowShift, fs, nrOfSpectrogram=1)
   model = loadModel(modelPath)

   print("keras.evaluate() result: ", model.evaluate(mX, mY))
   # predicting classes for 100% of the dataset mX
   predict(model, mX, mY, amount=1, verbose=0)
elif mode is Mode.LIVE:
    raise NotImplementedError("Live mode not implemented yet")
