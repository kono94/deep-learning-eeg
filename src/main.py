import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from scipy import fftpack, signal, stats
from skimage import util
from numpy.random import seed
import sys, getopt, time
from Filters import butter_bandpass_filter, notch_filter, butter_highpass_filter, normalize, sigma_clipping
from Model import createMasterCNN, plotAndSaveHistory, loadModel, predict
from Util import visualizeSpectrogram
from Mode import Mode

# setting random number generators to ensure some results
seed(1)
np.random.seed(1)

###  Train related variables ###
# 80% training, 20% validation data
trainSplit = 0.8
# number of samples to work through before updating the internal model parameters
batchSize = 256
# number of passes over the entire dataset
nrOfEpochs = 50
# window size of the generated spectrograms
spectroWindowSize = 128
# defines the grade of overlapping between spectrogram windows
# 8 timestamps will generate one pixel in the output spectrogram
spectroWindowShift = 8


### Global Parameters ###
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


### Starting parameters ###
# used to determine the mode (learning from dataset, predicting dataset or make
# live predictions from datastream)
try:
    opts, args = getopt.getopt(sys.argv[1:],'hw:t:p:m:l', ['train', 'predict', 'model', 'live'])
except getopt.GetoptError as err:
    print(err)
    sys.exit(2)

def help():
    print('-t --train filePath')
    print('-p --predict filePath')
    print('-l --live')
    print('-m --model modelPath')

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

# get raw eeg volt data
print("Loading in EEG data...")
X = np.loadtxt(filePath, usecols=channelsToUse, delimiter=",", skiprows=1)
# get labels
print("Loading in labels...")
Y = np.loadtxt(filePath, usecols=[8],dtype=np.str, delimiter=",", skiprows=1)


# split data to corresponding label
cropPuffer = [[] for i in range(0, numberOfChannels, 1)]
classPuffer = [[] for i in range(0, numberOfClasses, 1)]

# function to prepare the crops used for learning
# applies filters, normalizes and generates augmentated data crops 
def build_crops(channelCrops, label, classPuffer):
    minSizeOfCrop = len(channelCrops[0])
    if(len(channelCrops[0]) < cropWindowSize):
        print("Filtering crop because length is too small: ", len(channelCrops[0]))
        return False
   
    for ch in range(0, numberOfChannels, 1):
        data = channelCrops[ch]
        N = len(data)
        L = N / fs

        # remove 0.5 hz DC-offset
        y = butter_highpass_filter(data, 0.5, fs)
        # remove 50 hz power line frequency
        y = notch_filter(y, 50, fs)
        # extract frequencies between 2 and 60 hz
        # typical characteristics for motor imagery
        y = butter_bandpass_filter(y, 2, 60, fs)
        # remove high voltage spikes with six sigma clipping
        # +- sigma (default is 4, master thesis uses 6)
        y = sigma_clipping(y, 4, 4)
        # normalize each session and eletrode
        y = normalize(y)
        channelCrops[ch] = y
        if minSizeOfCrop > len(channelCrops[ch]):
            minSizeOfCrop = len(channelCrops[ch])

    for i in range(0, len(channelCrops)):
        if(len(channelCrops[0]) < cropWindowSize):
            print("Crop length too small after applying filters: ", channelCrops[0])
            return False

    totalNumberOfCrops = int((minSizeOfCrop- cropWindowSize)/cropWindowShift) -1
    for i in range(0, totalNumberOfCrops, 1):
        allChannels = []
        for j in range(0, numberOfChannels, 1):
            l = []
            allChannels.append(l)
        for w in range(0, cropWindowSize, 1):
            for ch in range(0, numberOfChannels, 1):
                allChannels[ch].append(channelCrops[ch][i*cropWindowShift + w])

        if label == "left":
            classPuffer[0].append(allChannels)
        if label == "right":
            classPuffer[1].append(allChannels)

## End build_crops

prevLabel = Y[0]
dataSetLen = len(X)
for i in range(0, dataSetLen, 1):
    sys.stdout.write("Processing csv: %d%% \r" % (i/dataSetLen * 100))
    sys.stdout.flush()
    if prevLabel != Y[i]:
        if prevLabel != "none":
            # build crops right here
            build_crops(cropPuffer, prevLabel, classPuffer)
            for ch in range(0, numberOfChannels, 1):
                cropPuffer[ch] = []
    for ch in range(0, numberOfChannels, 1):
        cropPuffer[ch].append(X[i][ch])
        if prevLabel == "none" and len(cropPuffer[ch]) > 256:
            cropPuffer[ch].pop(0)
    prevLabel = Y[i]


# visualize some spectrograms
visualizeSpectrogram(classPuffer, spectroWindowSize, spectroWindowShift, fs, nrOfSpectrogram=1)

print("Amount of left frames: ", len(classPuffer[0]))
print("Amount of right frames: ", len(classPuffer[1]))

minClassSize = len(classPuffer[0])
for i in range(0, len(classPuffer), 1):
    if minClassSize > len(classPuffer[i]):
        minClassSize = len(classPuffer[i])

print("Equalize left and right frames to same amount of: ", minClassSize)
mX = np.empty(shape=(minClassSize * numberOfClasses, numberOfChannels, cropWindowSize))
mY = np.empty(shape=(minClassSize * numberOfClasses, numberOfClasses))
index = 0
for i in range(0, minClassSize):
    sys.stdout.write("Processing crops: %d%%\r" % (i/minClassSize * 100))
    sys.stdout.flush()
    for cl in range(0, numberOfClasses):
        for ch in range(0, numberOfChannels):
            for w in range(0, cropWindowSize):
                mX[index][ch][w] = classPuffer[cl][i][ch][w]
        mY[index] = np.zeros(shape=numberOfClasses)
        mY[index][cl] = 1
        index = index + 1

print("X input shape: " , np.shape(mX))
print("Label input shape: ", np.shape(mY))

model = createMasterCNN(numberOfChannels, cropWindowSize, spectroWindowSize, spectroWindowShift, numberOfClasses)

if mode is Mode.TRAIN:
    # is this really necesary or does keras shuffle the data every epoch?
    # assuming that shuffle=True in keras.fit is shuffling correctly

    # rng_state = np.random.get_state()
    # np.random.shuffle(mX)
    # np.random.set_state(rng_state)
    # np.random.shuffle(mY)
    history = model.fit(mX, mY, shuffle=True, validation_split=(1 - trainSplit), batch_size=batchSize, epochs=nrOfEpochs)
    model.summary()
    model.save('models/model_' + str(int(round(time.time()))) +  '.h5')
    # saving acc and loss graph as png
    plotAndSaveHistory(history)
    # predicting classes for 1% of the dataset mX
    predict(model, mX, mY, 0.01)
elif mode is Mode.PREDICT:
   model = loadModel(modelPath)
   # TODO: verbose output. Look up keras docs
   print("keras.evaluate() result: ", model.evaluate(mX, mY))
   # predicting classes for 100% of the dataset mX
   predict(model, mX, mY, amount=1, verbose=0)
elif mode is Mode.LIVE:
    raise NotImplementedError("Live mode not implemented yet")
