import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack, signal, stats
from skimage import util
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPool2D, ReLU, Dropout, Lambda
from kapre.time_frequency import Spectrogram
from numpy.random import seed
seed(1)
np.random.seed(1)

########### FILTER FUNCTIONS #############
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = fs / 2
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def notch_filter(data, cutFreq, fs, quality=30):
     b, a = signal.iirnotch(cutFreq, quality, fs)
     y = signal.filtfilt(b, a, data)
     return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = fs /2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

def six_sigma_clipping(data, lowS, highS):
    y, low, upp = stats.sigmaclip(data, lowS, highS)
    return y


def normalize(data):
    mean = np.mean(data)
    sd = np.std(data)
    y = []
    for i in range(0, len(data), 1):
        y.append((data[i] - mean) / sd)
    return np.array(y)

fs = 250
channelsToUse = [2,3,5]
numberOfChannels = len(channelsToUse)
classes = 2
windowShift = 5
windowSize = 512
X = np.loadtxt("data/kek.csv", usecols=channelsToUse, delimiter=",", skiprows=1)
# get label
Y = np.loadtxt("data/kek.csv", usecols=[8],dtype=np.str, delimiter=",", skiprows=1)
# split data to corresponding label

cropPuffer = [[] for i in range(0, numberOfChannels, 1)]
classPuffer = [[] for i in range(0, classes, 1)]

def build_crops(channelCrops, label, classPuffer):
    minSizeOfCrop = len(channelCrops[0])
    if(len(channelCrops[0]) < windowSize):
        print(len(channelCrops[0]))
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
        y = six_sigma_clipping(y, 4, 4)
        # normalize each session and eletrode
        y = normalize(y)
        channelCrops[ch] = y
        if minSizeOfCrop > len(channelCrops[ch]):
            minSizeOfCrop = len(channelCrops[ch])

    for i in range(0, len(channelCrops)):
        if(len(channelCrops[0]) < windowSize):
            print("fail" + channelCrops[0])
            return False

    totalNumberOfCrops = int((minSizeOfCrop- windowSize)/windowShift) -1
    for i in range(0, totalNumberOfCrops, 1):
        allChannels = []
        for j in range(0, numberOfChannels, 1):
            l = []
            allChannels.append(l)
        for w in range(0, windowSize, 1):
            for ch in range(0, numberOfChannels, 1):
                allChannels[ch].append(channelCrops[ch][i*windowShift + w])

        if label == "left":
            classPuffer[0].append(allChannels)
        if label == "right":
            classPuffer[1].append(allChannels)


prevLabel = Y[0]
for i in range(0, len(X), 1):
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

# generate spectogram
spectroWindowSize = 128
spectroStep = 8
for i in range(0, 0, 1):
    plt.xlabel("Sample")
    plt.ylabel("Voltage in mV")
    plt.plot(classPuffer[0][i][0])
    plt.show()

    freqs, times, Sx = signal.spectrogram(np.asarray(classPuffer[0][i][0]), fs=fs, window='hanning', nperseg=spectroWindowSize,
                                          noverlap=spectroWindowSize - spectroStep, detrend=False, scaling='spectrum')


    f, ax = plt.subplots(figsize=(10, 5))
    ax.pcolormesh(times, freqs, np.log10(Sx), cmap='viridis')
    ax.set_ylabel('Frequency in Hz')
    ax.set_xlabel('Time in s')
    plt.show()


print(len(classPuffer[0]))
print(len(classPuffer[1]))

minClassSize = len(classPuffer[0])
for i in range(0, len(classPuffer), 1):
    if minClassSize > len(classPuffer[i]):
        minClassSize = len(classPuffer[i])

mX = np.empty(shape=(minClassSize * classes, numberOfChannels, windowSize))
mY = np.empty(shape=(minClassSize * classes, classes))
index = 0
for i in range(0, minClassSize, 1):
    for cl in range(0, classes, 1):
        for ch in range(0, numberOfChannels, 1):
            for w in range(0, windowSize, 1):
                mX[index][ch][w] = classPuffer[cl][i][ch][w]
        mY[index] = np.zeros(shape=classes)
        mY[index][cl] = 1
        index = index + 1

print(np.shape(mX))
print(np.shape(mY))

rng_state = np.random.get_state()
np.random.shuffle(mX)
np.random.set_state(rng_state)
np.random.shuffle(mY)

#create model
model = Sequential()
input_shape = (numberOfChannels, windowSize)
spectroWindowSize = 128
spectroWindowShift = 8
model.add(Spectrogram(input_shape=input_shape, n_dft=spectroWindowSize, n_hop=spectroWindowShift, padding='same',
                                 power_spectrogram=2.0, return_decibel_spectrogram=True))


#add model layers
#1
model.add(Conv2D(24, kernel_size=(12,12), input_shape=(64, 64, numberOfChannels), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(ReLU())
model.add(Dropout(rate=0.5))

#2
model.add(Conv2D(48, kernel_size=(8,8), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(ReLU())
model.add(Dropout(rate=0.5))

#3
model.add(Conv2D(96, kernel_size=(4,4), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(ReLU())
model.add(Dropout(rate=0.5))

#4
model.add(Flatten())
model.add(Dense(classes, activation='softmax'))

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy', 'mse'])

#
trainSplit = 0.8
batch = 256
epoch = 50
size = len(mX)
history = model.fit(mX, mY, validation_split=0.25, batch_size=batch, epochs=epoch)
print(history)
print(history.history)
model.summary()
model.save('models/tempModel2.h5')
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Now saved, let's load it.
model2 = load_model('models/tempModel.h5',
  custom_objects={'Spectrogram':Spectrogram})
#model2.summary()

rightPred = 0
wrongPred = 0
for i in range(int(size*0.99), int(size), 1):
    p = np.array([mX[i]])
    q = mY[i]
    pred1 = model2.predict_classes(p)
    prob1 = model2.predict(p)
    print("should predict ", q, " predicted", pred1, " Prob", prob1)
    index = 0
    for j in range(0, len(q), 1):
        if q[j] == 1:
            index = j
    if index == pred1[0]:
        rightPred += 1
    else:
        wrongPred += 1

print(rightPred)
print(wrongPred)
