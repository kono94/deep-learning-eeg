import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack, signal, stats
from skimage import util
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPool2D, ReLU, Dropout
from kapre.time_frequency import Spectrogram
from numpy.random import seed
seed(1)

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


channelsToUse = [2,3,4]
numberOfChannels = len(channelsToUse)
classes = 3
X = np.loadtxt("data/2019-10-16-17-30_MaximilianWernerMitHand.csv", usecols=channelsToUse, delimiter=",", skiprows=1)
# get label
Y = np.loadtxt("data/2019-10-16-17-30_MaximilianWernerMitHand.csv", usecols=[8],dtype=np.str, delimiter=",", skiprows=1)
# split data to corresponding label
print(Y)
fullX = []
for i in range(0, numberOfChannels, 1):
    l = []
    for i in range(0, classes, 1):
        k = []
        l.append(k)
    fullX.append(l)

for i in range(0, len(X), 1):
    classIndex = 0
    if Y[i] == "none":
       classIndex = 0
    elif Y[i] == "left":
       classIndex = 1
    elif Y[i] == "right":
       classIndex = 2
    for c in range(0, len(fullX), 1):
        fullX[c][classIndex].append(X[i][c])

for c in range(0, len(fullX), 1):
    for i in range(0, classes, 1):
        data = fullX[c][i]
        N = len(data)
        fs = 250
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
        y = six_sigma_clipping(y, 6, 6)
        # normalize each session and eletrode
        y = normalize(y)
        fullX[c][i] = y


# data augmentation, generating crops with windows size 512 and shift by 8 samples
miniumSampleSize = len(fullX[0][0])
for i in range(0, classes, 1):
    if(miniumSampleSize > len(fullX[0][i])):
        miniumSampleSize = len(fullX[0][i])

windowShift = 8
windowSize = 512
dataSetSize = int((miniumSampleSize - windowSize)/windowShift + 1)
mX = np.empty(shape=(dataSetSize, numberOfChannels, windowSize))
mY = np.zeros(shape=(dataSetSize, classes))
print(dataSetSize)
print(miniumSampleSize)
# [0] - [0]  0,4 0,3 0,4
#       [1]  0,3 0,3 0,3
#       [2]  0,2 0,2 0,2
# [1] - [0]  0,1 0,2 ,2 
#       [1]
#       [2]

for i in range(0, dataSetSize,  classes):
    for w in range(0, windowSize, 1):
        for ch in range(0, numberOfChannels, 1):
            for cl in range(0, classes, 1):
                mX[i + cl][ch][w] = fullX[ch][cl][i * windowShift + w]
                mY[i + cl] = [0, 0, 0]
                mY[i + cl][cl] = 1

plt.xlabel("Sample")
plt.ylabel("Voltage in mV")
plt.plot(mX[0][0])
plt.show()

# generate spectogram
spectroWindowSize = 128
spectroStep = 8
freqs, times, Sx = signal.spectrogram(mX[0][0], fs=fs, window='hanning', nperseg=spectroWindowSize,
			 noverlap=spectroWindowSize - spectroStep, detrend=False, scaling='spectrum')


f, ax = plt.subplots(figsize=(10, 5))
ax.pcolormesh(times, freqs, np.log10(Sx), cmap='viridis')
ax.set_ylabel('Frequency in Hz')
ax.set_xlabel('Time in s')
plt.show()

#create model
model = Sequential()
# 2-sec input
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
model.add(Dropout(rate=0.2))

#2
model.add(Conv2D(48, kernel_size=(8,8), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(ReLU())
model.add(Dropout(rate=0.2))

#3
model.add(Conv2D(96, kernel_size=(4,4), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(ReLU())
model.add(Dropout(rate=0.2))

#4
model.add(Flatten())
model.add(Dense(classes, activation='softmax'))

model.compile('adam', 'categorical_crossentropy')

# train it with raw audio sample inputs
rng_state = np.random.get_state()
np.random.shuffle(mX)
np.random.set_state(rng_state)
np.random.shuffle(mY)

trainSplit = 0.8
batch = 128
epoch = 20
model.fit(mX[0:int(dataSetSize*0.8)], mY[0:int(dataSetSize*0.8)], batch_size=batch, epochs=epoch, validation_data=(mX[int(dataSetSize*0.8):int(dataSetSize-1)], mY[int(dataSetSize*0.8):int(dataSetSize-1)]))

model.summary()
model.save('temp_model.h5')

# Now saved, let's load it.
model2 = load_model('temp_model.h5',
  custom_objects={'Spectrogram':Spectrogram})
model2.summary()

for i in range(int(dataSetSize*0.8), int(dataSetSize), 1):
    p = np.array([mX[i]])
    q = mY[i]
    pred1 = model.predict_classes(p)
    prob1 = model.predict(p)
    print("should predict ", q, " predicted", pred1, " Prob", prob1)