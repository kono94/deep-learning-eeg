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


dataset = np.loadtxt("data/leftRight.csv", delimiter=",", skiprows=1)
# Choose electrodes
X = dataset[:, 3:5]
# get label
Y = dataset[:, 8]

# split data to corresponding label
X0 = []
X1 = []
X2 = []
X0b = []
X1b = []
X2b = []

for i in range(0, len(X), 1):
    if Y[i] == 1:
        X1.append(X[i][0])
        X1b.append(X[i][1])
    elif Y[i] == 2:
        X2.append(X[i][0])
        X2b.append(X[i][1])
    elif Y[i] == 0:
        X0.append(X[i][0])
        X0b.append(X[i][1])

X_list = []
X_list.append(X0)
X_list.append(X1)
X_list.append(X2)
X_list.append(X0b)
X_list.append(X1b)
X_list.append(X2b)

X_list_filtered = []
for i in range(0, len(X_list), 1):
    data = X_list[i]
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
    X_list_filtered.append(y)


# data augmentation, generating crops with windows size 512 and shift by 8 samples
mX = np.empty((int((26*512)/8*3), 2, 512))
mY = np.zeros(shape=(int((26*512)/8*3), 3)) 

k = 0
for i in range(0, 26*512-1, 8):
    for j in range(0, 512, 1):
        mX[k][0][j] = X_list_filtered[0][i+j]
        mX[k+1][0][j] = X_list_filtered[1][i+j]
        mX[k+2][0][j] = X_list_filtered[2][i+j]

        mX[k][1][j] = X_list_filtered[3][i+j]
        mX[k+1][1][j] = X_list_filtered[4][i+j]
        mX[k+2][1][j] = X_list_filtered[5][i+j]

        mY[k] = [1,0,0]
        mY[k+1] = [0,1,0]
        mY[k+2] = [0,0,1]
    k += 3

plt.xlabel("Sample")
plt.ylabel("Voltage in mV")
plt.plot(mX[0][0])
plt.show()

# generate spectogram
windowSize = 128
step = 8
freqs, times, Sx = signal.spectrogram(mX[0][0], fs=fs, window='hanning', nperseg=windowSize,
			 noverlap=windowSize - step, detrend=False, scaling='spectrum')


f, ax = plt.subplots(figsize=(10, 5))
ax.pcolormesh(times, freqs, np.log10(Sx), cmap='viridis')
ax.set_ylabel('Frequency in Hz')
ax.set_xlabel('Time in s')
plt.show()

#create model
model = Sequential()
# 3 channels, 4-sec input
numberOfChannels = 2
classes = 3
input_shape = (numberOfChannels, 512)

model.add(Spectrogram(input_shape=input_shape, n_dft=128, n_hop=8, padding='same',
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

model.compile('adam', 'categorical_crossentropy')

# train it with raw audio sample inputs
rng_state = np.random.get_state()
np.random.shuffle(mX)
np.random.set_state(rng_state)
np.random.shuffle(mY)
model.fit(mX[0:3000], mY[0:3000], batch_size=256, epochs=10, validation_data=(mX[3000:4000], mY[3000:4000]))

model.summary()
model.save('temp_model.h5')

# Now saved, let's load it.
model2 = load_model('temp_model.h5',
  custom_objects={'Spectrogram':Spectrogram})
model2.summary()

p = np.array([mX[4200]])
q = mY[4200]
pred1 = model2.predict_classes(p)
prob1 = model2.predict(p)
print("should predict ", q, " predicted", pred1, " Prob", prob1)

p = np.array([mX[4300]])
q = mY[4300]
pred1 = model2.predict_classes(p)
prob1 = model2.predict(p)
print("should predict ", q, " predicted", pred1, " Prob", prob1)


p = np.array([mX[4400]])
q = mY[4400]
pred1 = model2.predict_classes(p)
prob1 = model2.predict(p)
print("should predict ", q, " predicted", pred1, " Prob", prob1)


p = np.array([mX[4403]])
q = mY[4403]
pred1 = model2.predict_classes(p)
prob1 = model2.predict(p)
print("should predict ", q, " predicted", pred1, " Prob", prob1)


p = np.array([mX[4406]])
q = mY[4406]
pred1 = model2.predict_classes(p)
prob1 = model2.predict(p)
print("should predict ", q, " predicted", pred1, " Prob", prob1)

for i in range(4000, 4100, 1):
    p = np.array([mX[i]])
    q = mY[i]
    pred1 = model.predict_classes(p)
    prob1 = model.predict(p)
    print("should predict ", q, " predicted", pred1, " Prob", prob1)
