import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack, signal, stats
from skimage import util

dataset = np.loadtxt("data/leftRight.csv", delimiter=",", skiprows=1)
# Choose electrode
X = dataset[:, 3]
# get label
Y = dataset[:, 8]

# split data to corresponding label
X0 = []
X1 = []
X2 = []
for i in range(0, len(X), 1):
    if Y[i] == 1:
        X1.append(X[i])
    elif Y[i] == 2:
        X2.append(X[i])
    elif Y[i] == 0:
        X0.append(X[i])


# choose which label to visualize
# X = np.array(X1)

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

N = np.shape(X)[0]
fs = 250
L = N / fs

# remove 0.5 hz DC-offset
y = butter_highpass_filter(X, 0.5, fs)

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

plt.xlabel("Sample")
plt.ylabel("Voltage in mV")
plt.plot(y)
plt.show()

# generate spectogram
windowSize = 1024
step = 16
freqs, times, Sx = signal.spectrogram(y, fs=fs, window='hanning', nperseg=windowSize,
			 noverlap=windowSize - step, detrend=False, scaling='spectrum')


f, ax = plt.subplots(figsize=(10, 5))
ax.pcolormesh(times, freqs, 10*np.log10(Sx), cmap='viridis')
ax.set_ylabel('Frequency in Hz')
ax.set_xlabel('Time in s')
plt.show()