from scipy import signal, stats
import numpy as np

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

def sigma_clipping(data, lowS, highS):
    y, low, upp = stats.sigmaclip(data, lowS, highS)
    return y

def normalize(data):
    mean = np.mean(data)
    sd = np.std(data)
    y = []
    for i in range(0, len(data), 1):
        y.append((data[i] - mean) / sd)
    return np.array(y)