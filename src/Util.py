import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

def visualizeSpectrogram(mX, spectroWindowSize, spectroWindowShift, fs, nrOfSpectrogram=1):
    for i in range(0, nrOfSpectrogram):
        plt.xlabel("Sample")
        plt.ylabel("Voltage in mV")
        plt.plot(mX[i][0])
        plt.show()
        print(np.shape(mX))
        freqs, times, Sx = signal.spectrogram(mX[i][0], fs=fs, window='hanning', nperseg=spectroWindowSize,
                                            noverlap=spectroWindowSize - spectroWindowShift, detrend=False, scaling='spectrum')


        f, ax = plt.subplots(figsize=(10, 5))
        ax.pcolormesh(times, freqs, np.log10(Sx), cmap='viridis')
        ax.set_ylabel('Frequency in Hz')
        ax.set_xlabel('Time in s')
        plt.show()