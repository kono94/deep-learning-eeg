import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

def visualizeSpectrogram(classPuffer, spectroWindowSize, spectroWindowShift, fs, nrOfSpectrogram=1):
    for i in range(0, nrOfSpectrogram):
        plt.xlabel("Sample")
        plt.ylabel("Voltage in mV")
        plt.plot(classPuffer[0][i][0])
        plt.show()

        freqs, times, Sx = signal.spectrogram(np.asarray(classPuffer[0][i][0]), fs=fs, window='hanning', nperseg=spectroWindowSize,
                                            noverlap=spectroWindowSize - spectroWindowShift, detrend=False, scaling='spectrum')


        f, ax = plt.subplots(figsize=(10, 5))
        ax.pcolormesh(times, freqs, np.log10(Sx), cmap='viridis')
        ax.set_ylabel('Frequency in Hz')
        ax.set_xlabel('Time in s')
        plt.show()