import numpy as np
import mne
import matplotlib.pyplot as plt

# Read the CSV file as a NumPy array
from sklearn.preprocessing import StandardScaler

data = np.loadtxt("data/blueRedMovement.csv", delimiter=",", skiprows=1)
X = data[0:200, 0:8]
k = []
for i in X:
    k.append(i[2])
plt.plot(k)
plt.show()
print(np.shape(X))
data = np.reshape(X, (8, 200))

print(np.shape(data))

print(data)
# Some information about the channels
ch_names = ['CH 1', 'CH 2', 'CH 3', 'CH 4', 'CH 5', 'CH 6', 'CH 7', 'CH 8']  # TODO: finish this list
ch_types = ['eeg', 'eeg', 'eeg', 'eeg','eeg', 'eeg', 'eeg', 'eeg']
# Sampling rate of the Nautilus machine
sfreq = 250  # Hz

# Create the info structure needed by MNE
info = mne.create_info(ch_names, sfreq, ch_types)

# Finally, create the Raw object
raw = mne.io.RawArray(data, info)

# Plot it!
# It is also possible to auto-compute scalings
scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
raw.plot(n_channels=8, scalings=scalings, title='Auto-scaled Data from arrays',
         show=True, block=True)

ica = mne.preprocessing.ICA(n_components=8, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [1, 2]  # details on how we picked these are omitted here
ica.plot_properties(raw, picks=ica.exclude)
orig_raw = raw.copy()
raw.load_data()
ica.apply(raw)
# show some frontal channels to clearly illustrate the artifact removal
chs = ['CH 1', 'CH 2', 'CH 3', 'CH 4', 'CH 5', 'CH 6', 'CH 7', 'CH 8']
chan_idxs = [raw.ch_names.index(ch) for ch in chs]
orig_raw.plot(order=chan_idxs, start=3, duration=4)
raw.plot(order=chan_idxs, start=3, duration=4)