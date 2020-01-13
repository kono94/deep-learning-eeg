from Filters import butter_bandpass_filter, notch_filter, butter_highpass_filter, normalize, sigma_clipping
import sys
import numpy as np

def applyFilters(data, fs):
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
    y = sigma_clipping(y, 6, 6)
    # normalize each session and eletrode
    y = normalize(y)
    return y

def preProcess(X, Y, numberOfChannels, numberOfClasses, cropWindowSize, cropWindowShift, fs):
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
            channelCrops[ch] = applyFilters(channelCrops[ch], fs)
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
   # xSwap = np.swapaxes(X,0,1)
   # for ch in range(0, numberOfChannels, 1):
   #     xSwap[ch] = applyFilters(xSwap[ch], fs)
	
   # X = np.swapaxes(xSwap,0,1)
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
    return mX, mY
