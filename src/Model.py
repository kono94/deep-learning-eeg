import numpy as np
import sys
import time
import librosa.core.time_frequency
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPool2D, ReLU, Dropout, Lambda
from kapre.time_frequency import Spectrogram
import matplotlib.pyplot as plt

def createMasterCNN(numberOfChannels, frameSize, spectroWindowSize, spectroWindowShift, numberOfClasses):
    input_shape = (numberOfChannels, frameSize)
    #create model
    model = Sequential()
    model.add(Spectrogram(input_shape=input_shape, n_dft=spectroWindowSize, n_hop=spectroWindowShift, padding='same',
                                    power_spectrogram=2.0, return_decibel_spectrogram=True))

    #add model layers
    #1
    # input_shape=(64, 64, numberOfChannels),
    model.add(Conv2D(24, kernel_size=(12,12), activation='relu'))
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
    model.add(Dense(numberOfClasses, activation='softmax'))

    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy', 'mse'])
    return model


def plotAndSaveHistory(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    currentTimestamp = str(int(round(time.time())))
    plt.savefig('accucary' + currentTimestamp + '.png')

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig('loss' + currentTimestamp + '.png')

def loadModel(path):
     print("Loading model from path: ", path)
     model = load_model('models/tempModel2.h5',
     custom_objects={'Spectrogram':Spectrogram})
     print("Successfull loaded model in!")
     model.summary()
     return model

def predict(model, X, Y, amount=0.1, verbose=1):
    if(amount > 1):
        raise ValueError("'amount' has to be between 0 and 1")

    maxAmount = int(len(X) * amount)
    rightPred = 0
    wrongPred = 0
    for i in range(0, maxAmount):
        sys.stdout.write("Predicting %d/%d\r" % (i,maxAmount))
        sys.stdout.flush()

        p = np.array([X[i]])
        q = Y[i]
        pred1 = model.predict_classes(p)
        prob1 = model.predict(p)
        if verbose == 1:
            print("should predict ", q, " predicted", pred1, " Prob", prob1)
        index = 0
        for j in range(0, len(q)):
            if q[j] == 1:
                index = j
        if index == pred1[0]:
            rightPred += 1
        else:
            wrongPred += 1

    print("Correct Predictions: ", rightPred)
    print("Wrong Predictions: ", wrongPred)
    print("Accuracy: ", rightPred/(rightPred + wrongPred))