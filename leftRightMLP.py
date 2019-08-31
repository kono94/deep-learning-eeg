# MULTI LAYER PERCEPTIONS
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 9
np.random.seed(seed)

dataset = np.loadtxt("data/blueRedMovement.csv", delimiter=",", skiprows=1)
# split into input (X) and output (Y) variables
number_of_channels = 3
X = dataset[:, 5:8]
Y = dataset[:, 8]
SampleRange = 5
matrix = []
labels = []

i = 0
while i < len(X) - SampleRange:
    frame = []
    for j in range(0, SampleRange):
        frame.append(X[i + j])
    matrix.append(frame)
    i += SampleRange
    labels.append(Y[i])

X = np.array(matrix)
Y = np.array(labels)

#X = np.expand_dims(X, axis=3)
#X = np.reshape(X.shape[0], 1, 8, SampleRange)

# standardizing the input feature
sc = StandardScaler()
# X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_train.shape[1]
print(num_classes)

X_train = np.reshape(X_train, [-1, number_of_channels*SampleRange])
X_test = np.reshape(X_test, [-1, number_of_channels*SampleRange])

classifier = Sequential()
classifier.add(Dense(32, input_shape=(number_of_channels*SampleRange,)))
classifier.add(Dense(num_classes, activation='softmax'))
classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# Fitting the data to the training dataset
classifier.fit(X_train, y_train, batch_size=32, epochs=6, validation_data=(X_test, y_test))

# Final evaluation of the model
eval_model = classifier.evaluate(X_train, y_train)
print("Baseline Error: %.2f%%" % (100 - eval_model[1] * 100))
print(eval_model[0] * 100)
print(eval_model[1] * 100)

# Save the model
# serialize model to JSON
model_digit_json = classifier.to_json()
with open("models/MLP_model_leftRight.json", "w") as json_file:
    json_file.write(model_digit_json)
# serialize weights to HDF5
classifier.save('models/MLP_model_leftRight.h5')
print("Saved model to disk")
