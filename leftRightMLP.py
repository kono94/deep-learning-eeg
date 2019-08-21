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

dataset = np.loadtxt("data/leftRight.csv", delimiter=",", skiprows=1)
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
SampleRange = 75
prev = Y[0]
matrix = []
labels = []
for i in range(0, len(X)-SampleRange, 1):
    if Y[i] != prev:
        labels.append(prev)
        prev = Y[i]
        frame = []
        for j in range(0, SampleRange):
            frame.append(X[i - SampleRange + j])
        matrix.append(np.column_stack(tuple(frame)))


X = np.array(matrix)
print(X.shape)
X = np.expand_dims(X, axis=3)
#X = np.reshape(X.shape[0], 1, 8, SampleRange)
print(X.shape)

# standardizing the input feature
sc = StandardScaler()
# X = sc.fit_transform(X)
Y = np.array(labels)
print(Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]
print(num_classes)
classifier = Sequential()
classifier.add(Conv2D(75, (5, 5), input_shape=(8, SampleRange, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(35, (5, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.2))
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(50, activation='relu'))
classifier.add(Dense(num_classes, activation='softmax'))

#classifier.add(Conv2D(64, (3, 3), input_shape=(8, SampleRange, 1)))
#classifier.add(Flatten())
# First Hidden Layer
#classifier.add(Dense(8, activation='relu', kernel_initializer='random_normal'))
# Second  Hidden Layer
#classifier.add(Dense(8, activation='relu', kernel_initializer='random_normal'))
# Output Layer
#classifier.add(Dense(num_classes, activation='softmax', kernel_initializer='random_normal'))
# Compiling the neural network
#classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting the data to the training dataset
classifier.fit(X_train, y_train, batch_size=64, epochs=20, validation_data=(X_test, y_test))

# Final evaluation of the model
eval_model = classifier.evaluate(X_train, y_train)
print("Baseline Error: %.2f%%" % (100 - eval_model[1] * 100))
print(eval_model[0])
print(eval_model[1])

# Save the model
# serialize model to JSON
model_digit_json = classifier.to_json()
with open("models/MLP_model_leftRight.json", "w") as json_file:
    json_file.write(model_digit_json)
# serialize weights to HDF5
classifier.save('models/MLP_model_leftRight.h5')
print("Saved model to disk")
