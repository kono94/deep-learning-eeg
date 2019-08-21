# MULTI LAYER PERCEPTIONS
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# fix random seed for reproducibility
seed = 9
np.random.seed(seed)

dataset = np.loadtxt("data/colorsReLabeled.csv", delimiter=",", skiprows=1)
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
# standardizing the input feature
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

classifier = Sequential()
# First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=8))
# Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
# Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
# Compiling the neural network
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the data to the training dataset
classifier.fit(X_train, y_train, batch_size=64, epochs=10000, validation_data=(X_test, y_test))

# Final evaluation of the model
eval_model = classifier.evaluate(X_train, y_train)
print("Baseline Error: %.2f%%" % (100 - eval_model[1] * 100))

# Save the model
# serialize model to JSON
model_digit_json = classifier.to_json()
with open("models/MLP_model_color.json", "w") as json_file:
    json_file.write(model_digit_json)
# serialize weights to HDF5
classifier.save('models/MLP_model_color.h5')
print("Saved model to disk")
