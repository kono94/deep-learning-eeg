from keras.models import model_from_json
import numpy as np
from sklearn.preprocessing import StandardScaler

# load json and create model
json_file = open('models/MLP_model_color.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("models/MLP_model_color.h5")
print("Loaded model from disk")

# make a prediction
dataset = np.loadtxt("data/colorsRelabeled.csv", delimiter=",", skiprows=1)
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

sc = StandardScaler()
X = sc.fit_transform(X)
i = 0
while i < len(X):
    k = np.array([X[i]])
    pred1 = loaded_model.predict_classes(k)
    prob1 = loaded_model.predict(k)
    print("should predict ", Y[i], " predicted", pred1, " Prob", prob1)
    i += 1
