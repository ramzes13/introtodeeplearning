from ..generic_conf import *

# MLP for Pima Indians Dataset saved to single file
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

__location__ = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__), 'data'
))

# load pima indians dataset
datasetPath = os.path.join(__location__, 'pima-indians-diabetes.data.csv')
modelPath = os.path.join(__location__, 'pima-indians-diabetes.model.h5')

dataset = loadtxt(datasetPath, delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
# define model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# save model and architecture to single file
model.save(modelPath)
print("Saved model to disk")
