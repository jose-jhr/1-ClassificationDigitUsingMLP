import tensorflow as tf
import matplotlib
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

model = load_model('model/model.h5')

# code get data
(X_train_all, Y_train_all), (x_test, y_test) = mnist.load_data()

# number imagen to convert (50000,784)

x_test = x_test.reshape((x_test.shape[0], 28 * 28))
x_test = x_test.astype("float32") / 255
y_test  = to_categorical(y_test)


# predice data xtest
predictions = model.predict(x_test)
index = 0
print(f"Number digit truth", y_test[index])
print('\n')
print(f"Predictions for each Class: \n")
for i in range(10):
    print("digit", i, ' probability: ', predictions[index][i])

predictions = model.predict(x_test)
predicted_labels = [np.argmax(i) for i in predictions]
# convert one-hot enoded label to integers
y_test_integer_labels = tf.argmax(y_test, axis=1)
# generate matrix confusion for the dataset
cm = tf.math.confusion_matrix(labels=y_test_integer_labels, predictions=predicted_labels)
# plot confusion matrix
plt.figure(figsize=[15, 8])
import seaborn as sn

sn.heatmap(cm, annot=True, fmt='d', annot_kws={'size': 14})
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


