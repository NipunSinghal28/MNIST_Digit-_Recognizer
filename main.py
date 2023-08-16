import numpy as np
import pandas as pd
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt


# DATA LOADING
data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build and train the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# TRAIN THE MODEL
model.fit(x_train, y_train, epochs=3)

# Evaluate the model on test data
loss, accuracy = model.evaluate(x_test, y_test)
print('model accuracy rate on data set', accuracy)
print('model loss on test data', loss)


# Load and preprocess the image for prediction
img_path = ('0.png')
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (28, 28))
img = np.invert(img)
img = img/255.0
img = np.expand_dims(img, axis=0) # Add batch dimensions


# MAKE A PREDICTION
prediction = model.predict(img)
predict_label = np.argmax(prediction)

# Display the image and predicted label
plt.imshow(np.squeeze(img), cmap=plt.cm.gray)
plt.title('Predicted label: {}'.format(predict_label))
plt.show()


