import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np

image_path = "C:/Users/ThinkPad/Desktop/programming/deep learning/numbers/3.png"
image = Image.open(image_path)
gray_image = image.convert("L")
image.show()

im = np.array(gray_image)
reshape = im.reshape((1, 28, 28))

reshape = tf.keras.utils.normalize(reshape, axis = 1)


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)


model = models.Sequential([
    layers.Flatten(input_shape = (28, 28)),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')
    ])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 10)

test_loss, test_acc = model.evaluate(x_test, y_test)

predictions = model.predict(x_test)

predicted_labels = [tf.argmax(prediction).numpy() for prediction in predictions]



plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
print(x_train[0])