'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# load data and preprocess them
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_img, train_label), (test_img, test_label) = fashion_mnist.load_data()
train_img = train_img / 255.0
test_img = test_img / 255.0
print(train_img[0], train_img.shape, train_label[0])

plt.figure(figsize=(2, 3))
plt.subplot(1, 2, 1)
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.imshow(train_img[0], cmap=plt.cm.binary)
plt.xlabel(class_names[0], color='blue')
plt.colorbar()
plt.tight_layout()
plt.show()

#construct
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train
model.fit(train_img, train_label, epochs=5)

# evaluate
loss, accuracy = model.evaluate(test_img, test_label, verbose=2)
print(loss, accuracy)


# prediction
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

img = test_img[1]
img = (np.expand_dims(img,0))
print(img.shape)
probability = probability_model.predict(img)
index = np.argmax(probability[0])
print(probability[0], np.max(probability[0]), index, class_names[index])

plt.ylim([0, 2])
thisplot = plt.bar(range(10), probability[0], color="#777777")
thisplot[index].set_color('blue')
plt.show()
'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_img, train_label),(test_img, test_label) = fashion_mnist.load_data()
train_img = train_img / 255.0
test_img = test_img / 255.0
print(train_img.shape, test_img.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(test_img, test_label, epochs=5)

loss, accuracy = model.evaluate(test_img, test_label, verbose=2)
print(loss, accuracy)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
prediction = probability_model.predict(test_img)
print(prediction[0], np.max(prediction[0]), np.argmax(prediction[0]), test_label[0])


#show
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
index = np.argmax(prediction[0])
plt.xticks([])
plt.grid(False)
plt.imshow(test_img[index])

plt.subplot(1, 2, 2)
bar = plt.bar(range(10), prediction[0], color="#777777")
_ = plt.xticks(range(10), class_names, rotation=45)
bar[index].set_color('red')
plt.show()