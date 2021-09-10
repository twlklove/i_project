import tensorflow as tf
import numpy as np
#np.set_printoptions(threshold=np.inf)

mnist = tf.keras.datasets.mnist
print('begin to load data: ')
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train[0:2], y_train[0:2])

exit()

print('begin to construct model')
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

#predictions = model(x_train[:1]).numpy() # vector if logits
#probabilities = tf.nn.softmax(predictions).numpy()
#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
'''
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
'''

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print('begin to train: ')
model.fit(x_train, y_train, epochs=5)

print('begin to evaluate: ')
model.evaluate(x_test, y_test, verbose=2)