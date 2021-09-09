import tensorflow as tf

def dump_info(name, train, test):
    print(name, train.shape, 'train=>', len(train), 'test=>', len(test))

#mnist
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
dump_info('mnist : ', x_train, y_test)

#fashion_mnist
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
dump_info('fashion_mnist : ', train_images, test_labels)

#cifar10
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
dump_info('cifar10 : ', train_images, test_labels)


#image classification
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
print(data_dir)
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))

roses = list(data_dir.glob('roses/*'))
#PIL.Image.open(str(roses[0]))

tulips = list(data_dir.glob('tulips/*'))
#PIL.Image.open(str(tulips[0]))

import cv2
img = cv2.imread(str(roses[0]))
print('flower : ', img.shape, 'train & test=>', image_count)


