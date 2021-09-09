import os
import gzip
import numpy as np
from log import *
from path import *
import tensorflow as tf
import struct
def load_data_from_npz(path):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)


def load_data_from_idx(base_path):
    dirname='cache'
    files = [
        'train_label_idx1_ubyte.gz', 'train_data_idx3_ubyte.gz', 
        'test_label_idx1_ubyte.gz', 'test_data_idx3_ubyte.gz'
    ]
    
    paths = []
    for fname in files:
        paths.append(os.path.join(base_path, fname))
    
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    
    with gzip.open(paths[1], 'rb') as imgpath:
        data = imgpath.read()
        fmt_header = '>4i' 
        offset = 0
        magic, image_num, num_rows, num_cols = struct.unpack_from(fmt_header, data, offset) 
        x_train = np.frombuffer(data, np.uint8, offset=16).reshape(image_num, num_rows, num_cols)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    
    with gzip.open(paths[3], 'rb') as imgpath:
        data = imgpath.read()
        fmt_header = '>4i' 
        offset = 0
        magic, image_num, num_rows, num_cols = struct.unpack_from(fmt_header, data, offset) 
        x_test = np.frombuffer(data, np.uint8, offset=16).reshape(image_num, num_rows, num_cols)
    
    return (x_train, y_train), (x_test, y_test)


from tensorflow.keras import datasets, layers, models
def do_train(x_train, y_train, x_test, y_test) :
    log_i('train : ', len(y_train),  x_train.shape)
    log_i('test :', len(y_test),  x_test.shape)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) 
    x_train, x_test = x_train / 255.0, x_test / 255.0

    kind = len(np.unique(y_test))
    count = len(y_train)
    epoch_num = 50
    if count > 5000 :
        epoch_num = 5
    elif count > 1000 :
        epoch_num = 10
     
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(kind))
    model.add(layers.Activation('softmax'))
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', #'categorical_crossentropy',
              metrics=['accuracy'])

    print('begin to train: ')
    model.fit(x_train, y_train, epochs=epoch_num)

    print('begin to evaluate: ')
    model.evaluate(x_test, y_test, verbose=2)

def do_train_flower():
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import PIL
    import tensorflow as tf
    
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    import pathlib

    data_dir='/root/.keras/datasets/flower_photos/'
    data_dir = pathlib.Path(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))

    batch_size = 32
    img_height = 180
    img_width = 180
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
              data_dir,
              validation_split=0.2,
              subset="training",
              seed=123,
              image_size=(img_height, img_width),
              batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
              data_dir,
              validation_split=0.2,
              subset="validation",
              seed=123,
              image_size=(img_height, img_width),
              batch_size=batch_size)
    class_names = train_ds.class_names
    print(class_names) 
   
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    data_augmentation = keras.Sequential(
      [
        layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                     input_shape=(img_height, 
                                                                  img_width,
                                                                  3)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
      ]
    )
    num_classes = 5
    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.summary()

    epochs=10
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(epochs)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
    
    img = keras.preprocessing.image.load_img(
        sunflower_path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
 

if __name__ == '__main__':
    #mnist = tf.keras.datasets.mnist
    #(x_train, y_train), (x_test, y_test) = mnist.load_data() # npz

    #base_path = os.path.join(base_path,'dst')
    #(x_train, y_train), (x_test, y_test) = load_data_from_idx(base_path) # idx 
    #do_train(x_train, y_train, x_test, y_test)
    do_train_flower()        
 
