import pathlib
import os
import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

np.set_printoptions(threshold=np.inf)

def dump_img(data_dir):
    img_format = '*/*.jpg' #'roses/*'
    imgs = list(data_dir.glob(img_format))
    image_count = len(imgs)
    print(image_count)

    img = PIL.Image.open(str(imgs[0]))
    img.show()

def get_data(data_dir, img_size, batch_size=32):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        #color_mode='grayscale',
        image_size=img_size,
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        #color_mode='grayscale',
        image_size=img_size,
        batch_size=batch_size)
    return train_ds, val_ds

def prefetch_data(train_ds, val_ds):
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

'''
def normalize_data():
    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))
'''

def visualize_data(xx_ds, class_names, input_shape, need_augmentation=False) :
    data_augmentation = None
    if need_augmentation :
        data_augmentation = create_data_augmentation(input_shape)

    plt.figure(figsize=(10, 10))
    for images, labels in xx_ds.take(1):
        for i in range(9):
            if need_augmentation:
                images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

    for image_batch, labels_batch in xx_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

def create_data_augmentation(input_shape) :
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=input_shape),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )
    return data_augmentation

def create_model_ResNet50(model_dir, input_shape, num_classes):
    if os.path.exists(model_dir):
        model = tf.keras.models.load_model(model_dir)
        if model is not None:
            return model

    #from tensorflow.keras.applications.resnet import ResNet50
    #model = ResNet50(weights=None, input_shape=input_shape, classes=num_classes)
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2
    model = ResNet50V2(weights=None, input_shape=input_shape, classes=num_classes)
    return model
def compile_model_ResNet50(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    return model

def create_model(model_dir, input_shape, num_classes):
    if os.path.exists(model_dir):
        model = tf.keras.models.load_model(model_dir)
        if model is not None:
            return model

    data_augmentation = create_data_augmentation(input_shape)
    model = Sequential([
        data_augmentation,
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=input_shape),
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
    return model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    return model

def train_model(model, train_ds, val_ds, epochs):
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    return history

def visualize_training_results(history, epochs):
    epochs_range = range(epochs)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

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

def predict(model, class_names, img):
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

if __name__=='__main__':
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    #dump_img(data_dir)
    model_dir = 'saved_model/my_model_1' #'saved_model/my_model'
    #if not os.path.exists(model_dir):
        #os.mkdir(model_dir)

    epochs = 3
    batch_size = 32
    img_height = 224 #60 #180
    img_width = 224 #60#180
    img_channel = 3
    img_size = (img_height, img_width)
    input_shape = (img_height, img_width, img_channel)

    train_ds, val_ds = get_data(data_dir, img_size, batch_size)
    class_names = train_ds.class_names
    print(class_names)
    train_ds, val_ds = prefetch_data(train_ds, val_ds)
    visualize_data(train_ds, class_names, input_shape, need_augmentation=False)

    model = create_model(model_dir, input_shape, len(class_names))
    model = compile_model(model)
    #model = create_model_ResNet50(model_dir, input_shape, len(class_names))
    #model = compile_model_ResNet50(model)

    history = train_model(model, train_ds, val_ds, epochs)

    visualize_training_results(history, epochs)

    # Evaluate model, option
    #loss, acc = model.evaluate(val_ds, verbose=1) #test_images, test_labels,
    #print('accuracy: {:5.2f}%'.format(100 * acc))

    model.save(model_dir)  #save model

    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
    #img = PIL.Image.open(str(sunflower_path))
    #img.show()
    img = keras.preprocessing.image.load_img(sunflower_path, target_size=img_size)
    predict(model, class_names, img)