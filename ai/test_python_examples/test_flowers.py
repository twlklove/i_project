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
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB7
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
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
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        #color_mode='grayscale',
        image_size=img_size,
        batch_size= batch_size
    )
    return train_ds, val_ds

def prepare(ds, img_size=(224, 224), shuffle=False, augment=False):
    AUTOTUNE = tf.data.AUTOTUNE
    resize_and_rescale = tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(img_size[0], img_size[1]),
        layers.experimental.preprocessing.Rescaling(1. / 255)
    ])

    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ])

    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.cache().shuffle(1000)

    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)

def prefetch_data(train_ds, val_ds, img_size=(224, 224)):
    train_ds = prepare(train_ds, img_size, shuffle=True, augment=True)
    val_ds = prepare(val_ds, img_size)

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

def visualize_data(xx_ds, class_names, input_shape) :
    plt.figure(figsize=(10, 10))
    for images, labels in xx_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

    for image_batch, labels_batch in xx_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

def create_model(model_dir, input_shape, num_classes, model_name=''):
    if os.path.exists(model_dir):
        try:
            model = tf.keras.models.load_model(model_dir)
            if model is not None:
                return model
        except IOError:
            pass
    #from operator import methodcaller
    #model = methodcaller(model_name)()(weights=None, input_shape=input_shape, classes=num_classes)
    #f = getattr(model, 'fun_name')
    #locals()[InceptionResNetV2]()
    include_top = True #
    model = globals()[model_name](weights=None, input_shape=input_shape, classes=num_classes,include_top=include_top)
    if not include_top :
        output = model.output
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        output = layers.Dropout(0.1)(output)
        output = layers.GlobalAveragePooling2D(name='avg_pool')(output)
        output = layers.Dense(num_classes, activation='softmax', name='predictions')(output)
        model = Model(inputs=model.input, outputs=output)

    return model

def compile_model(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #'sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model

def I_NET(weights=None, input_shape=None, classes=1000,include_top=True):
    model = tf.keras.models.Sequential([
        layers.Input(input_shape),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D()
      ])

    if include_top :
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(classes))
        model.add(layers.Softmax())

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

config_model = 'EfficientNetB7' #'InceptionResNetV2' #'I_NET' #'MobileNetV2' #'MobileNetV2' #InceptionResNetV2' #'ResNet50V2' #'VGG16'
epochs = 3
batch_size = 32
img_height = 168 #224 #224   # 60 #180
img_width = 168 #224 #224   #60#180
img_channel = 3

if __name__=='__main__':
    #dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    #data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    #data_dir = pathlib.Path(data_dir)
    data_dir = 'E:/i_share/i_test4'
    #dump_img(data_dir)
    img_shape_str = str(img_height) + '_' + str(img_width ) + '_' + str(img_channel)
    model_dir = os.path.join('saved_model', config_model+'_' + img_shape_str + '_model')

    img_size = (img_height, img_width)
    input_shape = (img_height, img_width, img_channel)

    train_ds, val_ds = get_data(data_dir, img_size, batch_size)
    class_names = train_ds.class_names
    print(class_names)

    train_ds, val_ds = prefetch_data(train_ds, val_ds, img_size)
    print(len(train_ds))
    #visualize_data(train_ds, class_names, input_shape)

    model = create_model(model_dir, input_shape, len(class_names), model_name=config_model)
    model = compile_model(model)
    history = train_model(model, train_ds, val_ds, epochs)
    visualize_training_results(history, epochs)

    # Evaluate model, option
    #loss, acc = model.evaluate(val_ds, verbose=1) #test_images, test_labels,
    #print('accuracy: {:5.2f}%'.format(100 * acc))

    model.save(model_dir)  #save model

    #sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_url = 'E:/i_share/Red_sunflower.jpg'
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
    #img = PIL.Image.open(str(sunflower_path))
    #img.show()
    img = keras.preprocessing.image.load_img(sunflower_path, target_size=img_size)
    predict(model, class_names, img)