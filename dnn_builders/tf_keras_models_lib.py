from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import models, layers
from tensorflow.keras import regularizers
import tensorflow as tf


def test_model(verbose=True):
    model = Sequential()
    model.add(layers.Dropout(0.2, input_shape=(784,)))
    model.add(layers.Dense(1000,
                           kernel_regularizer=regularizers.l2(0.01),
                           activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1000,
                           kernel_regularizer=regularizers.l2(0.01),
                           activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10,  activation='softmax'))
    if verbose:
        print("test CNN built")
        # display the model summary
        model.summary()
    return model


def test_model_conv(verbose=False):
    model = Sequential()
    input_shape = [28, 28, 1]

    model.add(layers.Conv2D(filters=8,
                            kernel_size=(3, 3),
                            activation='relu',
                            input_shape=input_shape))

    model.add(layers.Conv2D(filters=16,
                            kernel_size=(3, 3),
                            activation='relu',
                            input_shape=input_shape))

    model.add(layers.Flatten())

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(500,  activation='relu'))

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10,  activation='softmax'))
    if verbose:
        print("test CNN built")
        # display the model summary
        model.summary()
    return model


def resnet_50():
    model = tf.keras.applications.resnet50.ResNet50(
        include_top=True, weights='imagenet', input_tensor=None,
        pooling=None, classes=1000, input_shape=None,
    )
    return model


def mobilenet_v2():
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        include_top=True, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None, classes=1000
    )
    return model
