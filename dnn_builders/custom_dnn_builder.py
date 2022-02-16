"""
This module builds small custom DNNs, mostly for testing purposes
"""


def supported_custom_dnns():
    return ['tiny_dnn', 'small_dnn']


def create_custom_dnn_model(model_name, input_shape=[1, 28, 28], classes=10):
    """
    Build a custom dnn model
    Args:
        model_name: model name
        input_shape: input_examples data shape
        classes: number of classed (ofm in output Dense layer)

    Returns: dnn model

    """
    input_shape = [1, 28, 28] if input_shape is None else input_shape

    if model_name == "tiny_dnn":
        return build_tiny_keras_dnn(input_shape, classes)
    if model_name == "small_dnn":
        return build_small_keras_dnn(input_shape, classes)
    raise Exception("unknown custom model: " + model_name)


def build_tiny_keras_dnn(input_shape, num_classes):
    import tensorflow.keras as keras

    model = keras.Sequential(
        [
            # keras.layers.InputLayer(input_shape=input_shape, name="input_examples"),
            keras.layers.Conv2D(32, input_shape=input_shape, name="input_examples", kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def build_small_keras_dnn(input_shape, num_classes):
    import tensorflow.keras as keras
    # small keras dnn: Model 2 from blog
    # https://www.oscarjavierhernandez.com/other/2018/08/08/EDS3_summerschool_project.html#references
    model = keras.Sequential(
        [
            # keras.layers.InputLayer(input_shape=input_shape, name="input_examples"),
            keras.layers.Conv2D(32, input_shape=input_shape, name="input_examples", kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.5),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.5),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.5),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model

