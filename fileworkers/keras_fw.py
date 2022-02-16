from tensorflow import keras


def save_keras_model(model, path):
    model.save(path, save_format='h5')


def load_keras_model(path):
    model = keras.models.load_model(path)
    return model
