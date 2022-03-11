import onnx
import os


def read_onnx(path):
    """
    Load ONNX model from file
    :param path: path to ONNX model
    :return: onnx model
    """
    onnx_model = onnx.load(path)
    return onnx_model


def save_keras_as_onnx(keras_model, path):
    """
    Save keras model in ONNX format
    :param keras_model: keras model
    :param path: path to output ONNX model
    """
    import keras2onnx
    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)
    keras2onnx.save_model(onnx_model, path)


def onnx_file_paths_in_directory(directory):
    """
    Find all file paths, leading to DNN in ONNX format in given directory
    :param directory: directory
    :return: list of  file paths, leading to DNN in ONNX format in given directory
    """
    onnx_file_paths = []
    if os.path.exists(directory):
        for the_file in os.listdir(directory):
            file_path = os.path.join(directory, the_file)
            if os.path.isfile(file_path):
                if str(file_path).endswith(".onnx"):
                    onnx_file_paths.append(str(file_path))
        return onnx_file_paths
    else:
        raise Exception("Directory " + directory + "does not exist")


def dnn_name_from_onnx_path(onnx_path):
    """
    Extract dnn name from onnx path
    :return: dnn name, extracted from onnx path
    """
    onnx_path_parts = onnx_path.replace(".onnx", "").split("/")
    return onnx_path_parts[-1].replace(" ", "")

