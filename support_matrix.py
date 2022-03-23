"""
Here is the information about supported DNN models, datasets, etc.
Check this before creating you config!
"""

from dnn_builders.keras_models_builder import supported_keras_models
from dnn_builders.custom_dnn_builder import supported_custom_dnns
from models.dnn_model.supported_ops import print_supported_dnn_ops
from dnn_builders.test_analysis_dnn_builder import supported_test_analysis_dnns
import os
import sys


def main():
    # import current directory and it's subdirectories into system path for the current console
    # this would allow importing project modules without adding the project to the PYTHONPATH
    this_dir = os.path.dirname(__file__)
    sys.path.append(this_dir)
    print_info()


def print_info():
    print("We support: ")
    print(" DNN operators (from ONNX v7+)")
    print_supported_dnn_ops()
    print("  models from keras/tensorflow library: ", supported_keras_models())
    print("")
    print("  tests hand-made dnns:")
    print("    -keras: ", supported_custom_dnns())
    print("    -analysis dnn model: ", supported_test_analysis_dnns())
    print("")
    print("  custom cnn models, presented in formats: ", supported_dnn_extensions())


def supported_dnn_extensions():
    return [".onnx", ".h5", ".json"]


if __name__ == "__main__":
    main()