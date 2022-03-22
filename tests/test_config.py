import os
from util import get_project_root


def get_test_config():
    # path to input files
    input_files_folder_abs = os.path.join(get_project_root(), "input_examples")

    # path to input cnn
    input_cnn_path = str(os.path.join(input_files_folder_abs, "dnn", "mnist.onnx"))

    # path to input platform architecture
    input_platform_path = str(os.path.join(input_files_folder_abs, "architecture", "jetson.json"))

    # path to intermediate files, produced by steps of the tool during the tests
    intermediate_files_folder_abs = os.path.join(get_project_root(), "output", "mnist")

    test_config = {
        "input_files_folder_abs": input_files_folder_abs,
        "cnn": input_cnn_path,
        "platform": input_platform_path,
        "intermediate_files_folder_abs": intermediate_files_folder_abs
    }
    return test_config

