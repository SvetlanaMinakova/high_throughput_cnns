import os
from util import get_project_root


def get_test_config():
    # path to input files
    input_files_folder_abs = os.path.join(get_project_root(), "input_examples")

    # path to input cnn (onnx)
    input_cnn_path_onnx = str(os.path.join(input_files_folder_abs, "dnn", "mnist.onnx"))

    # path to input cnn (json)
    input_cnn_path_json = str(os.path.join(input_files_folder_abs, "dnn", "mnist.json"))

    # path to input platform architecture
    input_platform_path = str(os.path.join(input_files_folder_abs, "architecture", "jetson.json"))

    # path to intermediate files, produced by steps of the tool during the tests
    intermediate_files_folder_abs = os.path.join(get_project_root(), "output", "test")

    test_config = {
        "input_files_folder_abs": input_files_folder_abs,
        "cnn_json": input_cnn_path_json,
        "cnn_onnx": input_cnn_path_onnx,
        "platform": input_platform_path,
        "intermediate_files_folder_abs": intermediate_files_folder_abs
    }
    return test_config

