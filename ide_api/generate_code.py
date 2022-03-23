import os
from util import get_project_root


def generate_mixed_code_mnist():
    # path to input files
    input_files_folder_abs = os.path.join(get_project_root(), "input_examples")
    # path to input cnn
    input_cnn_path = str(os.path.join(input_files_folder_abs, "dnn", "mnist.onnx"))
    # path to input platform architecture
    input_platform_path = str(os.path.join(input_files_folder_abs, "architecture", "jetson.json"))
    # path to intermediate files, produced by steps of the tool during the tests
    intermediate_files_folder_abs = os.path.join(get_project_root(), "output", "mnist")