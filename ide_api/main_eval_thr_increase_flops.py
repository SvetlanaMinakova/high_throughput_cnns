from experiments.throughput_increase.ti_flops import increase_dnn_throughput
from dnn_builders.input_dnn_manager import load_or_build_dnn_for_analysis
from models.edge_platform.Architecture import get_jetson
import support_matrix
from fileworkers.onnx_fw import onnx_file_paths_in_directory
from util import get_project_root

""" 
 Estimate throughput increase, achieved by efficient mapping of 
 computations within a DNN onto computational resources of an edge platform
 Use FLOPS-based throughput evaluation
"""


def run_keras_dnns():
    architecture = get_jetson()
    dnn_names_list = support_matrix.supported_keras_models() # ["densenet121"]
    built_in_ops = ["activation", "normalization", "skip"]

    for dnn_name in dnn_names_list:
        print("///////////////////////////////////////////")
        print(dnn_name)
        dnn = load_or_build_dnn_for_analysis(dnn_name)
        # dnn.print_details()
        increase_dnn_throughput(dnn, architecture, built_in_ops, token_size=4)
        print("")


def run_onnx_folder():
    onnx_files_directory = "/home/svetlana/ONNX/OnnxZooModels"
    fp_list = onnx_file_paths_in_directory(onnx_files_directory)
    built_in_ops = ["activation", "normalization", "skip"]
    architecture = get_jetson()

    for dnn_path in fp_list:

        print("///////////////////////////////////////////")
        print(dnn_path)
        dnn = load_or_build_dnn_for_analysis(dnn_path)
        # dnn.print_details()
        increase_dnn_throughput(dnn, architecture, built_in_ops, token_size=4)
        print("")


def run_single_onnx():
    architecture = get_jetson()
    dnn_path = "/home/svetlana/ONNX/OnnxZooModels/yolov2.onnx"
    dnn = load_or_build_dnn_for_analysis(dnn_path)
    dnn.name = "yolo v2"
    # dnn.print_details()
    built_in_ops = ["activation", "normalization", "skip"]

    ga_conf_path = str(get_project_root()) + "/input_examples/DSE/ga_conf_generic.json"
    increase_dnn_throughput(dnn, architecture, built_in_ops, ga_conf_path, token_size=4)


run_single_onnx()