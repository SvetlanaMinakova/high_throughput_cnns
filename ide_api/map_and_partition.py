from DSE.mapping.map_and_partition import map_and_partition
from dnn_builders.input_dnn_manager import load_or_build_dnn_for_analysis
from models.edge_platform.Architecture import get_jetson
from models.dnn_model.dnn import set_built_in
from models.dnn_model.transformation.ops_fusion import fuse_built_in
from util import get_project_root
import os


def run_single_onnx():
    # create target platform model
    architecture = get_jetson()

    # read DNN
    dnn_path = "/home/svetlana/ONNX/OnnxZooModels/yolov2.onnx"
    dnn = load_or_build_dnn_for_analysis(dnn_path)
    dnn.name = "yolov2"

    # optimize dnn: fuse operators
    built_in_ops = ["activation", "normalization", "arithmetic", "skip"]
    set_built_in(dnn, built_in_ops)
    fuse_built_in(dnn)

    # print DNN
    # dnn.print_details()

    inp_files_directory = str(os.path.join(str(get_project_root()), "input_examples/DSE/" + dnn.name))
    outp_files_directory = str(os.path.join(str(get_project_root()), "output/" + dnn.name))

    # EXAMPLE 1: greedy mapping with FLOPS-based dnn throughput evaluation
    # mapping = map_and_partition(dnn, architecture,
    # output_dir=outp_files_directory, eval_type="flops", map_algo="greedy")

    # EXAMPLE 2: greedy mapping with throughput evaluation based on measurements on the platform
    """
    measurements_path = str(os.path.join(inp_files_directory, "eval_template.json"))
    mapping = map_and_partition(dnn, architecture, output_dir=outp_files_directory,
                                eval_type="measurements", eval_path=measurements_path, map_algo="greedy")
    """

    # EXAMPLE 3: GA-based mapping with FLOPS-based dnn throughput evaluation
    # ga_config_path = str(os.path.join(inp_files_directory, "ga_conf.json"))
    # mapping = map_and_partition(dnn, architecture,
    # output_dir=outp_files_directory, eval_type="flops", map_algo="ga", ga_conf_path=ga_config_path)

    # EXAMPLE 4: GA-based mapping with throughput evaluation based on measurements on the platform

    measurements_path = str(os.path.join(inp_files_directory, "eval_template.json"))
    ga_config_path = str(os.path.join(inp_files_directory, "ga_conf.json"))
    mapping = map_and_partition(dnn, architecture, output_dir=outp_files_directory, 
                                eval_type="measurements", eval_path=measurements_path,
                                map_algo="ga", ga_conf_path=ga_config_path)


run_single_onnx()