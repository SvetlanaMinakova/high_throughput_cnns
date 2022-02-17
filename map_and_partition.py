import os
import sys
import argparse
import traceback

from DSE.mapping.map_and_partition import map_and_partition
from dnn_builders.input_dnn_manager import load_or_build_dnn_for_analysis
from models.edge_platform.Architecture import get_jetson
from models.dnn_model.dnn import set_built_in
from models.dnn_model.transformation.ops_fusion import fuse_built_in
from util import get_project_root


def main():
    # import current directory and it's subdirectories into system path for the current console
    # this would allow to import project modules without adding the project to the PYTHONPATH
    this_dir = os.path.dirname(__file__)
    sys.path.append(this_dir)

    # import project modules
    from dnn_builders.input_dnn_manager import load_or_build_dnn_for_analysis
    from converters.dnn_to_task_graph import dnn_to_task_graph, dnn_to_task_graph_with_built_in
    from models.dnn_model.dnn import set_built_in
    from models.dnn_model.transformation.ops_fusion import fuse_built_in
    from util import get_project_root, print_stage
    from converters.json_converters.json_to_architecture import json_to_architecture
    from DSE.mapping.map_and_partition import map_and_partition

    # general arguments
    parser = argparse.ArgumentParser(description='The script maps an input CNN (in supported input format), '
                                                 'supplied with a JSON task-graph (SDF) model '
                                                 '(see ./dnn_to_sdf_task_graph.py), onto a target edge platform'
                                                 '(architecture). The mapping is saved in an output JSON file')

    parser.add_argument('--cnn', metavar='--cnn', type=str, action='store', required=True,
                        help='path to one or several CNNs. Can be a path to: '
                             '1) a path to an .onnx file; '
                             '2) a path to .h5 file (cnn in format of Keras DL framework). ')

    # platform is not needed for this conversion

    parser.add_argument('-p', metavar='--platform', type=str, action='store', required=True,
                        help='path to edge platform (architecture) description, saved in .json format')

    parser.add_argument('-tg', metavar='--task-graph', type=str, action='store', required=True,
                        help='path to .json file with the task-graph (SDF) model generated '
                             'for the input cnn by the ./dnn_to_sdf_task_graph.py script')

    parser.add_argument('-fo', metavar='--fused-ops', type=str, action='store',
                        default='activation,normalization,arithmetic,skip',
                        help='List built-in (fused) operators within a cnn. A cnn layers, performing a built-in'
                             ' operator is fused with another layer also referred as a "parent" layer. For example'
                             ' an activation (e.g. ReLU) layer, following a convolutional layer,'
                             ' can be fused with the convolutional layer. In this case, the activation layer is'
                             ' the fused layer, and the convolutional layer is the "parent" layer.'
                             ' Fused layers are always mapped onto the same processor'
                             ' as their "parent" layer. Also, fused layers do not have an intermediate'
                             ' output date (output buffer) of their own.'
                             ' NOTE: LIST OF FUSED OPS SHOULD NOT CHANGE BETWEEN THE SUBSEQUENT SCRIPTS')

    parser.add_argument('-map-algo', metavar='--map-algo', type=str, action='store', default="ga",
                        help='value in [greedy, ga]: algorithm used for mapping of (computations within) the CNN'
                             'onto (the processors of) the target edge platform')

    parser.add_argument('-e', metavar='--eval-path', type=str, action='store', default=None,
                        help='path to per-layer cnn execution time (latency) evaluation (JSON). '
                             'Use ./eval_template.py script to generate a template for this file.'
                             'If no eval-path is provided, execution time of every cnn layer is evaluated using '
                             'number of floating-point operations (FLOPS)')

    parser.add_argument('-ga-config', metavar='--ga-config', type=str, action='store', default=None,
                        help='path to path to GA config. Required for GA-based search (map_algo="ga")')

    parser.add_argument('-o', metavar='--out-dir', type=str, action='store', default="./output",
                        help='path to output files directory')

    # general flags
    parser.add_argument("--silent", help="do not provide print-out for the script steps", action="store_true", default=False)

    args = parser.parse_args()
    try:
        # Extract parameters from command-line arguments"
        cnn_path = args.cnn
        platform_path = args.p
        task_graph_path = args.tg
        map_algo = args.map_algo
        ga_config_path = args.ga_config

        eval_path = args.eval_path
        eval_type = "measurements"
        if eval_path is None:
            eval_type = "flops"

        output_dir = args.o
        silent = args.silent
        verbose = not silent
        fused_ops_spec = args.fo
        fused_ops = fused_ops_spec.split(',')

        # read DNN
        stage = "Reading input DNN"
        print_stage(stage, verbose)
        dnn = load_or_build_dnn_for_analysis(cnn_path, verbose=verbose)
        # print DNN
        # dnn.print_details()

        # optimize dnn: fuse operators
        if len(fused_ops) > 0:
            stage = "Optimize DNN: fuse layers that perform operators " + str(fused_ops_spec)
            print_stage(stage, verbose)
            set_built_in(dnn, fused_ops)
            fuse_built_in(dnn)

        # platform is not needed for this conversion

        stage = "Reading target platform architecture"
        print_stage(stage, verbose)
        architecture = json_to_architecture(platform_path)

        stage = "Mapping and partitioning"
        print_stage(stage, verbose)
        map_and_partition(dnn, architecture, output_dir,
                          task_graph_path=task_graph_path,
                          map_algo=map_algo,
                          eval_type=eval_type,
                          eval_path=eval_path,
                          ga_conf_path=ga_config_path,
                          verbose=verbose)

    except Exception as e:
        print(" Task Graph (SDF) model creation error: " + str(e))
        traceback.print_tb(e.__traceback__)


if __name__ == "__main__":
    main()


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
    measurements_path = str(os.path.join(inp_files_directory, "eval.json"))
    mapping = map_and_partition(dnn, architecture, output_dir=outp_files_directory,
                                eval_type="measurements", eval_path=measurements_path, map_algo="greedy")
    """

    # EXAMPLE 3: GA-based mapping with FLOPS-based dnn throughput evaluation
    # ga_config_path = str(os.path.join(inp_files_directory, "ga_conf.json"))
    # mapping = map_and_partition(dnn, architecture,
    # output_dir=outp_files_directory, eval_type="flops", map_algo="ga", ga_conf_path=ga_config_path)

    # EXAMPLE 4: GA-based mapping with throughput evaluation based on measurements on the platform

    measurements_path = str(os.path.join(inp_files_directory, "eval.json"))
    ga_config_path = str(os.path.join(inp_files_directory, "ga_conf.json"))
    mapping = map_and_partition(dnn, architecture, output_dir=outp_files_directory, 
                                eval_type="measurements", eval_path=measurements_path,
                                map_algo="ga", ga_conf_path=ga_config_path)
