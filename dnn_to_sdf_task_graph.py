import argparse
import traceback
import os
import sys

"""
Console API file
"""

# test run examples:
# onnx file, time eval
# python eval_aloha.py --cnn input_examples/dnns/onnx_dnns/o34461.onnx -p input_examples/platforms/jetson.json -t

# ofa-encoded dnn, time, memory, energy eval
# python eval_aloha.py --cnn d22344-e686668808668868886 -p input_examples/platforms/jetson.json -t -m -e

# a directory with onnx files, time, memory, energy eval
# python eval_aloha.py --cnn input_examples/dnns/onnx_dnns -p input_examples/platforms/jetson.json -t -m


def main():
    # import current directory and it's subdirectories into system path for the current console
    # this would allow to import project modules without adding the project to the PYTHONPATH
    this_dir = os.path.dirname(__file__)
    sys.path.append(this_dir)

    # import project modules
    from high_throughput.mapping.map_and_partition import map_and_partition
    from dnn_builders.input_dnn_manager import load_or_build_dnn_for_analysis
    from models.edge_platform.Architecture import get_jetson
    from models.dnn_model.dnn import set_built_in
    from models.dnn_model.transformation.ops_fusion import fuse_built_in
    from util import get_project_root

    # general arguments
    parser = argparse.ArgumentParser(description='The script converts an input CNN (in supported input format) into a '
                                                 'task-graph (SDF) model where every node is a task. One node '
                                                 'of the task graph (SDF) model is functionally equivalent '
                                                 'to one or more input DNN layers')

    parser.add_argument('--cnn', metavar='cnn', type=str, action='store', required=True,
                        help='path to one or several CNNs. Can be a path to: '
                             '1) a path to an .onnx file; '
                             '2) a path to .h5 file (cnn in format of Keras DL framework). ')

    parser.add_argument('--o', metavar='out-dir', type=str, action='store', required=True,
                        help='path to output files directory')

    args = parser.parse_args()
    try:
        # extract parameters from command-line arguments
        cnn_path = args.cnn
        output_dir = args.o

        # read DNN
        dnn = load_or_build_dnn_for_analysis(cnn_path)

        """
        # optimize dnn: fuse operators
        built_in_ops = ["activation", "normalization", "arithmetic", "skip"]
        set_built_in(dnn, built_in_ops)
        fuse_built_in(dnn)
        """

        # print DNN
        # dnn.print_details()

        inp_files_directory = str(os.path.join(str(get_project_root()), "input_examples/high_throughput/" + dnn.name))
        outp_files_directory = str(os.path.join(str(get_project_root()), "output/" + dnn.name))

    except Exception as e:
        print(" Task Graph (SDF) model creation error: " + str(e))
        traceback.print_tb(e.__traceback__)


if __name__ == "__main__":
    main()