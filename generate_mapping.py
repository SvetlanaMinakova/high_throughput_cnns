import os
import sys
import argparse
import traceback

# example
# python generate_mapping.py --cnn /home/svetlana/ONNX/OnnxZooModels/alexnet.onnx /
# -tg ./output/alexnet/task_graph.json -p ./output/architecture/jetson.json /
# -map-algo ga -e ./output/alexnet/eval.json /
# -ga-config ./input_examples/intermediate/ga_conf_generic.json -o ./output/alexnet

# ex 2
# ../kerasProj/venv/bin/python ./generate_mapping.py --cnn /home/svetlana/ONNX/OnnxZooModels/mnist.onnx -tg ./output/mnist/task_graph.json -p ./output/architecture/jetson.json -o ./output/mnist/ -e ./output/mnist/eval.json -map-algo greedy


def main():
    # import current directory and it's subdirectories into system path for the current console
    # this would allow to import project modules without adding the project to the PYTHONPATH
    this_dir = os.path.dirname(__file__)
    sys.path.append(this_dir)

    # import project modules
    from util import print_stage
    from converters.json_converters.json_to_architecture import json_to_architecture
    from DSE.mapping.map_and_partition import get_mapping, get_partitioning
    from DSE.eval_table.direct_measurements_et_builder import build_eval_table
    from converters.json_converters.json_task_graph import parse_task_graph_json
    from converters.json_converters.mapping_to_json import mapping_to_json
    from converters.json_converters.partitioning_to_json import partitioning_to_json
    from dnn_builders.input_dnn_manager import load_or_build_dnn_for_analysis
    from models.dnn_model.dnn import set_built_in
    from models.dnn_model.transformation.ops_fusion import fuse_built_in

    # general arguments
    parser = argparse.ArgumentParser(description='The script maps an input CNN represented '
                                                 'as a task-graph (SDF) model '
                                                 '(using ./dnn_to_sdf_task_graph.py), onto a target edge platform'
                                                 '(architecture). The mapping is saved in an output JSON file')

    parser.add_argument('--cnn', metavar='--cnn', type=str, action='store', required=True,
                        help='path to a CNN. Can be a path to: '
                             '1) a path to an .onnx file; '
                             '2) a path to .h5 file (cnn in format of Keras DL framework). '
                             '3) a path to .json file (cnn in internal format. This format'
                             'can be obtained from on .onnx or .h5 file using ./dnn_to_json script)')

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

    parser.add_argument('-tg', metavar='--task-graph', type=str, action='store', required=True,
                        help='path to .json file with the task-graph (SDF) model generated '
                             'for the input cnn by the ./dnn_to_sdf_task_graph.py script')

    parser.add_argument('-p', metavar='--platform', type=str, action='store', required=True,
                        help='path to edge platform (architecture) description, saved in .json format')

    parser.add_argument('-map-algo', metavar='--map-algo', type=str, action='store', default="greedy",
                        help='value in [greedy, ga]: algorithm used for mapping of (computations within) the CNN'
                             'onto (the processors of) the target edge platform')

    parser.add_argument('-ga-config', metavar='--ga-config', type=str, action='store', default=None,
                        help='path to path to GA config. Required for GA-based search (map_algo="ga")')

    parser.add_argument('-e', metavar='--eval-path', type=str, action='store', required=True,
                        help='path to per-layer cnn execution time (latency) evaluation (JSON). '
                             'Use ./sdf_latency_eval_template.py script to generate a template for this file.')

    parser.add_argument('-o', metavar='--out-dir', type=str, action='store', default="./output",
                        help='path to output files directory')

    # general flags
    parser.add_argument("--silent", help="do not provide print-out for the script steps", action="store_true", default=False)

    args = parser.parse_args()
    try:
        # Extract parameters from command-line arguments"
        cnn_path = args.cnn
        fused_ops_spec = args.fo
        fused_ops = fused_ops_spec.split(',')

        task_graph_path = args.tg
        platform_path = args.p
        map_algo = args.map_algo
        ga_conf_path = args.ga_config
        eval_path = args.e
        output_dir = args.o
        silent = args.silent
        verbose = not silent

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

        stage = "Reading task graph (SDF) model "
        print_stage(stage, verbose)
        task_graph = parse_task_graph_json(task_graph_path)

        stage = "Reading target platform architecture"
        print_stage(stage, verbose)
        architecture = json_to_architecture(platform_path)

        # build time eval matrix
        stage = "Read per-layer execution time (latency) eval table"
        print_stage(stage, verbose)
        eval_table = build_eval_table(eval_path, architecture.processors_types_distinct, task_graph)

        # --------------
        # get mapping
        stage = "Map dnn onto target hardware architecture (mapping algorithm = " + str(map_algo) + ")"
        print_stage(stage, verbose)
        mapping = get_mapping(task_graph, architecture, eval_table, map_algo, ga_conf_path, verbose)

        stage = "Save mapping as a JSON file"
        print_stage(stage, verbose)
        mapping_to_json(task_graph, architecture, mapping, output_dir, verbose)

        # --------------
        # get partitioning
        # PARTITIONING is outdated. Now the CNN partitioning together with the CNN mapping
        # and scheduling are saved encoded in teh final application model
        # --------------
        # get partitioning
        """
        stage = "Partition mapped DNN (create final CSDF graph)"
        print_stage(stage, verbose)
        partitions, connections = get_partitioning(dnn, task_graph, mapping)

        # save partitioning as a .json file
        partitioning_to_json(dnn, architecture, mapping, partitions, connections, output_dir, verbose)
        """

    except Exception as e:
        print("Mapping generation error: " + str(e))
        traceback.print_tb(e.__traceback__)


if __name__ == "__main__":
    main()
