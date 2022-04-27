import argparse
import traceback
import os
import sys


def main():
    # import current directory and it's subdirectories into system path for the current console
    # this would allow to import project modules without adding the project to the PYTHONPATH
    this_dir = os.path.dirname(__file__)
    sys.path.append(this_dir)

    # import project modules
    from dnn_builders.input_dnn_manager import load_or_build_dnn_for_analysis
    from models.dnn_model.dnn import set_built_in
    from models.dnn_model.transformation.ops_fusion import fuse_built_in
    from converters.json_converters.json_task_graph import parse_task_graph_json
    from dnn_partitioning.before_mapping.partition_dnn_with_task_graph import partition_dnn_with_task_graph
    from util import print_stage
    import codegen.arm_cl.arm_cl_dnn_visitor

    # general arguments
    parser = argparse.ArgumentParser(description='The script generates pure CPU (ARM-CL) code for a cnn')

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

    parser.add_argument('-tg', metavar='--task-graph', type=str, action='store', default=None,
                        help='path to .json file with the task-graph (SDF) model generated '
                             'for the input cnn by the ./dnn_to_sdf_task_graph.py script. This input is required'
                             'when --partitioned flag is used')

    parser.add_argument('-o', metavar='--out-dir', type=str, action='store', default="./output",
                        help='path to output files directory')

    # general flags
    parser.add_argument("--silent", help="do not provide print-out for the script steps", action="store_true", default=False)
    parser.add_argument("--partitioned", action='store_true', default=False,
                        help="generates ARM-CL code where every task graph node is represented as a DNN partition "
                             "(sub-network). This mode is used for Task Graph (SDF) latency measurement on the "
                             "platform. When this flag is use, the DNN task graph should be passed to the script")

    args = parser.parse_args()
    try:
        # Extract parameters from command-line arguments"
        cnn_path = args.cnn
        output_dir = args.o
        silent = args.silent
        verbose = not silent
        fused_ops_spec = args.fo
        fused_ops = fused_ops_spec.split(',')
        # per-layer (benchmark) mode
        partitioned = args.partitioned
        task_graph_path = args.tg

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

        if partitioned:
            stage = "Reading task graph (SDF) model "
            print_stage(stage, verbose)
            if task_graph_path is None:
                raise Exception("Task Graph path is None. You must specify path to task graph to use --partitioned flag")
            task_graph = parse_task_graph_json(task_graph_path)

            stage = "Partitioning DNN with task graph (SDF) model "
            print_stage(stage, verbose)
            partitioned_dnn, connections = partition_dnn_with_task_graph(dnn, task_graph)
            
            stage = "Generating ARM-CL (CPU) code (PER PARTITION)"
            print_stage(stage, verbose)
            code_folder = output_dir + "/code/cpu_partitioned"
            codegen.arm_cl.arm_cl_dnn_visitor.visit_dnn_partitioned(partitioned_dnn,
                                                                    connections,
                                                                    code_folder,
                                                                    verbose)
        else:
            stage = "Generating ARM-CL (CPU) code"
            print_stage(stage, verbose)
            code_folder = output_dir + "/code/cpu"
            codegen.arm_cl.arm_cl_dnn_visitor.visit_dnn(dnn, code_folder, verbose)

    except Exception as e:
        print("ARM-CL (CPU) code generation error: " + str(e))
        traceback.print_tb(e.__traceback__)


if __name__ == "__main__":
    main()