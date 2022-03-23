import argparse
import traceback
import os
import sys


def main():
    # import current directory and it's subdirectories into system path for the current console
    # this would allow importing project modules without adding the project to the PYTHONPATH
    this_dir = os.path.dirname(__file__)
    sys.path.append(this_dir)

    # import project modules
    from dnn_builders.input_dnn_manager import load_or_build_dnn_for_analysis
    from converters.json_converters.dnn_to_json import dnn_to_json
    from models.dnn_model.dnn import set_built_in
    from models.dnn_model.transformation.ops_fusion import fuse_built_in
    from util import print_stage

    # general arguments
    parser = argparse.ArgumentParser(description='The script converts an input CNN (in .onnx or .h5 format) into an '
                                                 'internal (analytical) CNN model. The internal DNN '
                                                 'model is a light-weighted and generic DNN model highly suitable '
                                                 'for analysis, performed by the tool. When you pass an ONNX or .h5'
                                                 'DNN model to the tool, this model is always first converted to the '
                                                 'internal analytical model. However, conversion does not always '
                                                 'go smooth and depends on the third-party tools '
                                                 '(such as onnx package). This script enables you to take a look'
                                                 'into the analytical CNN model, automatically obtained from the'
                                                 'onnx or .h5 format and manually change it. The analytical '
                                                 'CNN model is saved as a JSON file.')

    parser.add_argument('--cnn', metavar='--cnn', type=str, action='store', required=True,
                        help='path to a CNN. Can be a path to: '
                             '1) a path to an .onnx file; '
                             '2) a path to .h5 file (cnn in format of Keras DL framework). ')

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

    parser.add_argument('-o', metavar='--out-dir', type=str, action='store', default="./output",
                        help='path to output files directory')

    # general flags
    parser.add_argument("--silent", help="do not provide print-out for the script steps", action="store_true", default=False)

    args = parser.parse_args()
    try:
        # Extract parameters from command-line arguments"
        cnn_path = args.cnn
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

        stage = "Saving analytical DNN model as a .json file"
        print_stage(stage, verbose)
        out_dnn_path = str(os.path.join(output_dir, "cnn.json"))
        dnn_to_json(dnn, out_dnn_path)

    except Exception as e:
        print("Analytical (internal) CNN model derivation error: " + str(e))
        traceback.print_tb(e.__traceback__)


if __name__ == "__main__":
    main()