import argparse
import os
import subprocess
import traceback
import sys


def main():
    """ Run the whole pipeline for example mnist CNN and example jetson TX2 platform"""
    # import current directory and it's subdirectories into system path for the current console
    # this would allow to import project modules without adding the project to the PYTHONPATH
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(root_dir)

    # general arguments
    parser = argparse.ArgumentParser(description='The script runs a full-pipeline test for an example '
                                                 './input_examples/dnn/mnist.onnx CNN and an example '
                                                 './input_examples/architecture/jetson.json platform architecture')

    parser.add_argument('--info-level', metavar='--info-level', type=int, action='store', default=1,
                        help='Info-level, i.e., amount of information to print out during the test run. '
                             'If info-level == 0, no information is printed to console; '
                             'If info-level == 1, only test information '
                             '(e.g., which steps of the test were successful) is printed to console;'
                             'If info-level == 2, test information (e.g., which steps of the test were successful) '
                             'as well as script-specific verbose output is printed to the console')

    try:
        args = parser.parse_args()
        info_level = args.info_level

        # import project modules
        from util import get_project_root
        from fileworkers.common_fw import clear_folder, create_or_overwrite_dir

        input_files_folder_abs = os.path.join(get_project_root(), "input_examples")
        intermediate_files_folder_abs = os.path.join(get_project_root(), "output", "mnist")

        # cleanup intermediate files directory
        # print("cleanup folder", intermediate_files_folder_abs)
        clear_folder(intermediate_files_folder_abs)
        create_or_overwrite_dir(intermediate_files_folder_abs)

        run_test(input_files_folder_abs, intermediate_files_folder_abs, info_level)

        return True
    except Exception as e:
        print(" Integration test error: " + str(e))
        traceback.print_tb(e.__traceback__)
        return False


def run_test(input_files_folder_abs, intermediate_files_folder_abs, info_level):
    """
    Run integration test
    :param input_files_folder_abs: absolute path to input (cnn and platform arch. files)
    :param intermediate_files_folder_abs: absolute path to intermediate files, produced
        by subsequent scripts of the tool
    :param info_level: amount of information to print out during the test run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only test information (e.g., which steps of the test were successful)
        is printed to console.
        If info-level == 2, test information (e.g., which steps of the test were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if test ran successfully and False otherwise
    """
    if info_level > 0:
        print("RUN TEST")
    step_1_result = run_dnn_to_sdf_task_graph(input_files_folder_abs, intermediate_files_folder_abs, info_level)


def run_dnn_to_sdf_task_graph(input_files_folder_abs, intermediate_files_folder_abs, info_level):
    """ Run dnn_to_sdf_tasK_graph script
    :param input_files_folder_abs: absolute path to input (cnn and platform arch. files)
    :param intermediate_files_folder_abs: absolute path to intermediate files, produced
        by subsequent scripts of the tool
    :param info_level: amount of information to print out during the test run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only test information (e.g., which steps of the test were successful)
        is printed to console.
        If info-level == 2, test information (e.g., which steps of the test were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if test ran successfully and False otherwise
    """
    if info_level > 0:
        print("  - RUN CNN-to-SDF (task graph) conversion")

    # import project modules
    from util import get_project_root
    script_root = get_project_root()
    script_name = "dnn_to_sdf_task_graph"
    input_param = {
        # "-cnn": str(os.path.join(input_files_folder_abs, "dnn", "mnist.onnx")),
        # "-o": str(os.path.join(intermediate_files_folder_abs)),
        # "-fo": 'activation,normalization,arithmetic,skip'
    }
    flags = []
    flags.append("--help")
    if info_level < 2:
        flags.append("--silent")

    output_files = ["task_graph.json"]
    script_result = run_script(script_root, script_name, input_param, flags, output_files)
    return script_result


def run_script(script_root, script_name, input_param: {}, flags: [], output_files: []):
    """
    Run script and check whether it was properly executed.
    If the script was properly executed it would generate a specific set of output files.
    :param script_root: folder, where script is located
    :param script_name: name of the script
    :param input_param: (dictionary) list of input parameters, where key=parameter name,
        value = parameter value
    :param flags: list of input flags
    :param output_files: list of output files, that are expected to be generated
    :return: True, if the list of output files was generated, False otherwise
    """
    script_path = str(os.path.join(script_root, (script_name + ".py")))
    script_call = script_path + " "
    # add parameters
    for item in input_param.items():
        param, val = item
        script_call += param + " " + str(val) + " "

    # add flags
    for flag in flags:
        script_call += flag + " "

    # print("execute script:")
    # print(script_call)

    result = subprocess.run(
        [sys.executable, script_call], capture_output=True, text=True
    )

    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    return True


if __name__ == "__main__":
    main()

