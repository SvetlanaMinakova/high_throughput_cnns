import argparse
import os
import traceback
import sys


def main():
    """ Run parts of the pipeline for example CNN and example platform, defined in ../test_config.py"""
    # import current directory and it's subdirectories into system path for the current console
    # this would allow importing project modules without adding the project to the PYTHONPATH
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(root_dir)

    # general arguments
    parser = argparse.ArgumentParser(description='The script runs unit (per-step) tests for an example '
                                                 'CNN and an example platform, defined in ./test_config.py')
    parser.add_argument("-s", "--step", type=str, action='store', required=True,
                        help='Script (step) to test. Select from [dnn_to_sdf,'
                             'sdf_latency_eval_template, generate_mapping, generate_final_app]')

    parser.add_argument('--info-level', metavar='--info-level', type=int, action='store', default=1,
                        help='Info-level, i.e., amount of information to print out during the tests run. '
                             'If info-level == 0, no information is printed to console; '
                             'If info-level == 1, only tests information '
                             '(e.g., which steps of the tests were successful) is printed to console;'
                             'If info-level == 2, tests information (e.g., which steps of the tests were successful) '
                             'as well as script-specific verbose output is printed to the console')

    try:
        args = parser.parse_args()
        info_level = args.info_level
        step = args.step

        # import project modules
        from util import get_project_root
        from fileworkers.common_fw import clear_folder, create_or_overwrite_dir
        from tests.test_config import get_test_config

        unit_test_result = run_test_step(step, info_level)
        return unit_test_result

    except Exception as e:
        print(" Integration tests error: " + str(e))
        traceback.print_tb(e.__traceback__)
        return 1


def run_test_step(step: str, info_level):
    from tests.test_config import get_test_config
    config = get_test_config()
    if step == "parse_inputs":
        result = parse_inputs(config, info_level)
        return result

    if step == "dnn_to_sdf":
        result = run_dnn_to_sdf_task_graph(config, info_level)
        return result

    if step == "sdf_latency_eval_template":
        result = run_sdf_latency_eval_template(config, info_level)
        return result

    if step == "generate_mapping":
        result = run_generate_mapping(config, info_level)
        return result

    if step == "generate_final_app":
        result = run_generate_final_app(config, info_level)
        return result

    if step == "generate_code":
        result = run_generate_code(config, info_level)
        return result

    raise Exception("Unknown tests step: " + step)


def parse_inputs(config: {}, info_level):
    """
    Parse toolflow inputs: dnn models and target platform architecture
        in all supported formats
    :param config: test config (see ../test_config.py) used
        by subsequent scripts of the tool
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    if info_level > 0:
        print("Parse toolflow inputs (dnn models and target platform architecture)")
    dnn_parsed = parse_dnn(config, info_level)
    platform_parsed = parse_platform_architecture(config, info_level)
    test_passed = dnn_parsed and platform_parsed
    if info_level > 0:
        if test_passed:
            print("TEST PASSED")
        else:
            print("TEST FAILED")
    return test_passed


def parse_dnn(config: {}, info_level):
    """ try to parse input dnns of available formats
    :param config: test config (see ../test_config.py) used
        by subsequent scripts of the tool
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    try:
        dnn_from_onnx_parsed = parse_dnn_with_extension(config["cnn_onnx"], "onnx", info_level, "test_cnn_onnx")
        dnn_from_json_parsed = parse_dnn_with_extension(config["cnn_json"], "json", info_level, "test_cnn_json")
        test_passed = dnn_from_onnx_parsed and dnn_from_json_parsed
    except Exception as e:
        test_passed = False
    return test_passed


def parse_platform_architecture(config: {}, info_level):
    """ try to parse target platform architecture in all supported formats
    :param config: test config (see ../test_config.py) used
        by subsequent scripts of the tool
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    from converters.json_converters.json_to_architecture import json_to_architecture
    test_passed = True
    if info_level > 0:
        print("  Parse target platform architecture ( json )")
    try:
        arch_parsed = json_to_architecture(config["platform"])
        if arch_parsed is None:
            test_passed = False
    except Exception as e:
        test_passed = False

    if info_level > 0:
        if test_passed:
            print("  - SUCCESS")
        else:
            print("  - FAILURE")
    return test_passed


def parse_dnn_with_extension(dnn_path: str, dnn_extension: str, info_level, dnn_name="test_dnn"):
    """ Check if dnn can be parsed
    :param dnn_path: full path to dnn
    :param dnn_extension: target dnn extension
    :param dnn_name: target dnn name
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    from dnn_builders.input_dnn_manager import load_or_build_dnn_for_analysis

    if info_level > 0:
        print("  Parse DNN (", dnn_extension, ")")
    test_passed = True

    try:
        verbose = info_level > 1
        dnn = load_or_build_dnn_for_analysis(dnn_path, dnn_name, verbose)
        if dnn is None:
            test_passed = False

    except Exception as e:
        test_passed = False

    if info_level > 0:
        if test_passed:
            print("  - SUCCESS")
        else:
            print("  - FAILURE")

    return test_passed


def run_dnn_to_sdf_task_graph(config: {}, info_level):
    """ Run dnn_to_sdf_tasK_graph script
    :param config: test config (see ../test_config.py)
        by subsequent scripts of the tool
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    if info_level > 0:
        print("RUN CNN-to-SDF (task graph) conversion")

    # import project modules
    from util import get_project_root
    from t_unit_common import run_script_and_check_output

    script_root = get_project_root()
    script_name = "dnn_to_sdf_task_graph"
    input_param = {
        "--cnn": config["cnn_json"],
        "-o": config["intermediate_files_folder_abs"],
        "-fo": 'activation,normalization,arithmetic,skip'
    }
    flags = []
    # flags.append("--help")
    if info_level < 2:
        flags.append("--silent")

    output_files = ["task_graph.json"]
    output_file_abs_paths = [str(os.path.join(config["intermediate_files_folder_abs"], f)) for f in output_files]
    test_passed = run_script_and_check_output(script_root,
                                              script_name,
                                              input_param,
                                              flags,
                                              output_file_abs_paths,
                                              info_level)
    return test_passed


def run_sdf_latency_eval_template(config, info_level):
    """ Run sdf_latency_eval_template script
    :param config: test config (see ../test_config.py)
        by subsequent scripts of the tool
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    if info_level > 0:
        print("GENERATE task graph (SDF) Latency Eval template")

    # import project modules
    from util import get_project_root
    from t_unit_common import run_script_and_check_output

    script_root = get_project_root()
    script_name = "sdf_latency_eval_template"

    output_files = ["eval_template.json"]
    output_file_abs_paths = [str(os.path.join(config["intermediate_files_folder_abs"], f)) for f in output_files]

    # PART 1 (EMPTY)
    if info_level > 0:
        print("  PART 1: empty eval template")
    input_param = {
        "-tg": str(os.path.join(config["intermediate_files_folder_abs"], "task_graph.json")),
        "-p": config["platform"],
        "-o": config["intermediate_files_folder_abs"]
    }
    flags = []
    if info_level < 2:
        flags.append("--silent")
    test1_passed = run_script_and_check_output(script_root,
                                               script_name,
                                               input_param,
                                               flags,
                                               output_file_abs_paths,
                                               info_level)

    # PART 2 (FLOPS-BASED)
    if info_level > 0:
        print("  PART 2: FLOPS-based eval template")
    input_param = {
        "--cnn": config["cnn_json"],
        "-tg": str(os.path.join(config["intermediate_files_folder_abs"], "task_graph.json")),
        "-p": config["platform"],
        "-o": config["intermediate_files_folder_abs"]
    }
    flags = ["--flops"]
    if info_level < 2:
        flags.append("--silent")

    test2_passed = run_script_and_check_output(script_root,
                                               script_name,
                                               input_param,
                                               flags,
                                               output_file_abs_paths,
                                               info_level)

    test_passed = test1_passed and test2_passed
    return test_passed


def run_generate_mapping(config, info_level):
    """ Run generate_mapping script
    :param config: test config (see ../test_config.py)
        by subsequent scripts of the tool
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    if info_level > 0:
        print("GENERATE mapping")

    output_files = ["mapping.json"]
    output_file_abs_paths = [str(os.path.join(config["intermediate_files_folder_abs"], f)) for f in output_files]

    # import project modules
    from util import get_project_root
    from t_unit_common import run_script_and_check_output

    script_root = get_project_root()
    script_name = "generate_mapping"

    # PART 1 (GREEDY)
    if info_level > 0:
        print("  PART 1: greedy mapping")
    input_param = {
        "--cnn": config["cnn_json"],
        "-tg": str(os.path.join(config["intermediate_files_folder_abs"], "task_graph.json")),
        "-p": config["platform"],
        "-o": config["intermediate_files_folder_abs"],
        "-e": str(os.path.join(config["intermediate_files_folder_abs"], "eval_template.json")),
        "-map-algo": "greedy"
    }
    flags = []
    if info_level < 2:
        flags.append("--silent")

    test1_passed = run_script_and_check_output(script_root,
                                               script_name,
                                               input_param,
                                               flags,
                                               output_file_abs_paths,
                                               info_level)

    # PART 2 (GA-BASED)
    if info_level > 0:
        print("  PART 2: GA-based mapping (may take a while)")

    input_param = {
        "--cnn": config["cnn_json"],
        "-tg": str(os.path.join(config["intermediate_files_folder_abs"], "task_graph.json")),
        "-p": config["platform"],
        "-o": config["intermediate_files_folder_abs"],
        "-e": str(os.path.join(config["intermediate_files_folder_abs"], "eval_template.json")),
        "-ga-config": str(os.path.join(config["input_files_folder_abs"], "ga_config", "ga_conf_generic.json")),
        "-map-algo": "ga"
    }
    flags = []
    if info_level < 2:
        flags.append("--silent")

    test2_passed = run_script_and_check_output(script_root,
                                               script_name,
                                               input_param,
                                               flags,
                                               output_file_abs_paths,
                                               info_level)

    test_passed = test1_passed and test2_passed
    return test_passed


def run_generate_final_app(config, info_level):
    """ Run generate_final_app script
    :param config: test config (see ../test_config.py)
        by subsequent scripts of the tool
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    if info_level > 0:
        print("GENERATE final application")

    # import project modules
    from util import get_project_root
    from t_unit_common import run_script_and_check_output

    script_root = get_project_root()
    script_name = "generate_final_app_model"
    input_param = {
        "--cnn": config["cnn_json"],
        "-tg": str(os.path.join(config["intermediate_files_folder_abs"], "task_graph.json")),
        "-p": config["platform"],
        "-m": str(os.path.join(config["intermediate_files_folder_abs"], "mapping.json")),
        "-o": config["intermediate_files_folder_abs"],
    }
    flags = []
    if info_level < 2:
        flags.append("--silent")

    output_files = ["app.json"]
    output_file_abs_paths = [str(os.path.join(config["intermediate_files_folder_abs"], f)) for f in output_files]
    test_passed = run_script_and_check_output(script_root,
                                              script_name,
                                              input_param,
                                              flags,
                                              output_file_abs_paths,
                                              info_level)
    return test_passed


def run_generate_code(config, info_level):
    """ Run generate_code* scripts
    :param config: test config (see ../test_config.py) used
        by subsequent scripts of the tool
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    if info_level > 0:
        print("GENERATE code")

    print_prefix = "  "
    wrapper_code_generated = run_generate_code_wrapper(config, info_level, print_prefix)
    cpu_code_generated = run_generate_code_arm_cl(config, info_level, print_prefix)
    gpu_code_generated = run_generate_code_tensorrt(config, info_level, print_prefix)
    mixed_code_generated = run_generate_code_mixed(config, info_level, print_prefix)
    test_passed = wrapper_code_generated and cpu_code_generated and gpu_code_generated and mixed_code_generated
    return test_passed


def run_generate_code_wrapper(config, info_level, print_prefix="  "):
    """ Run generate_code_wrapper script
    :param config: test config (see ../test_config.py) used
        by subsequent scripts of the tool
    :param print_prefix: space before printout statements printed to the console
        during the tests execution
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    if info_level > 0:
        print(print_prefix, "GENERATE wrapper code")

    output_files = ["appMain.cpp", "Makefile", "SharedBuffer.h"]
    output_file_abs_paths = [str(os.path.join(config["intermediate_files_folder_abs"], "code", "wrapper", f)) for f in output_files]

    # import project modules
    from util import get_project_root
    from t_unit_common import run_script_and_check_output

    script_root = get_project_root()
    script_name = "generate_code_wrapper"

    # PART 1 (UNTIMED)
    if info_level > 0:
        print(print_prefix, "PART 1: Untimed")
    input_param = {
        "--cnn": config["cnn_json"],
        "-a": str(os.path.join(config["intermediate_files_folder_abs"], "app.json")),
        "-o": config["intermediate_files_folder_abs"]
    }
    flags = []
    if info_level < 2:
        flags.append("--silent")

    test1_passed = run_script_and_check_output(script_root,
                                               script_name,
                                               input_param,
                                               flags,
                                               output_file_abs_paths,
                                               info_level)

    # PART 2 (TIMED)
    if info_level > 0:
        print(print_prefix, "PART 2: Timed")

    input_param = {
        "--cnn": config["cnn_json"],
        "-a": str(os.path.join(config["intermediate_files_folder_abs"], "app.json")),
        "-o": config["intermediate_files_folder_abs"],
        "-e": str(os.path.join(config["intermediate_files_folder_abs"], "eval_template.json")),
    }
    flags = []
    if info_level < 2:
        flags.append("--silent")

    test2_passed = run_script_and_check_output(script_root,
                                               script_name,
                                               input_param,
                                               flags,
                                               output_file_abs_paths,
                                               info_level)

    test_passed = test1_passed and test2_passed
    return test_passed


def run_generate_code_arm_cl(config, info_level, print_prefix="  "):
    """ Run generate_code_arm_cl script
    :param config: test config (see ../test_config.py) used
        by subsequent scripts of the tool
    :param print_prefix: space before printout statements printed to the console
        during the tests execution
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    if info_level > 0:
        print(print_prefix, "GENERATE ARM-CL (CPU) code")

    output_files = ["appMain.cpp", "Makefile", "SharedBuffer.h"]
    output_file_abs_paths1 = [str(os.path.join(config["intermediate_files_folder_abs"], "code",
                                               "cpu", f)) for f in output_files]
    output_file_abs_paths2 = [str(os.path.join(config["intermediate_files_folder_abs"], "code",
                                               "cpu_partitioned", f)) for f in output_files]

    # import project modules
    from util import get_project_root
    from t_unit_common import run_script_and_check_output

    script_root = get_project_root()
    script_name = "generate_code_arm_cl"

    # PART 1 (Whole)
    if info_level > 0:
        print(print_prefix, "PART 1: Whole")
    input_param = {
        "--cnn": config["cnn_json"],
        "-o": config["intermediate_files_folder_abs"]
    }
    flags = []
    if info_level < 2:
        flags.append("--silent")

    test1_passed = run_script_and_check_output(script_root,
                                               script_name,
                                               input_param,
                                               flags,
                                               output_file_abs_paths1,
                                               info_level)

    # PART 2 (Partitioned)
    if info_level > 0:
        print(print_prefix, "PART 2: Split into partitions (sub-networks) using task graph (BENCHMARK/DEBUG MODE)")

    input_param = {
        "--cnn": config["cnn_json"],
        "-tg": str(os.path.join(config["intermediate_files_folder_abs"], "task_graph.json")),
        "-o": config["intermediate_files_folder_abs"]
    }
    flags = ["--partitioned"]
    if info_level < 2:
        flags.append("--silent")

    test2_passed = run_script_and_check_output(script_root,
                                               script_name,
                                               input_param,
                                               flags,
                                               output_file_abs_paths2,
                                               info_level)

    test_passed = test1_passed and test2_passed
    return test_passed


def run_generate_code_tensorrt(config, info_level, print_prefix="  "):
    """ Run generate_code_tensorrt script
    :param config: test config (see ../test_config.py) used
        by subsequent scripts of the tool
    :param print_prefix: space before printout statements printed to the console
        during the tests execution
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    if info_level > 0:
        print(print_prefix, "GENERATE TensorRT (GPU) code")

    output_files = ["appMain.cpp", "Makefile", "SharedBuffer.h"]
    output_file_abs_paths1 = [str(os.path.join(config["intermediate_files_folder_abs"], "code",
                                               "gpu", f)) for f in output_files]
    output_file_abs_paths2 = [str(os.path.join(config["intermediate_files_folder_abs"], "code",
                                               "gpu_partitioned", f)) for f in output_files]

    # import project modules
    from util import get_project_root
    from t_unit_common import run_script_and_check_output

    script_root = get_project_root()
    script_name = "generate_code_tensorrt"

    # PART 1 (Whole)
    if info_level > 0:
        print(print_prefix, "PART 1: Whole")
    input_param = {
        "--cnn": config["cnn_json"],
        "-o": config["intermediate_files_folder_abs"]
    }
    flags = []
    if info_level < 2:
        flags.append("--silent")

    test1_passed = run_script_and_check_output(script_root,
                                               script_name,
                                               input_param,
                                               flags,
                                               output_file_abs_paths1,
                                               info_level)

    # PART 2 (Partitioned)
    if info_level > 0:
        print(print_prefix, "PART 2: Split into partitions (sub-networks) using task graph (DEBUG MODE)")

    input_param = {
        "--cnn": config["cnn_json"],
        "-tg": str(os.path.join(config["intermediate_files_folder_abs"], "task_graph.json")),
        "-o": config["intermediate_files_folder_abs"]
    }
    flags = ["--partitioned"]
    if info_level < 2:
        flags.append("--silent")

    test2_passed = run_script_and_check_output(script_root,
                                               script_name,
                                               input_param,
                                               flags,
                                               output_file_abs_paths2,
                                               info_level)

    test_passed = test1_passed and test2_passed
    return test_passed


def run_generate_code_mixed(config, info_level, print_prefix="  "):
    """ Run generate_code_mixed script
    :param config: test config (see ../test_config.py) used
        by subsequent scripts of the tool
    :param print_prefix: space before printout statements printed to the console
        during the tests execution
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    if info_level > 0:
        print(print_prefix, "GENERATE mixed ARM-CL (CPU) + TensorRT (GPU) code "
                            "with partitioning and mapping based on final CSDF app")

    output_files = ["appMain.cpp", "Makefile", "SharedBuffer.h"]
    output_file_abs_paths = [str(os.path.join(config["intermediate_files_folder_abs"],
                                              "code", "mixed", f)) for f in output_files]

    # import project modules
    from util import get_project_root
    from t_unit_common import run_script_and_check_output

    script_root = get_project_root()
    script_name = "generate_code_mixed"

    input_param = {
        "--cnn": config["cnn_json"],
        "-a": str(os.path.join(config["intermediate_files_folder_abs"], "app.json")),
        "-o": config["intermediate_files_folder_abs"]
    }
    flags = []
    if info_level < 2:
        flags.append("--silent")

    test_passed = run_script_and_check_output(script_root,
                                              script_name,
                                              input_param,
                                              flags,
                                              output_file_abs_paths,
                                              info_level)
    return test_passed


if __name__ == "__main__":
    main()

    