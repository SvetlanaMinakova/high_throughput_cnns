import argparse
import os
import subprocess
import traceback
import sys


def main():
    """ Run the whole pipeline for example mnist CNN and example jetson TX2 platform"""
    # import current directory and it's subdirectories into system path for the current console
    # this would allow importing project modules without adding the project to the PYTHONPATH
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(root_dir)

    # general arguments
    parser = argparse.ArgumentParser(description='The script runs an integration (end-to-end) tests for an example '
                                                 './input_examples/dnn/mnist.onnx CNN and an example '
                                                 './input_examples/architecture/jetson.json platform')

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

        # import project modules
        from util import get_project_root
        from fileworkers.common_fw import clear_folder, create_or_overwrite_dir
        from tests.test_config import get_test_config

        config = get_test_config()
        # cleanup intermediate files directory
        # print("cleanup folder", intermediate_files_folder_abs)
        clear_folder(config["intermediate_files_folder_abs"])
        create_or_overwrite_dir(config["intermediate_files_folder_abs"])
        integration_test_result = run_integration_test(info_level)
        return integration_test_result

    except Exception as e:
        print(" Integration tests error: " + str(e))
        traceback.print_tb(e.__traceback__)
        return 1


def run_integration_test(info_level):
    """
    Run integration tests
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    if info_level > 0:
        print("RUN INTEGRATION TEST")
    # steps (scripts) in execution order
    steps = ["dnn_to_sdf",
             "sdf_latency_eval_template",
             "generate_mapping",
             "generate_final_app"]
    for step in steps:
        step_executed = run_test_step(step, info_level)
        if step_executed is False:
            if info_level > 0:
                print("INTEGRATION TEST FAILED")
                return 1
    print("INTEGRATION TEST PASSED")
    return 0


def run_test_step(step: str, info_level):
    from tests.test_config import get_test_config
    config = get_test_config()

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

    raise Exception("Unknown tests step: " + step)


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
    script_root = get_project_root()
    script_name = "dnn_to_sdf_task_graph"
    input_param = {
        "--cnn": config["cnn"],
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
        "--cnn": config["cnn"],
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
    script_root = get_project_root()
    script_name = "generate_mapping"

    # PART 1 (GREEDY)
    if info_level > 0:
        print("  PART 1: greedy mapping")
    input_param = {
        "--cnn": config["cnn"],
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

    # PART 1 (GA-BASED)
    if info_level > 0:
        print("  PART 2: GA-based mapping")

    input_param = {
        "--cnn": config["cnn"],
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
    script_root = get_project_root()
    script_name = "generate_final_app_model"
    input_param = {
        "--cnn": config["cnn"],
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


def run_script_and_check_output(script_root,
                                script_name,
                                input_param: {},
                                flags: [],
                                output_file_paths: [],
                                info_level):
    """
    Run script and check whether it was properly executed.
    If the script was properly executed it would generate a specific set of output files.
    :param script_root: folder, where script is located
    :param script_name: name of the script
    :param input_param: (dictionary) list of input parameters, where key=parameter name,
        value = parameter value
    :param flags: list of input flags
    :param output_file_paths: list of absolute paths to expected output files
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True, if the script was successfully executed and
        all the output files was generated, False otherwise
    """
    # run script
    script_successfully_executed = run_script(script_root, script_name, input_param, flags, info_level)
    if info_level > 0:
        print("  - script successfully executed:", script_successfully_executed)

    # check the output files
    output_files_generated = files_exist(output_file_paths)
    if info_level > 0:
        print("  - script output files are successfully generated:", output_files_generated)

    test_passed = True if script_successfully_executed and output_files_generated else False
    if info_level > 0:
        if test_passed:
            print("TEST PASSED")
        else:
            print("TEST FAILED")
    return test_passed


def run_script(script_root, script_name, input_param: {}, flags: [], info_level):
    """
    Run script
    :param script_root: folder, where script is located
    :param script_name: name of the script
    :param input_param: (dictionary) list of input parameters, where key=parameter name,
        value = parameter value
    :param flags: list of input flags
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True, if the list of output files was generated, False otherwise
    """
    script_path = str(os.path.join(script_root, (script_name + ".py")))
    # define script call. The first parameter is the path to (current) python interpreter.
    # The second parameter is the (absolute) path to executable script
    script_call = [sys.executable, script_path]

    # add parameters
    for item in input_param.items():
        param, val = item
        script_call.append(param)
        script_call.append(val)

    # add flags
    for flag in flags:
        script_call.append(flag)

    # print("call param", script_call)
    result = subprocess.run(script_call, capture_output=True, text=True)
    success = True if result.returncode == 0 else False
    # print("success:", success)

    # print stdout and stderr returned by script
    if info_level > 0:
        print("  - script stdout:", result.stdout)
        print("  - script stderr:", result.stderr)

    return success


def files_exist(files_abs_paths):
    """
    Check if files exist
    :param files_abs_paths abs paths to files
    :return: True if all the files exist, False otherwise
    """
    for file_path in files_abs_paths:
        if not os.path.exists(file_path):
            return False
    return True


if __name__ == "__main__":
    main()
