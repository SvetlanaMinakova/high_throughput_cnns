from experiments.throughput_increase.ti_direct_measurements import increase_dnn_throughput
from models.edge_platform.Architecture import get_jetson
from util import get_project_root
import os
import sys

""" 
 Estimate additional increase, achieved by efficient mapping of 
 computations within a DNN onto computational resources of an edge platform
 Use real measurements for additional evaluation 
"""


def all_inputs_present(task_graph_path, eval_path, ga_conf_path):
    if not os.path.exists(task_graph_path):
        sys.stderr.write("Throughput increase error: task graph " + task_graph_path + " does not exist\n")
        return False
    if not os.path.exists(eval_path):
        sys.stderr.write("Throughput increase error: eval " + eval_path + " does not exist\n")
        return False
    if not os.path.exists(ga_conf_path):
        sys.stderr.write("Throughput increase error: ga config " + ga_conf_path + " does not exist\n")
        return False
    return True


def run_metafiles_folder():
    inp_files_directory = str(os.path.join(str(get_project_root()), "../input_examples/intermediate/new"))
    architecture = get_jetson()
    sub_folder_names = next(os.walk(inp_files_directory))[1]

    for sub_folder_name in sub_folder_names:
        print("///////////////////////////////////////////")
        print(sub_folder_name)

        sub_folder_path = str(os.path.join(inp_files_directory, sub_folder_name))
        dnn_name = sub_folder_name
        task_graph_path = str(os.path.join(sub_folder_path, "task_graph.json"))
        eval_path = str(os.path.join(sub_folder_path, "eval_template.json"))
        ga_conf_path = str(os.path.join(inp_files_directory, "ga_conf_generic.json"))
        if all_inputs_present(task_graph_path, eval_path, ga_conf_path):
            increase_dnn_throughput(dnn_name, architecture, task_graph_path, eval_path, ga_conf_path)


run_metafiles_folder()