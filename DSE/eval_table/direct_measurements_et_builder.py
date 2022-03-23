from models.edge_platform.Architecture import get_jetson
from converters.json_converters.json_task_graph import parse_task_graph_json
import json


def build_from_eval_and_tg_paths(task_graph_path, eval_path, architecture):
    """
    Build eval table from direct measurements
    :param architecture: target platform architecture
    :param task_graph_path: path to task-graph, obtained from the DNN and stored in .json format
    :param eval_path: path to evaluation, obtained from the measurements on the platform and stored in .json format
    :return: eval table
    """
    app_graph = parse_task_graph_json(task_graph_path)
    et = build_eval_table(eval_path, architecture.processors_types_distinct, app_graph)
    return et


def build_eval_table(json_eval, processor_types_distinct, app_graph):
    """
    Build evaluations table from .json file, containing direct measurements on the platform
    :param json_eval: .json evaluations file. File should contain fields with evaluations
     of DNN layers per processor_type_distinct in the platform processor
        example of vgg19 json_eval file is in /data/e0.json
    :param processor_types_distinct distinct types of processors, available
    on the platform (e.g. small_CPU, large_CPU, GPU)
    :param app_graph: application graph
    """
    eval_table = []
    with open(json_eval, 'r') as file:
        if file is None:
            raise FileNotFoundError

        for _ in processor_types_distinct:
            eval_table.append([])

        else:
            evals = json.load(file)
            proc_id = 0
            for proc_type in processor_types_distinct:

                if proc_type not in evals:
                    raise Exception(proc_type, "eval_table not found!")

                proc_eval = evals[proc_type]

                if proc_type == "GPU":
                    proc_eval = restore_gpu_evals(proc_eval, evals["layers"], app_graph)

                eval_table[proc_id] = proc_eval
                proc_id = proc_id + 1

    return eval_table


def restore_gpu_evals(proc_eval, gpu_eval_names, app_graph):
    """
    Fill with 0 time evaluations of nodes, merged to the parent nodes by tensorrt
    (such as BN or ReLU nodes)
    """
    restored_eval = []
    for task_id in range(len(app_graph.tasks)):
        task_eval = 0.0
        if len(app_graph.jobs_per_task) > task_id:
            layers_per_task = app_graph.jobs_per_task[task_id]
            for layer_name in layers_per_task:
                layer_eval = get_gpu_eval(layer_name, gpu_eval_names, proc_eval)
                task_eval += layer_eval
        restored_eval.append(task_eval)

    # print("restored_GPU_eval: ", restored_eval)
    return restored_eval


def get_gpu_eval(task_name, gpu_eval_names, proc_eval):
    """
    Get GPU evaluation for the task
    """
    for gpu_eval_id in range(0, len(gpu_eval_names)):
        gpu_eval_name = gpu_eval_names[gpu_eval_id]

        # evaluation name starts with task name : this task was executed on GPU and evaluated
        if gpu_eval_name.startswith(task_name):
            return proc_eval[gpu_eval_id]

        # evaluation name contains task name in middle/end part: this task was merged with other task and its time == 0
        if task_name in gpu_eval_name:
            return 0.0
    # nothing found: return zero-eval_table by default
    return 0.0

