from eval.latency.flops_based.layer_perf_estimator import eval_layer_latency_ms
from models.TaskGraph import task_name_to_layer_ids


def build_flops_time_eval_table(dnn, app_graph, architecture, verbose=True):
    """
    Build time evaluation matrix using flops-based latency evaluation
    :param dnn: deep neural network to evaluate, represented as internal dnn model
    :param app_graph application graph
    :param architecture target platform architecture
    :param verbose: verbose
    :return: time_eval_matrix:
        matrix [M]x[N] where m in [0, len (processor_types_distinct)-1] represents processor type,
        n in [0, layers_num] represents layer, matrix[m][n] contains execution time of layer n on processor m
    """

    layers = dnn.get_layers()

    eval_table = []
    proc_id = 0

    for proc_type_id in range(len(architecture.processors_types_distinct)):
        proc_eval = []
        proc_giga_flops = architecture.get_max_giga_flops_for_proc_type(proc_id)

        for task in app_graph.tasks:
            task_time = 0
            layer_ids = task_name_to_layer_ids(task)
            for layer_id in layer_ids:
                layer = layers[layer_id]
                layer_exec_time = eval_layer_latency_ms(layer, proc_giga_flops)
                task_time += layer_exec_time
            proc_eval.append(task_time)

        proc_id = proc_id + 1
        eval_table.append(proc_eval)

    return eval_table

