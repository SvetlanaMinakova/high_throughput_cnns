import random


def get_moc_eval_table(app_graph, platform, acceleration_per_proc_type: {}):
    """
    Get moc evaluation table
    :param app_graph: application graph
    :param platform: platform
    :param acceleration_per_proc_type: acceleration per type of processor:
        a dictionary where key = name of distinct processor,
        value (int) = relative acceleration of processor, compared to the baseline (1)
        For all the processors, unspecified in acceleration_per_proc_type,
        acceleration = 1
    :return: mov evaluation table, generated for given application graph and platform randomly
    """
    eval_table = []
    proc_acceleration = 1
    proc_id = 0

    for proc_type in platform.processors_types_distinct:
        proc_eval = []
        if proc_type in acceleration_per_proc_type.keys():
            proc_acceleration = acceleration_per_proc_type[proc_type]

        for task in app_graph.tasks:
            moc_task_time = random.uniform(0, 1) / proc_acceleration
            proc_eval.append(moc_task_time)

        proc_id = proc_id + 1
        eval_table.append(proc_eval)

    return eval_table