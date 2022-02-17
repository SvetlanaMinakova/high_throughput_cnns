""" Sequential mapping"""


def map_sequential(tasks_num, processors_num, proc_id):
    """
    Get simple mapping, where  all tasks are mapped on a single
    processor and executed sequentially
    """
    mapping = [[] for proc in range(processors_num)]
    for task_id in range(tasks_num):
        mapping[proc_id].append(task_id)
    return mapping

