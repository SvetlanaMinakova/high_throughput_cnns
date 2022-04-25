def get_sum_proc_times(mapping, architecture, time_eval_matrix):
    """Get total execution time for every processor in architecture"""
    sum_proc_times = []
    proc_id = 0
    for proc_tasks in mapping:
        proc_type_id = architecture.get_proc_type_id(proc_id)
        sum_proc_time = get_sum_proc_time(time_eval_matrix, proc_tasks, proc_type_id)
        sum_proc_times.append(sum_proc_time)
        proc_id = proc_id + 1
    return sum_proc_times


def get_sum_proc_time(eval_table, proc_tasks, proc_type_id):
    """Get total processor execution time"""
    sum_time = 0

    for proc_task in proc_tasks:
        sum_time += eval_table[proc_type_id][proc_task]

    return sum_time


def get_pipeline_execution_proc_time(sum_proc_times):
    """ Get time, required for processors to run concurrently"""
    return max(sum_proc_times)

