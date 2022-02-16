from util import mega, milli
"""
Computation of communication penalty occurring during the CNN
execution on a multi-processor platform
"""


def get_processor_communication_penalty(proc_id,
                                        app_graph,
                                        architecture,
                                        mapping,
                                        token_bytes=4):
    """
    Get communication penalty for processor
    :param proc_id: id of the processor
    :param app_graph: application graph
    :param architecture = target platform
    :param mapping: mapping of the application graph on the target platform
    :param token_bytes: size of one token in bytes
    """
    # communication penalty in milliseconds (ms)
    communication_penalty = 0.0
    proc_tasks = mapping[proc_id]
    proc_type_id = architecture.get_proc_type_id(proc_id)
    proc_type = architecture.processors_types[proc_type_id]

    for proc_task in proc_tasks:
        # penalty for writing on other processors
        proc_tasks_outs = app_graph.tasks_adjacent_list[proc_task]
        for out in proc_tasks_outs:
            # out-task (task, receiving data from proc_task)
            # is mapped on a different processor
            if out not in proc_tasks:
                out_processor_id = get_task_processor_id(mapping, out)
                out_proc_type = architecture.get_proc_type_id(out_processor_id)
                bandwidth = architecture.get_communication_speed_mb_s(proc_type, out_proc_type)
                tokens = app_graph.tasks_out_comm_cost[proc_task]
                outp_write_penalty = get_memory_access_penalty_ms(tokens, token_bytes, bandwidth)
                communication_penalty += outp_write_penalty

        # penalty for reading from other processors
        proc_tasks_inp = app_graph.tasks_reverse_adjacent_list[proc_task]
        for inp in proc_tasks_inp:
            # in-task (task, producing data for proc_task)
            # is mapped on a different processor
            if inp not in proc_tasks:
                in_processor_id = get_task_processor_id(mapping, inp)
                in_proc_type = architecture.get_proc_type_id(in_processor_id)
                bandwidth = architecture.get_communication_speed_mb_s(in_proc_type, proc_type)
                tokens = app_graph.tasks_out_comm_cost[proc_task]
                inp_read_penalty = get_memory_access_penalty_ms(tokens, token_bytes, bandwidth)
                communication_penalty += inp_read_penalty

    return communication_penalty


def get_memory_access_penalty_ms(tokens, token_bytes, bandwidth_mb_s):
    """
    Get penalty for one memory access operation in milliseconds
    :param tokens: number of tokens written/red to memory during the operation
    :param token_bytes: size of one token in bytes
    :param bandwidth_mb_s: memory bandwidth in megabytes per second
    :return:
    """
    if bandwidth_mb_s == 0:
        return 0
    data_mb = (tokens * token_bytes / float(mega()))
    data_transfer_time_s = data_mb / bandwidth_mb_s
    data_transfer_time_ms = data_transfer_time_s / float(milli())
    return data_transfer_time_ms


def get_communication_penalty(app_graph, architecture, mapping):
    """
    Get communication penalty for exchanging data among processors
    :param app_graph:
    :param architecture: platform architecture, defined as an object of Architecture class
    :param mapping: mapping
    :return:
    """
    communication_penalty = 0.0
    for processor_id in range(len(mapping)):
        processor_com_penalty = get_processor_communication_penalty(processor_id, app_graph, architecture, mapping)
        communication_penalty += processor_com_penalty
    return communication_penalty


def get_task_processor_id(mapping, task_id):
    """
    Get id of the processor, where task is performed
    :param task_id: task id
    :param mapping of tasks on the processors
    :return: id of the processor, where task is performed
    """
    for proc_id in range(len(mapping)):
        processor_tasks = mapping[proc_id]
        if task_id in processor_tasks:
            return proc_id
    raise Exception("task " + str(task_id) + " not found in the mapping")

