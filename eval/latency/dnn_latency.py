def eval_latency_sequential(eval_table, architecture, mapping):
    """
    Eval latency of a DNN, executed sequentially
    :param eval_table: matrix [M]x[N] where m in [0, len (processor_types_distinct)-1] represents processor type,
        n in [0, layers_num] represents layer, matrix[m][n] contains execution time of layer n on processor m
    :param architecture: target platform architecture
    :return: latency of a DNN, executed sequentially
    """
    from DSE.eval_table.eval_dnn_latency_from_et import get_sum_proc_times
    times_per_proc = get_sum_proc_times(mapping, architecture, eval_table)

    # aggregate
    latency = sum(times_per_proc)
    return latency


def eval_latency_pipeline(eval_table, architecture, mapping):
    """
    Eval latency of a DNN, executed as a pipeline
    :param eval_table: matrix [M]x[N] where m in [0, len (processor_types_distinct)-1] represents processor type,
        n in [0, layers_num] represents layer, matrix[m][n] contains execution time of layer n on processor m
    :param architecture: target platform architecture
    :return: latency of a DNN, executed sequentially
    """
    from DSE.eval_table.eval_dnn_latency_from_et import get_sum_proc_times
    times_per_proc = get_sum_proc_times(mapping, architecture, eval_table)
    latency = max(times_per_proc)
    return latency

