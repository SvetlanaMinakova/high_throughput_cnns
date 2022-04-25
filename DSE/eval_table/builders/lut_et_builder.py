from eval.latency.lut.LUT_builder import build_lut_tree_from_jetson_benchmark, LUTTree


def build_time_eval_matrix_from_lut(dnn, app_graph, architecture, lut_file_per_proc_type: {}, verbose=True):
    """
    Build time evaluation matrix from a look-up table (LUT)
    :param dnn: deep neural network to evaluate, represented as internal dnn model
    :param app_graph application graph
    :param lut_file_per_proc_type: dictionary, where key = lut file name,
     value= path to lut file with on-board per-layer execution time measurements for the specified processor type
    :param architecture target platform architecture
    Otherwise, dnn is scheduled sequentially
    :param verbose: verbose
    :return: time_eval_matrix:
        # matrix [M]x[N] where m in [0, len (processor_types_distinct)-1] represents processor type,
        # n in [0, layers_num] represents layer, matrix[m][n] contains execution time of layer n on processor m
    """
    if lut_file_per_proc_type is None:
        if verbose:
            print("Throughput eval_table: no LUT is specified as input_examples. Random execution times are used.")

        from DSE.eval_table.builders.moc_et_builder import get_moc_eval_table
        eval_matrix = get_moc_eval_table(app_graph, architecture, acceleration_per_proc_type={})
        return eval_matrix

    # build lookup tables per processor
    luts = build_luts_per_processor(lut_file_per_proc_type)

    # init eval_table matrix
    eval_matrix = []
    # dummy execution times in case luts are missing for some of the processors
    moc_gpu_time = 1

    for processor_type_id in range(len(architecture.processors_types_distinct)):
        proc_type = architecture.processors_types_distinct[processor_type_id]
        if proc_type not in luts.keys():
            dummy_time = moc_gpu_time
            if verbose:
                print("Throughput eval_table: no LUT file found for processor '" + proc_type +
                      "' Dummy exec. time of " + str(dummy_time) + " ms is used instead.")
            proc_eval = create_dummy_eval_matrix_row(len(dnn.get_layers()), dummy_time)

        else:
            lut = luts[proc_type]
            proc_eval = create_eval_matrix_row_from_lut(dnn, lut, verbose)

        eval_matrix.append(proc_eval)

    return eval_matrix


def build_luts_per_processor(lut_file_per_proc_type: {}):
    """
    Build lookup tables per processor
    :param lut_file_per_proc_type: dictionary, where key = lut file name,
     value= path to lut file with on-board per-layer execution time measurements for the specified processor type
    :return:
    """
    lut_per_proc = {}
    for item in lut_file_per_proc_type.items():
        proc_type, lut_file = item
        lut = build_lut_tree_from_jetson_benchmark(lut_file)
        lut_per_proc[proc_type] = lut
    return lut_per_proc


def create_eval_matrix_row_from_lut(dnn, lut:LUTTree, verbose=True):
    """
    Create a dummy row for eval_table matrix
    :param lut: lookup table, represented as a tree and containing per-layer performance estimation of CNN layer,
    executed on a target platform
    :param verbose: verbose
    :return: dummy row for eval_table matrix
    """
    eval_row = []
    # TODO: extend!
    supported_ops = ["conv", "gemm"]
    for layer in dnn.get_layers():
        layer_time = 0
        if layer.op in supported_ops:
            lut_tree_node = lut.find_lut_tree_node(layer)
            if lut_tree_node is None:
                if verbose:
                    print("Throughput eval_table: cannot find record for layer: ", layer)
            else:
                layer_time = lut_tree_node.val
        eval_row.append(layer_time)

    return eval_row


def create_dummy_eval_matrix_row(row_len, dummy_value):
    """
    Create a dummy row for eval_table matrix
    :return: dummy row for eval_table matrix
    """
    dummy_row = [dummy_value for _ in range(row_len)]
    return dummy_row


def create_empty_eval_matrix(architecture, app_graph, gpu_type_id):
    """
    Create an empty eval_table matrix, initialized with dummy values
    :param app_graph application graph
    :param architecture target platform architecture
    Otherwise, dnn is scheduled sequentially
    :param gpu_type_id id of accelerator distinct processor type
    :return:
    """
    eval_matrix = []
    moc_cpu_time = 1000
    moc_gpu_time = 0

    proc_id = 0
    for proc_type in architecture.processors_types_distinct:
        if proc_id == gpu_type_id:
            init_time = moc_gpu_time
        else:
            init_time = moc_cpu_time
        proc_eval = []

        for task in app_graph.layers:
            moc_task_time = init_time
            proc_eval.append(moc_task_time)

        proc_id = proc_id + 1
        eval_matrix.append(proc_eval)
    return eval_matrix

