import os.path
from util import print_stage


def supported_throughput_eval_types():
    return ["measurements", "flops"]


def supported_mapping_algorithms():
    return ["greedy", "ga"]


def map_and_partition(dnn,
                      architecture,
                      output_dir,
                      eval_type="flops",
                      map_algo="greedy",
                      task_graph_path=None,
                      eval_path=None,
                      ga_conf_path=None,
                      verbose=True):
    """
    Map and partition a DNN
    :param dnn: (analytical) dnn model
    :param architecture: target platform architecture
    :param eval_type: type of evaluation: FLOPs or measurements on the platform
    :param map_algo: string in ["greedy", "ga"]: algorithm to use for mapping and partitioning
    :param task_graph_path: path to DNN task graph, saved in JSON format. If == None,
        the DNN task graph will be built from the input DNN automatically
    :param eval_path path to direct measurements of per-layer DNN latency. Required for eval_type="measurements"
    :param ga_conf_path: path to GA config. If None, GA-based search (map_algo="ga"  cannot be performed)
    :param output_dir: if not None, all files created by the script, will be saved as .json files in this directory.
        - mapping of the dnn task graph onto processors of the target hardware architecture (platform)
        - partitioning of the dnn task graph
    :param verbose: print details
    :return: mapping (SDF) and partitioning (CSDF)
    """
    # imports
    from converters.json_converters.mapping_to_json import mapping_to_json
    from converters.json_converters.partitioning_to_json import partitioning_to_json

    # -------------
    if output_dir is None:
        raise Exception("NULL output directory")

    stage = "Create output directory if it does not exist"
    print_stage(stage, verbose)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # -------------
    # prepare data
    stage = "Read or build dnn task graph"
    print_stage(stage, verbose)
    dnn_task_graph = get_task_graph(dnn, task_graph_path, verbose)

    # build time eval matrix
    stage = "Read or build per-layer execution time (latency) eval table"
    print_stage(stage, verbose)
    eval_table = get_time_eval_table(dnn, dnn_task_graph, architecture, eval_type, eval_path, verbose)

    # --------------
    # get mapping
    stage = "Map dnn onto target hardware architecture (mapping algorithm = " + str(map_algo) + ")"
    print_stage(stage, verbose)
    mapping = get_mapping(dnn_task_graph, architecture, eval_table, map_algo, ga_conf_path, verbose)

    # save mapping as a .json file
    mapping_to_json(dnn_task_graph, architecture, mapping, output_dir, verbose)

    # --------------
    # get partitioning
    stage = "Partition mapped DNN (create final CSDF graph)"
    print_stage(stage, verbose)
    partitions, connections = get_partitioning(dnn, dnn_task_graph, mapping)

    # save partitioning as a .json file
    partitioning_to_json(dnn, architecture, mapping, partitions, connections, output_dir, verbose)


def get_task_graph(dnn, task_graph_path=None, verbose=False):
    """
    Get a task graph (CSDF) for a DNN
    :param dnn: DNN (analytical model)
    :param task_graph_path: path to DNN task graph, saved in JSON format. If == None,
        the DNN task graph will be built from the input DNN automatically
    :param verbose: flag. if True, print details
    :return: DNN task graph
    """
    # imports
    from converters.dnn_to_task_graph import dnn_to_task_graph_with_built_in, dnn_to_task_graph
    from converters.json_converters.json_task_graph import parse_task_graph_json

    if task_graph_path is None:
        if verbose:
            print("  - build task graph")
        task_graph = dnn_to_task_graph(dnn)
    else:
        if verbose:
            print("  - parse task graph", task_graph_path)
        task_graph = parse_task_graph_json(task_graph_path)
    return task_graph


def get_time_eval_table(dnn, dnn_task_graph, architecture, eval_type="flops", eval_path=None, verbose=False):
    """
    Get per-layer execution time (latency) evaluation matrix:
     matrix [M]x[N] where m in [0, len (processor_types_distinct)-1] represents processor type,
        n in [0, layers_num] represents layer, matrix[m][n] contains execution time of layer n on processor m
    :param dnn: DNN (analytical model)
    :param dnn_task_graph: dnn task graph (SDF)
    :param architecture: target platform architecture
    :param eval_type: type of evaluation: FLOPs or measurements on the platform
    :param eval_path path to direct measurements of per-layer DNN latency. required for eval_type="measurements". If == None,
        FLOPS-based latency/throughput evaluation will be used instead
    :param verbose: flag. if True, print details
    :return: per-layer execution time (latency) evaluation matrix
    """
    # imports
    from DSE.eval_table.flops_et_builder import build_flops_time_eval_table
    from DSE.eval_table.direct_measurements_et_builder import build_eval_table

    if eval_type == "measurements":
        if eval_path is None:
            raise Exception("Eval path is None. Throughput evaluation "
                            "using eval_type=measurements requires an explicitly specified eval path.")
        else:
            if verbose:
                print("  - parse eval table")
            eval_table = build_eval_table(eval_path, architecture.processors_types_distinct, dnn_task_graph)
            return eval_table

    if eval_type == "flops":
        if verbose:
            print("  - build FLOPS-based eval table")
        eval_table = build_flops_time_eval_table(dnn, dnn_task_graph, architecture, verbose=False)
        return eval_table

    raise Exception("Unknown eval type " + eval_type + ". Please choose from " + str(supported_throughput_eval_types()))


def get_mapping(dnn_task_graph, architecture, eval_table, map_algo="greedy", ga_conf_path=None, verbose=False):
    """
    Map DNN task graph onto target hardware architecture
    :param dnn_task_graph: dnn task graph (SDF)
    :param architecture: target platform architecture
    :param eval_table: Get per-layer execution time (latency) evaluation matrix:
     matrix [M]x[N] where m in [0, len (processor_types_distinct)-1] represents processor type,
        n in [0, layers_num] represents layer, matrix[m][n] contains execution time of layer n on processor m
    :param map_algo: string in ["greedy", "ga"]: algorithm to use for mapping and partitioning
    :param ga_conf_path: path to GA config. If None, GA-based search (map_algo="ga"  cannot be performed)
    :param verbose: flag. if True, print details
    :return: per-layer execution time (latency) evaluation matrix
    """
    # imports
    from DSE.mapping.ga import GA
    from DSE.mapping.greedy_mapping import map_greedy
    from converters.json_converters.json_ga_conf_parser import parse_ga_conf

    if map_algo == "greedy":
        # run greedy mapping exploration
        accelerator_id = architecture.get_first_accelerator_proc_id()
        accelerator_id = max(accelerator_id, 0)
        greedy_mapping = map_greedy(dnn_task_graph, architecture, eval_table, accelerator_id, verbose)
        return greedy_mapping

    if map_algo == "ga":
        # run GA-based mapping exploration
        if ga_conf_path is not None:
            ga_conf = parse_ga_conf(ga_conf_path)

            ga = GA(dnn_task_graph, architecture, eval_table,
                    epochs=ga_conf["epochs"],
                    population_start_size=ga_conf["population_start_size"],
                    selection_percent=ga_conf["selection_percent"],
                    mutation_probability=ga_conf["mutation_probability"],
                    mutation_percent=ga_conf["mutation_percent"],
                    max_no_improvement_epochs=ga_conf["max_no_improvement_epochs"],
                    eval_communication=ga_conf["eval_communication_costs"],
                    verbose=verbose,
                    preset_preferred_proc_probability=ga_conf["preset_preferred_proc_probability"],
                    preferred_proc_id=ga_conf["preferred_proc_id"])
            ga.generate_random_population()
            ga_mapping = ga.run()
            return ga_mapping
        else:
            raise Exception("GA-based (map_algo=ga) mapping error: null GA configuration path")

    raise Exception("Unsupported map_algo: " + str(map_algo) + ". Please choose from " +
                    str(supported_mapping_algorithms()))


def get_partitioning(dnn, dnn_task_graph, mapping):
    """
    Partition dnn according to the task graph, generated from this DNN
    :param dnn: dnn, represented as (analysis) dnn model
    :param dnn_task_graph: task graph, generated from this DNN
    :param mapping: [tasks_per_processor[i]], i in [1, len(processors_num)], where
     processors_num is the total number of processors available on the platform,
     tasks_per_processor[i] = [task_1, task_2, ..., task_t] is the list of tasks mapped on the processor i,
     where task_t is id of task in the task graph

    :return: tuple: partitions, connections where:
        partitions is a  list of dnn partitions, where every partition is a DNN, that is
        a sub-graph of the original dnn, and all partitions together represent
        functionality of the original DNN
        connections: connections between the DNN partitions
    """
    from dnn_partitioning.after_mapping.partition_dnn_with_mapping import partition_dnn_with_task_graph_and_mapping
    partitions, connections = partition_dnn_with_task_graph_and_mapping(dnn, dnn_task_graph, mapping)
    return partitions, connections
