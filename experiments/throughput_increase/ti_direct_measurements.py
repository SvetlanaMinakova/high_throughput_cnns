from util import milli

"""
Evaluate increase of a dnn throughput, using efficient dnn_partitioning, mapping and scheduling, and
Direct measurements based latency/throughput evaluation
The experiment corresponds to Chapter 3
"""


def increase_dnn_throughput(dnn_name, architecture, task_graph_path, eval_path, ga_conf_path=None):
    """
    Eval dnn throughput increase
    :param architecture target platform architecture
    :param dnn_name: dnn model name
    :param task_graph_path: path to DNN task graph, saved in json format
    :param eval_path path to direct measurements, performed for the application
    :param ga_conf_path: path to GA_based search config. If None, GA search is not performed
    :return:
    """

    # imports
    from high_throughput.eval_table.direct_measurements_et_builder import build_eval_table
    from converters.json_converters.json_task_graph import parse_app_graph_json
    from high_throughput.mapping.ga import GA
    from high_throughput.mapping.greedy_mapping import map_greedy
    from converters.json_converters.json_ga_conf_parser import parse_ga_conf
    from eval.latency.dnn_latency import eval_latency_sequential, eval_latency_pipeline
    from high_throughput.mapping.sequential import map_sequential

    # -------------
    # prepare data

    # build application graph
    dnn_task_graph = parse_app_graph_json(task_graph_path)
    # build time eval matrix
    eval_table = build_eval_table(eval_path, architecture.processors_types_distinct, dnn_task_graph)

    # --------------
    # sequential latency

    # default processor id for DNN execution: first accelerator (if available) or first CPU (otherwise)
    accelerator_id = architecture.get_first_accelerator_proc_id()
    accelerator_id = max(accelerator_id, 0)
    # sequential execution on one processor
    seq_mapping = map_sequential(len(dnn_task_graph.tasks), len(architecture.src_and_dst_processor_types), accelerator_id)
    lat_ms_sequential = eval_latency_sequential(eval_table, architecture, seq_mapping)
    thr_fps_sequential = lat_ms_to_thr_fps(lat_ms_sequential)

    # --------------
    # run greedy exploration
    greedy_mapping = map_greedy(dnn_task_graph, architecture, eval_table, accelerator_id, verbose=False)
    lat_ms_greedy = eval_latency_pipeline(eval_table, architecture, greedy_mapping)
    thr_fps_greedy = lat_ms_to_thr_fps(lat_ms_greedy)

    # improvement
    speed_up_perc_greedy = (lat_ms_sequential-lat_ms_greedy)/float(lat_ms_sequential) * 100.0

    # --------------
    # run GA-based exploration (optional)
    lat_ms_ga = thr_fps_ga = speed_up_perc_ga = 0.0
    ga_executed = False

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
                verbose=ga_conf["verbose"],
                preset_preferred_proc_probability=ga_conf["preset_preferred_proc_probability"],
                preferred_proc_id=ga_conf["preferred_proc_id"])
        ga.generate_random_population()
        ga_mapping = ga.run()
        lat_ms_ga = eval_latency_pipeline(eval_table, architecture, ga_mapping)
        thr_fps_ga = lat_ms_to_thr_fps(lat_ms_ga)

        # improvement
        speed_up_perc_ga = (lat_ms_sequential-lat_ms_ga)/float(lat_ms_ga) * 100.0

        ga_executed = True

    # ------------
    # print results
    print("*********************************************************************")
    print("model latency reduction (model =", dnn_name, ") : ")
    print()
    print(" - layer-by-layer on", architecture.src_and_dst_processor_types[accelerator_id])
    print("      latency: ", round(lat_ms_sequential, 2), "; throughput: ", round(thr_fps_sequential, 2), "fps")
    # print()
    print(" - greedy pipeline:")
    print("      latency: ", round(lat_ms_greedy, 2), "; throughput: ", round(thr_fps_greedy, 2), "fps",
          "; speed-up: ", round(speed_up_perc_greedy, 2), "%")
    if ga_executed:
        print(" - ga-based pipeline:")
        print("      latency: ", round(lat_ms_ga, 2), "; throughput: ", round(thr_fps_ga, 2), "fps",
              "; speed-up: ", round(speed_up_perc_ga, 2), "%")
    print()


def lat_ms_to_thr_fps(lat_ms):
    """ Convert latency in milliseconds (ms) into throughput in frames per second (fps)"""
    if lat_ms == 0:
        return 0
    thr = 1/(lat_ms * float(milli()))
    return thr





