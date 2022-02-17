from fileworkers.json_fw import save_as_json
import os


def mapping_to_json(task_graph, architecture, mapping, output_dir, verbose):
    """
    Save mapping of a DNN onto target edge platform as a .json file
    :param task_graph: task-graph of a DNN
    :param architecture: target hardware platform
    :param mapping: mapping of the DNN onto th target hardware platform
    :param output_dir output files directory
    :param verbose: flag. if True, print details
    :return:
    """
    output_file_path = str(os.path.join(output_dir, "mapping.json"))

    if verbose:
        print("  - save dnn mapping into", output_file_path)

    mapping_as_dict = {}
    # meta-data
    tasks = task_graph.tasks
    processors = architecture.processors

    for proc_id in range(len(processors)):
        proc_name = processors[proc_id]
        task_ids = mapping[proc_id]
        task_ids_sorted = sorted(task_ids)
        # task_names = [tasks[task_id] for task_id in task_ids_sorted]
        layers_per_proc = []
        for task_id in task_ids_sorted:
            for job in task_graph.jobs_per_task[task_id]:
                layers_per_proc.append(job)

        mapping_as_dict[proc_name] = layers_per_proc

    save_as_json(output_file_path, mapping_as_dict)