from fileworkers.json_fw import save_as_json
import os


def mapping_to_json(dnn, architecture, mapping, output_dir, verbose):
    """
    Save mapping of a DNN onto target edge platform as a .json file
    :param dnn: DNN (analytical model)
    :param architecture: target hardware platform
    :param mapping: mapping of the DNN onto th target hardware platform
    :param output_dir output files directory
    :param verbose: flag. if True, print details
    :return:
    """
    output_file_path = str(os.path.join(output_dir, "mapping.json"))

    if verbose:
        print("  - save mapping of", dnn.name, "dnn onto target edge platform in", output_file_path)

    mapping_as_dict = {}
    # meta-data
    layers = dnn.get_layers()
    processors = architecture.src_and_dst_processor_types

    for proc_id in range(len(processors)):
        proc_name = processors[proc_id]
        task_ids = mapping[proc_id]
        task_ids_sorted = sorted(task_ids)
        task_names = [layers[task_id].name for task_id in task_ids_sorted]
        mapping_as_dict[proc_name] = task_names

    save_as_json(output_file_path, mapping_as_dict)