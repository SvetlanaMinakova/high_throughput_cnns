import json


def get_latency_per_subnet(json_eval, dnn_inf_model):
    """
    Get latency per dnn inference model sub-network
    :param json_eval: json file with measurements on the platform
    :param dnn_inf_model: dnn inference (CSDF) model
    :return: dictionary, where key (string) = name of the dnn inference model sub-network,
        value (float) = estimated latency (execution time) of the sub-network
    """
    latency_per_subnet = {}
    with open(json_eval, 'r') as file:
        if file is None:
            raise FileNotFoundError

        json_eval = json.load(file)

        subnet_id = 0
        for subnet in dnn_inf_model.json_partitions:
            subnet_time = 0
            proc_type = subnet["processor_type"]

            if proc_type not in json_eval:
                raise Exception(proc_type + " latency evaluation not found!")

            for layer_name in subnet["layers"]:
                l_time = get_layer_latency(json_eval, layer_name, proc_type)
                subnet_time += l_time

            latency_per_subnet[subnet["name"]] = subnet_time
            subnet_id += 1

    return latency_per_subnet


def get_layer_latency(json_eval, layer_name, proc_type):
    """
    Get execution time (latency) of a layer
    :param json_eval: per-task dnn latency evaluation in .json format
    :param layer_name: name of the layer
    :param proc_type: type of the processor, where layer is executed
    :return:
    """
    for l_id in range(len(json_eval["layers"])):
        l_name = json_eval["layers"]
        if layer_name == l_name or layer_name in l_name:
            l_time = json_eval[proc_type][l_id]
            return l_time
    return 0
