import os


def partitioning_to_json(dnn, architecture, mapping, partitions, connections, output_dir, verbose):
    """
    Save mapped DNN partitioning as a .json file
    :param dnn: DNN (analytical model)
    :param architecture: target hardware platform
    :param mapping: mapping of the DNN onto th target hardware platform
    :param partitions: list of dnn partitions, where every partition is a DNN, that is
        a sub-graph of the original dnn, and all partitions together represent
        functionality of the original DNN
    :param connections: connections between the DNN partitions
    :param output_dir output files directory
    :param verbose: flag. if True, print details
    :return:
    """
    from fileworkers.json_fw import save_as_json
    output_file_path = str(os.path.join(output_dir, "partitioning.json"))

    if verbose:
        print("  - save partitioning (CSDF) of", dnn.name, "in", output_file_path)

    # meta-data
    processors = architecture.src_and_dst_processor_types
    json_partitions = []
    json_connections = []

    # create .json partitions
    for partition in partitions:
        json_partition = {"name": partition.name,
                          "layers": [layer.name for layer in partition.get_layers()]}
        json_partitions.append(json_partition)

    # create .json connections between partitions
    for connection in connections:
        json_connection = {"name": connection.name,
                           "src": connection.src.name,
                           "dst": connection.dst.name}
        json_connections.append(json_connection)

    partitioning_as_dict = {"partitions": json_partitions,
                            "connections": json_connections}

    save_as_json(output_file_path, partitioning_as_dict)
