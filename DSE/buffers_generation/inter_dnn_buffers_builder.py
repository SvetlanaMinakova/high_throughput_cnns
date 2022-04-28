import functools # for custom-key comparison among objects
from models.dnn_model.dnn import DNN
from models.app_model.InterDNNConnection import InterDNNConnection
from models.data_buffers.data_buffers import DataBuffer
from DSE.scheduling.dnn_scheduling import DNNScheduling


def generate_inter_partition_buffers(connections: [InterDNNConnection],
                                     schedule_type=DNNScheduling.PIPELINE,
                                     start_buf_id=0):
    """
    Generate buffers (areas of memory that store data) to store data exchanged between DNN partitions
        :param connections: connections between the DNN sub-networks (partitions)
        :param schedule_type: type of schedule between CNN partitions:
            - sequential: partitions are executed one-by-one
            - pipeline: partitions are executed in a parallel pipelined fashion
        :param start_buf_id: start buffer id, by default 0. All buffers generated
            by this function will be assigned ids starting from the specified start_buf_id
        :return DNNInferenceModel class object
        """

    inter_partition_buf = []
    buf_id = start_buf_id

    for connection in connections:
        buffer_name = "B" + str(buf_id)
        buffer_size = connection.data_w * connection.data_h * connection.data_ch
        buffer = DataBuffer(buffer_name, buffer_size)
        buffer.users.append(connection.name)
        # buffer type
        # in case of sequential schedule, buffer is a single-buffer (a simple area of memory)
        # which, at every moment in time can be used either for reading or for writing
        # in case of pipeline schedule, every connection is associated with a double-buffer:
        # a special composition of two single buffers,
        # which enables for overlapping reading and writing
        buffer.type = "double_buffer" if schedule_type == DNNScheduling.PIPELINE else "single_buffer"
        buffer.subtype = "io_buffer"
        inter_partition_buf.append(buffer)
        buf_id += 1

    return inter_partition_buf


def sort_partitions_by_pos_in_dnn(dnn, partitions: [DNN]):
    """
    Sort DNN partitions in order of partitions' definition within the DNN
    :param dnn: DNN
    :param partitions: DNN partitions (sub-networks)
    :return: partitions sorted in order of their definition within
    the DNN topology (from input layer to output layer)
    """

    def get_layer_id_in_dnn(layer_name):
        layer = dnn.find_layer_by_name(layer_name)
        layer_id = -1 if layer is None else layer.id
        return layer_id

    def compare_partitions_by_pos_in_dnn(partition1: DNN, partition2: DNN):
        """
        Compare two dnn partitions by their position in the dnn
        :param partition1: first partition
        :param partition2: second partition
        :return: -1, 1 or 0 depending on the results of comparison
        """
        partition1_start_layer_id = get_layer_id_in_dnn(partition1.get_layers()[0].name)
        partition2_start_layer_id = get_layer_id_in_dnn(partition2.get_layers()[0].name)

        if partition1_start_layer_id < partition2_start_layer_id:
            return -1

        if partition1_start_layer_id > partition2_start_layer_id:
            return 1

        # partition1_start_layer_id == partition2_start_layer_id

        partition1_end_layer_id = get_layer_id_in_dnn(partition1.get_layers()[-1].name)
        partition2_end_layer_id = get_layer_id_in_dnn(partition2.get_layers()[-1].name)

        if partition1_end_layer_id < partition2_end_layer_id:
            return -1

        if partition1_end_layer_id > partition2_end_layer_id:
            return 1

        # partition1_start_layer_id == partition2_start_layer_id and partition1_end_layer_id == partition2_end_layer_id
        return 0

    ########################
    # main part of the script
    sorted_partitions = sorted(partitions,
                               key=functools.cmp_to_key(compare_partitions_by_pos_in_dnn))

    return sorted_partitions


def sort_inter_dnn_connections_by_pos_in_dnn(dnn, connections: [InterDNNConnection]):
    """
    Sort connections between partitions of a DNN
    in order of connections definition within the partitioned DNN topology
    :param dnn: DNN
    :param connections: connections between the DNN partitions (sub-networks)
    :return: connections between the DNN partitions, sorted
        in order of their definition within the partitioned DNN topology
    """

    def get_layer_id_in_dnn(layer_name):
        layer = dnn.find_layer_by_name(layer_name)
        layer_id = -1 if layer is None else layer.id
        return layer_id

    def compare_partitions_by_pos_in_dnn(partition1: DNN, partition2: DNN):
        """
        Compare two dnn partitions by their position in the dnn
        :param partition1: first partition
        :param partition2: second partition
        :return: -1, 1 or 0 depending on the results of comparison
        """
        partition1_start_layer_id = get_layer_id_in_dnn(partition1.get_layers()[0].name)
        partition2_start_layer_id = get_layer_id_in_dnn(partition2.get_layers()[0].name)

        if partition1_start_layer_id < partition2_start_layer_id:
            return -1

        if partition1_start_layer_id > partition2_start_layer_id:
            return 1

        # partition1_start_layer_id == partition2_start_layer_id

        partition1_end_layer_id = get_layer_id_in_dnn(partition1.get_layers()[-1].name)
        partition2_end_layer_id = get_layer_id_in_dnn(partition2.get_layers()[-1].name)

        if partition1_end_layer_id < partition2_end_layer_id:
            return -1

        if partition1_end_layer_id > partition2_end_layer_id:
            return 1

        # partition1_start_layer_id == partition2_start_layer_id and partition1_end_layer_id == partition2_end_layer_id
        return 0

    def compare_connections_by_pos_in_dnn(connection1: InterDNNConnection, connection2: InterDNNConnection):
        # compare by source
        src_comparison = compare_partitions_by_pos_in_dnn(connection1.src, connection2.src)
        if src_comparison != 0:
            return src_comparison

        # if partitions are equal by source, compare by destination
        dst_comparison = compare_partitions_by_pos_in_dnn(connection1.dst, connection2.dst)
        return dst_comparison

    ########################
    # main part of the script

    sorted_connections = sorted(connections,
                                key=functools.cmp_to_key(compare_connections_by_pos_in_dnn))

    return sorted_connections


def generate_external_input_buffers(dnn, partitions=None):
    external_input_buffers = []
    external_input_id = 0
    buffer_subtype = "input_buffer"
    buffer_prefix = "B_in"
    if len(dnn.get_inputs()) > 0:
        for external_input in dnn.get_inputs():
            buffer_name = buffer_prefix + str(external_input_id)
            user_partition = find_partition_for_layer(dnn, external_input.dnn_layer.name, partitions)
            buffer = generate_buffer_for_external_io(external_input.data_layer,
                                                     buffer_name,
                                                     buffer_subtype,
                                                     user_partition)
            external_input_buffers.append(buffer)
            external_input_id += 1
    else:
        dnn_layers = dnn.get_layers()
        if len(dnn_layers) > 0:
            buffer_name = buffer_prefix + str(external_input_id)
            input_layer = dnn.get_layers()[0]
            user_partition = find_partition_for_layer(dnn, input_layer.name, partitions)
            buffer = generate_io_buffer_for_non_data_layer(input_layer, buffer_name, buffer_subtype, user_partition)
            external_input_buffers.append(buffer)

    return external_input_buffers


def generate_external_output_buffers(dnn, partitions=None):
    """
    Generate buffers that store data produced to/consumed from
        a dnn by external data sources/sinks
    :param dnn: DNN
    :param partitions: DNN partitions (for partitioned DNN)
    :return: buffers that store data produced to/consumed from
        a dnn by external data sources/sinks
    """
    external_output_buffers = []
    external_output_id = 0
    buffer_subtype = "output_buffer"
    buffer_prefix = "B_out"
    if len(dnn.get_outputs()) > 0:
        for external_output in dnn.get_outputs():
            buffer_name = buffer_prefix + str(external_output_id)
            user_partition = find_partition_for_layer(dnn, external_output.dnn_layer.name, partitions)
            buffer = generate_buffer_for_external_io(external_output.data_layer,
                                                     buffer_name,
                                                     buffer_subtype,
                                                     user_partition)
            external_output_buffers.append(buffer)
            external_output_id += 1
    else:
        dnn_layers = dnn.get_layers()
        if len(dnn_layers) > 0:
            buffer_name = buffer_prefix + str(external_output_id)
            output_layer = dnn.get_layers()[-1]
            user_partition = find_partition_for_layer(dnn, output_layer.name, partitions)
            buffer = generate_io_buffer_for_non_data_layer(output_layer, buffer_name, buffer_subtype, user_partition)
            external_output_buffers.append(buffer)

    return external_output_buffers


def find_partition_for_layer(dnn, layer_name, partitions=None):
    if partitions is None:
        return dnn.name
    for partition in partitions:
        if partition.find_layer_by_name(layer_name) is not None:
            return partition.name
    return dnn.name


def generate_buffer_for_external_io(data_layer, name, subtype, user_partition):
    buffer_size = data_layer.oh * data_layer.ow * data_layer.ofm
    buffer = DataBuffer(name, buffer_size)
    buffer.users.append(user_partition)
    buffer.type = "single_buffer"
    buffer.subtype = subtype
    return buffer


def generate_io_buffer_for_non_data_layer(dnn_layer, name, subtype, user_partition):
    if subtype == "input_buffer":
        buffer_size = dnn_layer.ih * dnn_layer.iw * dnn_layer.ifm
    else:
        buffer_size = dnn_layer.oh * dnn_layer.ow * dnn_layer.ofm
    buffer = DataBuffer(name, buffer_size)
    buffer.users.append(user_partition)
    buffer.type = "single_buffer"
    buffer.subtype = subtype
    return buffer
