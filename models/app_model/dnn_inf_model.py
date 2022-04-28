import functools # for custom-key comparison among objects
from models.dnn_model.dnn import DNN, ExternalInputConnection, ExternalOutputConnection
from models.app_model.InterDNNConnection import InterDNNConnection
from models.edge_platform.Architecture import Architecture
from models.TaskGraph import TaskGraph
from models.data_buffers import DataBuffer
from DSE.scheduling.dnn_scheduling import DNNScheduling
from dnn_partitioning.after_mapping.partition_dnn_with_mapping import DNNPartitioner


class DNNInferenceModel:
    """
    Final DNN inference execution model
    Attributes:
        schedule_type: type of schedule between partitions: sequential (one-by-one) or pipelined
        json_partitions: list of json descriptions (dictionaries), where each description corresponds to a
            dnn sub-network (partition)
        json_connections: list of json descriptions (dictionaries), where each description corresponds to a
            connection between dnn partitions
        inter_partition_buffers: list of buffers (each defined as an object of DataBuffer class) that store
            data exchanged between the dnn partitions during the application execution
    """
    def __init__(self, schedule_type: DNNScheduling,
                 partitions: [DNN],
                 connections: [InterDNNConnection],
                 inter_partition_buffers: [DataBuffer]):

        self.schedule_type = schedule_type
        self.json_partitions = partitions
        self.json_connections = connections
        self.inter_partition_buffers = inter_partition_buffers


def generate_dnn_inference_model(dnn: DNN,
                                 architecture: Architecture,
                                 task_graph: TaskGraph,
                                 mapping,
                                 schedule_type=DNNScheduling.PIPELINE):
    """
    Generate Final DNN inference execution model:
        :param dnn: DNN
        :param architecture: target platform architecture
        :param task_graph: dnn task graph
        :param mapping: mapping of dnn task graph into target platform architecture.
         An array mapping = proc_1_tasks, proc_2_tasks, ..., proc_n_tasks
         where proc_i_tasks = [task_id_i1, task_id_i2, ..., task_id_iMi] is a set of ids of tasks, executed on
         i-th processor of target edge platform; Mi is the total number of tasks, executed on
         i-th processor of target edge platform.
         :param schedule_type: type of schedule between CNN partitions:
            - sequential: partitions are executed one-by-one
            - pipeline: partitions are executed in a parallel pipelined fashion
        :return DNNInferenceModel class object
        """

    def _generate_partitions_description():
        partitions_desc = []
        partition_name_to_proc_id = partitioner.partition_name_to_proc_id
        for partition in sorted_partitions:
            name = partition.name
            processor_id = partition_name_to_proc_id[name]
            processor_type = architecture.processors_types[processor_id]
            json_partition = {"name": name,
                              "processor_id": processor_id,
                              "processor_type": processor_type,
                              "layers": [layer.name for layer in partition.get_layers()]}
            partitions_desc.append(json_partition)
        return partitions_desc

    def _generate_connections_description():
        connections_desc = []
        for connection in sorted_connections:
            json_connection = {"name": connection.name,
                               "src": connection.src.name,
                               "dst": connection.dst.name}
            connections_desc.append(json_connection)
        return connections_desc

    def _generate_inter_partition_buffers(start_buf_id=0):
        """
        Generate buffers between partitions
        :param start_buf_id:
        :return:
        """
        inter_partition_buf = []
        buf_id = start_buf_id

        for connection in sorted_connections:
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

    # generate partitions and connections between them
    partitioner = DNNPartitioner(dnn, task_graph, mapping)
    partitioner.partition()
    # print("  - DNN is partitioned")

    sorted_partitions = sorted(partitioner.get_partitions(),
                               key=functools.cmp_to_key(compare_partitions_by_pos_in_dnn))

    sorted_connections = sorted(partitioner.get_inter_partition_connections(),
                                key=functools.cmp_to_key(compare_connections_by_pos_in_dnn))

    # create .json description of partitions
    json_partitions = _generate_partitions_description()
    # print("  - Final app model partitions generated")

    # create .json description of connections between partitions
    json_connections = _generate_connections_description()

    # sort partitions descriptions by ids of layers within partitions

    # generate application buffers
    app_buffers = []
    # DNN input buffers
    # input_buffers = generate_external_input_buffers(dnn, sorted_partitions)
    # app_buffers = app_buffers + input_buffers
    # DNN output buffers
    # output_buffers = generate_external_output_buffers(dnn, sorted_partitions)
    # app_buffers = app_buffers + output_buffers
    # buffers among DNN partitions
    inter_partition_buffers = _generate_inter_partition_buffers()
    app_buffers = app_buffers + inter_partition_buffers

    dnn_inference_model = DNNInferenceModel(schedule_type,
                                            json_partitions,
                                            json_connections,
                                            app_buffers)
    return dnn_inference_model


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

