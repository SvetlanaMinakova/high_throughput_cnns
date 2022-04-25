from models.dnn_model.dnn import DNN
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
                 inter_partition_buffers):

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
        for partition in partitioner.get_partitions():
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
        for connection in partitioner.get_inter_partition_connections():
            json_connection = {"name": connection.name,
                               "src": connection.src.name,
                               "dst": connection.dst.name}
            connections_desc.append(json_connection)
        return connections_desc

    def _init_inter_partition_buffers(inter_partition_buf):
        if schedule_type == DNNScheduling.PIPELINE:
            _init_inter_partition_buffers_pipeline(inter_partition_buf)
        else:
            _init_inter_partition_buffers_sequential(inter_partition_buf)

    def _init_inter_partition_buffers_pipeline(inter_partition_buf):
        """
        Generate buffers for pipelined application
        in case of pipeline schedule, every connection is associated with two buffers:
        an input buffer for data-consuming partition and an output buffer for data-producing partition
        """
        inter_partition_buf.clear()
        for connection in partitioner.get_inter_partition_connections():
            # double-buffer
            buffer_name = "B" + str(len(inter_partition_buf))
            buffer_size = connection.data_w * connection.data_h * connection.data_ch
            buffer = DataBuffer(buffer_name, buffer_size)
            buffer.users.append(connection.name)
            buffer.type = "double_buffer"
            inter_partition_buffers.append(buffer)

    def _init_inter_partition_buffers_sequential(inter_partition_buf):
        """
        Generate buffers for sequential application
        in case of sequential schedule, every connection is associated with a
        single buffer, which serves as an input to data-consuming partition and as an
        output buffer for data-producing partition
        """
        inter_partition_buf.clear()

        for connection in partitioner.get_inter_partition_connections():
            # single-buffer
            buffer_name = "B" + str(len(inter_partition_buf))
            buffer_size = connection.data_w * connection.data_h * connection.data_ch
            buffer = DataBuffer(buffer_name, buffer_size)
            buffer.users.append(connection.name)
            buffer.type = "single_buffer"
            inter_partition_buffers.append(buffer)

    # generate partitions and connections between them
    partitioner = DNNPartitioner(dnn, task_graph, mapping)
    partitioner.partition()
    # print("  - DNN is partitioned")

    # describe every partition (task)
    partitions = _generate_partitions_description()
    # print("  - Final app model partitions generated")

    # create .json connections between partitions
    connections = _generate_connections_description()

    # generate external buffers
    inter_partition_buffers = []
    _init_inter_partition_buffers(inter_partition_buffers)

    dnn_inference_model = DNNInferenceModel(schedule_type, partitions, connections, inter_partition_buffers)
    return dnn_inference_model

