import functools # for custom-key comparison among objects
from models.dnn_model.dnn import DNN
from models.app_model.InterDNNConnection import InterDNNConnection
from models.edge_platform.Architecture import Architecture
from models.TaskGraph import TaskGraph
from models.data_buffers.data_buffers import DataBuffer
from DSE.scheduling.dnn_scheduling import DNNScheduling
from DSE.partitioning.after_mapping.partition_dnn_with_mapping import DNNPartitioner
from DSE.buffers_generation.inter_dnn_buffers_builder import generate_inter_partition_buffers
from DSE.buffers_generation.inter_dnn_buffers_builder import sort_partitions_by_pos_in_dnn
from DSE.buffers_generation.inter_dnn_buffers_builder import sort_inter_dnn_connections_by_pos_in_dnn


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

    ########################
    # main part of the script

    # generate partitions and connections between them
    partitioner = DNNPartitioner(dnn, task_graph, mapping)
    partitioner.partition()
    # print("  - DNN is partitioned")

    sorted_partitions = sort_partitions_by_pos_in_dnn(dnn, partitioner.get_partitions())
    sorted_connections = sort_inter_dnn_connections_by_pos_in_dnn(dnn, partitioner.get_inter_partition_connections())

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
    inter_partition_buffers = generate_inter_partition_buffers(sorted_connections, schedule_type)
    app_buffers = app_buffers + inter_partition_buffers

    dnn_inference_model = DNNInferenceModel(schedule_type,
                                            json_partitions,
                                            json_connections,
                                            app_buffers)
    return dnn_inference_model


