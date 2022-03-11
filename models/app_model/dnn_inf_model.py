from models.dnn_model.dnn import DNN
from models.edge_platform.Architecture import Architecture
from models.TaskGraph import TaskGraph
from models.data_buffers import DataBuffer
from DSE.scheduling.dnn_scheduling import DNNScheduling
from dnn_partitioning.after_mapping.partition_dnn_with_mapping import DNNPartitioner


class DNNInferenceModel:
    """
    Final DNN inference execution model
    Attributes:
        schedule type: type of schedule between partitions: sequential (one-by-one) or pipelined
    """
    def __init__(self, schedule_type: DNNScheduling, partitions, connections, inter_partition_buffers):
        """
        # TODO: implement buffers reuse
        """
        self.schedule_type = schedule_type
        self.partitions = partitions
        self.connections = connections
        self.inter_partition_buffers = inter_partition_buffers


def generate_dnn_inference_model(dnn: DNN,
                                 architecture: Architecture,
                                 task_graph: TaskGraph,
                                 mapping,
                                 schedule_type=DNNScheduling.PIPELINE,
                                 reuse_buffers=True):
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
        :param reuse_buffers (flag): if True, buffers that store data between CNN partitions
        will be reused. Otherwise, every partition will be allocated its own buffer
        # TODO: implement buffers reuse
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

    def _init_inter_partition_buffers():
        if schedule_type == DNNScheduling.PIPELINE:
            _init_inter_partition_buffers_pipeline()
        else:
            _init_inter_partition_buffers_sequential()

    def _init_inter_partition_buffers_pipeline():
        """
        Generate buffers for pipelined application
        in case of pipeline schedule, every connection is associated with two buffers:
        an input buffer for data-consuming partition and an output buffer for data-producing partition
        """
        naive_buffers = []
        # inter-DNN connection
        for connection in partitioner.get_inter_partition_connections():
            # double-buffer
            for i in range(2):
                buffer_name = "B" + str(len(naive_buffers))
                buffer_size = connection.data_w * connection.data_h * connection.data_ch
                data_buffer = DataBuffer(buffer_name, buffer_size)
                data_buffer.users.append(connection.name)
                naive_buffers.append(data_buffer)

        # TODO: buffers reuse

        #
        for data_buffer_id in range(len(naive_buffers)):
            data_buffer = naive_buffers[data_buffer_id]
            buffer_subtype = "out" if data_buffer_id % 2 == 0 else "in"
            buffer_desc = {"name": data_buffer.name,
                           "size_tokens": data_buffer.size,
                           "users": data_buffer.users,
                           "type": "double_buffer",
                           "subtype": buffer_subtype
                           }
            inter_partition_buffers.append(buffer_desc)

    def _init_inter_partition_buffers_sequential():
        """
        Generate buffers for sequential application
        in case of sequential schedule, every connection is associated with a
        single buffer, which serves as an input to data-consuming partition and as an
        output buffer for data-producing partition
        """
        naive_buffers = []

        for connection in partitioner.get_inter_partition_connections():
            # single-buffer
            buffer_name = "B" + str(len(naive_buffers))
            buffer_size = connection.data_w * connection.data_h * connection.data_ch
            data_buffer = DataBuffer(buffer_name, buffer_size)
            data_buffer.users.append(connection.name)
            naive_buffers.append(data_buffer)

        # TODO: buffers reuse

        #
        for data_buffer in naive_buffers:
            buffer_desc = {"name": data_buffer.name,
                           "size_tokens": data_buffer.size,
                           "users": data_buffer.users,
                           "type": "single_buffer",
                           "subtype": "none"
                           }
            inter_partition_buffers.append(buffer_desc)

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
    _init_inter_partition_buffers()

    dnn_inference_model = DNNInferenceModel(schedule_type, partitions, connections, inter_partition_buffers)
    return dnn_inference_model
