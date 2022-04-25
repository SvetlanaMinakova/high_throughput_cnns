from models.app_model.InterDNNConnection import InterDNNConnection
from models.dnn_model.dnn import DNN
from models.app_model.dnn_inf_model import DNNInferenceModel
import copy


def partition_dnn_with_dnn_inference_model(dnn: DNN, dnn_inf_model: DNNInferenceModel):
    """
    Partition dnn according to the task graph, generated from this DNN
    :param dnn: dnn, represented as (analysis) dnn model
    :param dnn_inf_model: DNN inference model that specifies
        partitioning, mapping and scheduling of the dnn on the target platform

    :return: tuple: partitions, connections where:
        partitions is a  list of dnn partitions, where every partition is a DNN, that is
        a sub-graph of the original dnn, and all partitions together represent
        functionality of the original DNN
        connections: connections between the DNN partitions
    """
    partitioner = DNNPartitioner(dnn, dnn_inf_model)
    partitioner.partition()
    partitions = partitioner.get_partitions()
    connections = partitioner.get_inter_partition_connections()
    return partitions, connections


class DNNPartitioner:
    """
    Partitions DNN into sub-graphs (partitions)
    according to the task graph, created from the DNN
    """
    def __init__(self, dnn: DNN, dnn_inf_model):
        self.dnn = dnn
        self.layers = dnn.get_layers()
        self.dnn_inf_model = dnn_inf_model
        self.partition_name_to_proc_id = {}

        # meta-data
        self.__partitions = []
        self.__inter_partition_connections = []

    def partition(self):
        self.__create_partitions()
        self.__add_connections_within_partitions()
        self.__add_external_ios()
        self.__transfer_external_ios()

    def __create_partitions(self):
        self.__partitions = []
        for partition_desc in self.dnn_inf_model.json_partitions:
            layer_names = partition_desc["layers"]
            partition = self.__create_partition(partition_desc["name"], layer_names)
            self.__partitions.append(partition)

    def __create_partition(self, partition_name, layer_names):
        partition = DNN(name=partition_name)
        for layer_name in layer_names:
            layer = self.dnn.find_layer_by_name(layer_name)
            layer_copy = copy.deepcopy(layer)
            partition.add_layer(layer_copy)

        return partition

    def __add_connections_within_partitions(self):
        for connection in self.dnn.get_connections():
            src_partition_id = self.__find_partition_id(connection.src.id)
            dst_partition_id = self.__find_partition_id(connection.dst.id)
            # print("src partition id: ", src_partition_id, "dst partition id: ", dst_partition_id, "with total", len(self.__partitions), "partitions")
            # connection is within partitions
            if src_partition_id == dst_partition_id:
                partition = self.__partitions[src_partition_id]
                partition.connect_layers_by_name(connection.src.name, connection.dst.name)

    def __transfer_external_ios(self):
        """ Transfer external I/Os from original DNN to partitions"""
        self.__transfer_external_inputs()
        self.__transfer_external_outputs()

    def __transfer_external_inputs(self):
        """ Transfer external inputs (data sources) from original DNN to partitions"""
        inputs = self.dnn.get_inputs()
        for external_input in inputs:
            layer_id = external_input.dnn_layer.id
            partition_id = self.__find_partition_id(layer_id)
            partition = self.__partitions[partition_id]
            name = external_input.data_layer.name
            iw = external_input.data_layer.iw
            ih = external_input.data_layer.ih
            ifm = external_input.data_layer.ifm
            partition.add_external_input(name, iw, ih, ifm)

    def __transfer_external_outputs(self):
        """ Transfer external outputs (data consumers) from original DNN to partitions"""
        outputs = self.dnn.get_outputs()
        for external_output in outputs:
            layer_id = external_output.dnn_layer.id
            partition_id = self.__find_partition_id(layer_id)
            partition = self.__partitions[partition_id]
            name = external_output.data_layer.name
            ow = external_output.data_layer.ow
            oh = external_output.data_layer.oh
            ofm = external_output.data_layer.ofm
            partition.add_external_output(name, ow, oh, ofm)

    def __add_external_ios(self):
        """ Add external I/Os that occur due to communication between partitions"""
        for connection in self.dnn.get_connections():
            src_partition_id = self.__find_partition_id(connection.src.id)
            dst_partition_id = self.__find_partition_id(connection.dst.id)

            # connection is among partitions (not within one)
            if src_partition_id != dst_partition_id:
                io_name = "external_" + connection.src.name + "_" + connection.dst.name
                src_partition = self.__partitions[src_partition_id]
                dst_partition = self.__partitions[dst_partition_id]
                # TODO:check
                # add external output to the source layer in the source partition
                src_layer_in_partition = src_partition.find_layer_by_name(connection.src.name)
                src_partition.add_external_output(io_name,
                                                  connection.src.ow,
                                                  connection.src.oh,
                                                  connection.src.ofm,
                                                  src_layer_in_partition)
                # add external input to the destination layer in the destination partition
                dst_layer_in_partition = dst_partition.find_layer_by_name(connection.dst.name)
                dst_partition.add_external_input(io_name,
                                                 connection.dst.iw,
                                                 connection.dst.ih,
                                                 connection.dst.ifm,
                                                 dst_layer_in_partition)

                # add external I/O
                inter_partition_connection = InterDNNConnection(io_name,
                                                                src_partition, dst_partition,
                                                                connection.src.ow, connection.src.oh,
                                                                connection.src.ofm)
                self.__inter_partition_connections.append(inter_partition_connection)

    def __find_partition_id(self, layer_id):
        layer = self.layers[layer_id]
        layer_name = layer.name
        partition_id = 0
        for partition_desc in self.dnn_inf_model.json_partitions:
            if layer_name in partition_desc["layers"]:
                return partition_id
            partition_id += 1

    #########
    # getters

    def get_partitions(self):
        return self.__partitions

    def get_inter_partition_connections(self):
        return self.__inter_partition_connections

    #######
    # print functions

    def print_partitions(self):
        for partition in self.__partitions:
            print("PARTITION: ")
            partition.print_details()
            print("")
