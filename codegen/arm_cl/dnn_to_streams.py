from models.dnn_model.dnn import DNN, Layer, Connection

"""
To handle branching and residual connections,
ARM-CL requires DNN to be represented as a set of linear parts (branches)
Every branch is then defined as a Sub-stream. The data flow between the 
branches (sub-streams) is then controlled by special operators such as concat
"""


class DNNSubStreamsGenerator:
    """
    class that generates DNN sub-streams for ARM-CL library
    """
    def __init__(self, dnn: DNN):
        self.dnn = dnn

        # meta-data
        self.__sub_streams = []
        self.__multi_input_sub_streams = []
        self.__cur_sub_stream = []
        self.__sub_stream_inputs = []
        self.__sub_stream_outputs = []
        self.__sub_stream_groups = []
        self.__residual_empty_partitions = {}
        self.residual_connections = []
        
    def __clean_meta_data(self):
        self.__sub_streams = []
        self.__multi_input_sub_streams = []
        self.__cur_sub_stream = []
        self.__sub_stream_inputs = []
        self.__sub_stream_outputs = []
        self.__sub_stream_groups = []
        # empty partition, simulating residual connection
        # dictionary of format key=sub_stream_id, value = (inp_layer_name, outp_layer_name))
        self.__residual_empty_partitions = {}
        self.residual_connections = []

    def dnn_to_sub_streams(self):
        """
        Represent DNN as a set of partitions (sub-streams)
        :return tuple: (sub-streams, sub-stream-inputs, sub-stream-outputs, sub_stream_groups), where
         - sub-streams: is a list of sub-streams. Every sub-stream is a list of layer names
         - sub-stream inputs: matrix of sub-stream inputs n x m, where n = number of DNN sub-streams,
           m = number of input_examples sub-streams for current sub-stream. Every input_examples = id of a sub-stream
         - sub-stream inputs: matrix of sub-stream inputs n x k, where n = number of DNN sub-streams,
           k = number of output sub-streams for current sub-stream. Every input_examples = id of a sub-stream
        - sub-stream groups: list of sub-streams, where every sub-stream is a list of layer names,
            and all sub-streams in a group:
                1) receive input_examples from the same sub-stream
                2) produce output to the same sub-stream
        """

        self.__clean_meta_data()
        self.__register_residual_connections()
        self.__generate_sub_streams()
        self.__generate_sub_stream_ios()
        self.__group_sub_streams_by_io()
        return self.__sub_streams, self.__sub_stream_inputs, self.__sub_stream_outputs, self.__sub_stream_groups

    def __register_residual_connections(self):
        for connection in self.dnn.get_connections():
            if self.__is_residual_connection(connection):
                # print("residual connection registered: ", connection)
                self.residual_connections.append(connection)

    def __is_residual_connection(self, connection):
        src = connection.src
        dst = connection.dst

        src_is_multi_output = True if len(self.dnn.get_layer_output_connections(src)) > 1 else False
        if not src_is_multi_output:
            return False

        dst_is_multi_input = True if len(self.dnn.get_layer_input_connections(dst)) > 1 else False
        if not dst_is_multi_input:
            return False
        return True

    def __generate_sub_streams(self):
        for layer in self.dnn.get_layers():
            self.__visit_layer(layer)

        # add last sub-stream
        self.__append_cur_sub_stream_if_not_empty()

    def __visit_layer(self, layer):
        """
        Visit dnn layer
        :param layer: layer
        """
        input_connections = self.dnn.get_layer_input_connections(layer)
        output_connections = self.dnn.get_layer_output_connections(layer)
        # print(layer.name, "has", len(input_connections), "inputs and", len(output_connections), "outputs")

        for connection in input_connections:
            if self.__is_registered_as_residual(connection):
                self._process_residual_connection(connection)

        # if layer has multiple inputs, put it into a separate (isolated) partition
        if len(input_connections) > 1:
            self.__visit_as_multi_input_layer(layer)
            return

        # if layer has multiple outputs, put all follow-up layers in a separate partition
        if len(output_connections) > 1:
            self.__visit_as_multi_output_layer(layer, output_connections)
            return

        # if layer has external inputs, put it into a new partition
        if self.__contains_external_inputs(input_connections):
            self.__visit_as_layer_with_external_inputs(layer)
            return

        # if layer was not visited as a special case above, add layer to the current partition
        self.__cur_sub_stream.append(layer.name)

    def __visit_as_multi_input_layer(self, layer):
        """
        Process layer that has multiple inputs
        """
        self.__save_in_isolated_partition(layer)

    def __visit_as_layer_with_external_inputs(self, layer):
        """
        if layer has multiple inputs, put it and all follow-up layers into a new partition
        """
        # first, add the current partition (without the layer) to partitions list
        self.__append_cur_sub_stream_if_not_empty()
        # then create a new partition with layer in it
        self.__cur_sub_stream = [layer.name]

    def __visit_as_multi_output_layer(self, layer, output_connections):
        self.__save_in_isolated_partition(layer)
        """
        has_residual_output_connections = False
        for connection in output_connections:
            if self.__is_registered_as_residual(connection):
                has_residual_output_connections = True
        if has_residual_output_connections:
            self.__save_in_isolated_partition(layer)
            
        else:
            # add current partition to the list of partitions
            self.__append_cur_sub_stream_if_not_empty()
            # put layers in a new partition
            self.__cur_sub_stream = []
            # add layer to the current partition
            self.__cur_sub_stream.append(layer.name)
            # add partition to the list of partitions
            self.__append_cur_sub_stream_if_not_empty()
            # put all follow-up layers in a new partition
            self.__cur_sub_stream = []
        """

    def __save_in_isolated_partition(self, layer):
        """
        Put layer into a separate (isolated) partition
        """
        # first, add the current partition (without the layer) to partitions list
        self.__append_cur_sub_stream_if_not_empty()
        # then create a new partition with layer in it
        self.__cur_sub_stream = [layer.name]
        self.__append_cur_sub_stream()
        cur_sub_stream_id = len(self.__sub_streams) - 1
        # add layer name to the list of multi-inputs
        self.__multi_input_sub_streams.append(cur_sub_stream_id)
        # create new partition for follow-up layers
        self.__cur_sub_stream = []

    def _process_residual_connection(self, con):
        # print("process residual connection", con)
        # process residual connection
        residual_con_partition = []
        self.__sub_streams.append(residual_con_partition)
        residual_con_partition_id = len(self.__sub_streams) - 1
        self.__residual_empty_partitions[residual_con_partition_id] = (con.src.name, con.dst.name)

    def __is_registered_as_residual(self, connection):
        if connection in self.residual_connections:
            return True
        return False

    def __generate_sub_stream_ios(self):
        # Generate inputs and outputs for every sub-stream
        # NOTE: step should be performed after sub-streams (lists of layers) are generated
        self.__init_sub_stream_ios()
        for connection in self.dnn.get_connections():
            src_name = connection.src.name
            dst_name = connection.dst.name
            src_sub_stream_id = self.find_sub_stream_id(src_name)
            dst_sub_stream_id = self.find_sub_stream_id(dst_name)
            if src_sub_stream_id is None:
                raise Exception("Sub-streams I/Os generation error: src sub-stream id for connection " + str(connection) + " is None!")
            if dst_sub_stream_id is None:
                raise Exception("Sub-streams I/Os generation error: dst sub-stream id for connection " + str(connection) + " is None!")
            # this connection is a connection between two sub-streams
            if src_sub_stream_id != dst_sub_stream_id:
                if self.__is_registered_as_residual(connection):
                    # find residual connection, simulated as an empty partition
                    residual_sub_stream_id = self.__find_residual_sub_stream_id(src_name, dst_name)
                    # first, connect src layer to residual connection
                    self.__generate_sub_stream_io(src_sub_stream_id, residual_sub_stream_id)
                    # then, connect residual connection to the destination layer
                    self.__generate_sub_stream_io(residual_sub_stream_id, dst_sub_stream_id)
                else:
                    # connect src to dst directly
                    self.__generate_sub_stream_io(src_sub_stream_id, dst_sub_stream_id)

    def __generate_sub_stream_io(self, src_sub_stream_id, dst_sub_stream_id):
        """Generate I/O connection between two sub-streams"""
        self.__sub_stream_outputs[src_sub_stream_id].append(dst_sub_stream_id)
        self.__sub_stream_inputs[dst_sub_stream_id].append(src_sub_stream_id)

    def find_sub_stream_id(self, layer_name):
        sub_stream_id = 0
        for sub_stream in self.__sub_streams:
            if layer_name in sub_stream:
                return sub_stream_id
            sub_stream_id += 1

    def __find_sub_stream_id_starting_with(self, layer_name):
        sub_stream_id = 0
        for sub_stream in self.__sub_streams:
            if len(sub_stream) > 0:
                if sub_stream[0] == layer_name:
                    return sub_stream_id
            sub_stream_id += 1

    def __find_sub_stream_id_ending_with(self, layer_name):
        sub_stream_id = 0
        for sub_stream in self.__sub_streams:
            if len(sub_stream) > 0:
                if sub_stream[-1] == layer_name:
                    return sub_stream_id
            sub_stream_id += 1

    def __find_residual_sub_stream_id(self, src_name, dst_name):
        for item in self.__residual_empty_partitions.items():
            k, v = item
            if src_name in v and dst_name in v:
                return k

    def __init_sub_stream_ios(self):
        for i in range(len(self.__sub_streams)):
            self.__sub_stream_inputs.append([])
            self.__sub_stream_outputs.append([])

    def __group_sub_streams_by_io(self):
        for sub_stream_id in range(len(self.__sub_streams)):
            sub_stream = self.__sub_streams[sub_stream_id]
            inputs = self.__sub_stream_inputs[sub_stream_id]
            outputs = self.__sub_stream_outputs[sub_stream_id]
            group = self.__get_group_with_ios(inputs, outputs)

            if group is None:
                group = SubStreamGroup(inputs, outputs)
                self.__sub_stream_groups.append(group)

            group.add_sub_stream(sub_stream)

    def __get_group_with_ios(self, inputs, outputs):
        for group in self.__sub_stream_groups:
            if group.inputs == inputs and group.outputs == outputs:
                return group
        return None

    def __append_cur_sub_stream_if_not_empty(self):
        if self.__cur_sub_stream:
            self.__sub_streams.append(self.__cur_sub_stream)

    def __append_cur_sub_stream(self):
        self.__sub_streams.append(self.__cur_sub_stream)

    def __contains_external_inputs(self, input_connections):
        """
        Checks if set of connections contains external (belonging to other, previously visited sub-streams) inputs
        :param input_connections: input_examples connections of the current layer
        :return: True, if layer has external inputs and False otherwise
        """
        for connection in input_connections:
            if connection.src.name not in self.__cur_sub_stream:
                for visited_partition in self.__sub_streams:
                    if connection.src.name in visited_partition:
                        return True
        return False

    ##########
    # Getters

    def get_sub_streams(self):
        return self.__sub_streams

    def get_sub_stream_groups(self):
        return self.__sub_stream_groups

    def get_multi_input_stream_ids(self):
        return self.__multi_input_sub_streams

    def is_multi_branch_dnn(self):
        return len(self.__sub_streams) > 1

    ###################
    # PRINT FUNCTIONS #

    def print_multi_input_sub_streams(self):
        print("Multi-input_examples sub-streams: ")
        for sub_stream_id in self.__multi_input_sub_streams:
            sub_stream = self.__sub_streams[sub_stream_id]
            inputs = self.__sub_stream_inputs[sub_stream_id]
            outputs = self.__sub_stream_outputs[sub_stream_id]
            print("Sub-stream ", sub_stream_id, ":", sub_stream, ", inputs: ", inputs, ",outputs: ", outputs)

    def print_residual_connections(self):
        print("Residual partitions")
        for item in self.__residual_empty_partitions.items():
            k, v = item
            print("sub-stream:", k, "connection", v)

    def print_sub_streams(self):
        sub_stream_id = 0
        for sub_stream in self.__sub_streams:
            print("sub-stream", sub_stream_id, sub_stream)
            sub_stream_id += 1

    def print_grouped_sub_streams(self):
        group_id = 0
        for group in self.__sub_stream_groups:
            print("Sub-stream group:", group_id)

            print("  inputs:", group.inputs)
            print("  outputs:", group.outputs)

            print("  sub-streams:")
            for sub_stream in group.sub_streams:
                print("   ", sub_stream)
            group_id += 1


class SubStreamGroup:
    def __init__(self, input_stream_ids, output_stream_ids):
        self.inputs = input_stream_ids
        self.outputs = output_stream_ids
        self.sub_streams = []

    def add_sub_stream(self, sub_stream):
        self.sub_streams.append(sub_stream)

