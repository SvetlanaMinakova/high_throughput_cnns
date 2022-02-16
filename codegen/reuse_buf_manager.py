from models.dnn_model.dnn import DNN
""" Annotate reuse buffers"""


def annotate_reuse_buf_with_users(dnn_partitions: [DNN], dnn_buffers):
    """
    Annotate reuse buffer with partitions that use this buffer
    :param dnn_partitions: DNN partitions
    :param dnn_buffers: DNN buffers
    """
    def _is_external_input_of_partition(connection, partition):
        partition_external_inputs = partition.get_inputs()
        for external_input in partition_external_inputs:
            if connection.dst == external_input.dnn_layer:
                return True
        return False

    def _is_external_output_of_partition(connection, partition):
        partition_external_outputs = partition.get_outputs()
        for external_output in partition_external_outputs:
            if connection.src == external_output.dnn_layer:
                return True
        return False




