from models.app_model.dnn_inf_model import DNNInferenceModel
from DSE.scheduling.dnn_scheduling import str_to_scheduling
from models.data_buffers.data_buffers import DataBuffer
import json
from converters.json_converters.json_util import extract_or_default


def json_to_dnn_inf_model(path: str):
    """
    Convert a JSON File into a DNN inference model
    :param path: path to .json file which encodes the DNN inference model
    :return: DNN inference model
    """

    with open(path, 'r') as file:
        if file is None:
            raise FileNotFoundError
        else:
            json_dnn_inf_model = json.load(file)
            # parse fields
            schedule_type_str = extract_or_default(json_dnn_inf_model, "schedule_type", "PIPELINE")
            schedule_type = str_to_scheduling(schedule_type_str)
            partitions = extract_or_default(json_dnn_inf_model, "json_partitions", [])
            connections = extract_or_default(json_dnn_inf_model, "json_connections", [])

            inter_partition_buffers = []
            inter_partition_buffers_desc = extract_or_default(json_dnn_inf_model, "inter_partition_buffers", [])
            for desc in inter_partition_buffers_desc:
                buffer = parse_inter_partition_buffer(desc)
                inter_partition_buffers.append(buffer)

            dnn_inf_model = DNNInferenceModel(schedule_type, partitions, connections, inter_partition_buffers)
            return dnn_inf_model


def parse_inter_partition_buffer(json_buf):
    """
    Parse inter-CNN buffer description
    :param json_buf: json description of inter-CNN buffer
    :return: inter-CNN buffer, defined as a DataBuffer
    """
    name = extract_or_default(json_buf, "name", "B")
    size = extract_or_default(json_buf, "size", 0)
    buffer = DataBuffer(name, size)

    buffer.users = extract_or_default(json_buf, "users", [])
    buffer.type = extract_or_default(json_buf, "type", "none")
    buffer.subtype = extract_or_default(json_buf, "subtype", "none")
    return buffer

