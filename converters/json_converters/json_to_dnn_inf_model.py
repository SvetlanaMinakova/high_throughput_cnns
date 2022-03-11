from models.app_model.dnn_inf_model import DNNInferenceModel
from DSE.scheduling.dnn_scheduling import str_to_scheduling
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
            partitions = extract_or_default(json_dnn_inf_model, "partitions", [])
            connections = extract_or_default(json_dnn_inf_model, "connections", [])
            inter_partition_buffers = extract_or_default(json_dnn_inf_model, "inter_partition_buffers", [])

            dnn_inf_model = DNNInferenceModel(schedule_type, partitions, connections, inter_partition_buffers)
            return dnn_inf_model

