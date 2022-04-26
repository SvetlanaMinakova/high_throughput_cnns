from models.dnn_model.dnn import DNN
from models.data_buffers import DataBuffer


def get_buffer_definition(buffer: DataBuffer) -> str:
    buf_definition = ""
    return buf_definition


def buf_type_to_buf_class(buf_type: str):
    if buf_type == "double_buffer":
        return "DoubleBuffer"
    else:
        return "SingleBuffer"



