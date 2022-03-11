from models.app_model.dnn_inf_model import DNNInferenceModel
from converters.json_converters.JSONNestedClassVisitor import JSONNestedClassVisitor


def dnn_inf_model_to_json(dnn_inf_model: DNNInferenceModel, filepath: str):
    """
    Convert DNN inference model into a JSON File
    :param dnn_inf_model: DNN inference model
    :param filepath: path to target .json file
    :return: JSON string, encoding the analytical DNN model
    """
    visitor = JSONNestedClassVisitor(dnn_inf_model, filepath)
    visitor.run()