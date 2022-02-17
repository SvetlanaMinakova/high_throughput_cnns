from models.edge_platform.Architecture import Architecture
from converters.json_converters.JSONNestedClassVisitor import JSONNestedClassVisitor


def architecture_to_json(architecture: Architecture, filepath: str):
    """
    Convert a target edge platform (architecture) into a JSON File
    :param architecture: edge platform (architecture)
    :param filepath: path to target .json file
    :return: JSON string, encoding the edge platform (architecture)
    """
    visitor = JSONNestedClassVisitor(architecture, filepath)
    visitor.run()

