from models.edge_platform.Architecture import Architecture
import json
from converters.json_converters.json_util import extract_or_default


def json_to_architecture(path: str):
    """
    Convert a JSON File into a target edge platform (architecture)
    :param path: path to .json file which encodes the target edge platform (architecture)
    :return: target edge platform (architecture)
    """

    with open(path, 'r') as file:
        if file is None:
            raise FileNotFoundError
        else:
            json_architecture = json.load(file)

            # parse simple fields
            name = extract_or_default(json_architecture, "name", "architecture")
            processors = extract_or_default(json_architecture, "processors", [])
            processors_types = extract_or_default(json_architecture, "processors_types", [])
            processors_types_distinct = extract_or_default(json_architecture, "processors_types_distinct", [])
            accelerators = extract_or_default(json_architecture, "accelerators", [])
            max_giga_flops_per_proc_type = extract_or_default(json_architecture,
                                                              "max_giga_flops_per_proc_type",
                                                              [1 for _ in processors_types_distinct])

            # create architecture
            architecture = Architecture(processors,
                                        processors_types,
                                        processors_types_distinct)
            architecture.name = name
            architecture.accelerators = accelerators
            architecture.max_giga_flops_per_proc_type = max_giga_flops_per_proc_type

            # parse communication matrix
            communication_matrix = extract_or_default(json_architecture,
                                                      "communication_speed_matrix_mb_s",
                                                      [[0 for i in range(len(processors))] for j in range(len(processors))])
            architecture.communication_speed_matrix_mb_s = communication_matrix
            return architecture

