import json
from converters.json_converters.json_util import extract_or_default


def parse_ga_conf(path):
    """ Parse high-additional GA config """
    with open(path, 'r') as file:
        if file is None:
            raise FileNotFoundError
        else:
            conf_as_dict = {}
            conf = json.load(file)
            conf_as_dict["epochs"] = extract_or_default(conf, "epochs", 10)
            conf_as_dict["population_start_size"] = extract_or_default(conf, "population_start_size", 100)
            conf_as_dict["selection_percent"] = extract_or_default(conf, "selection_percent", 50)
            conf_as_dict["mutation_probability"] = extract_or_default(conf, "mutation_probability", 0.0)
            conf_as_dict["mutation_percent"] = extract_or_default(conf, "mutation_percent", 10)
            conf_as_dict["max_no_improvement_epochs"] = extract_or_default(conf, "max_no_improvement_epochs", 10)
            conf_as_dict["eval_communication_costs"] = extract_or_default(conf, "eval_communication_costs", False)
            conf_as_dict["verbose"] = extract_or_default(conf, "verbose", True)
            conf_as_dict["preferred_proc_id"] = extract_or_default(conf, "preferred_proc_id", -1)
            conf_as_dict["preset_preferred_proc_probability"] = \
                extract_or_default(conf, "preset_preferred_proc_probability", -1)
            return conf_as_dict

