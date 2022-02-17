from models.edge_platform.Architecture import get_jetson
from converters.json_converters.architecture_to_json import architecture_to_json
from util import get_project_root
import os


def save_as_json():
    """ Save a platform as a JSON file"""
    jetson = get_jetson()
    file_path = str(os.path.join(str(get_project_root()), "output/architecture/jetson.json"))
    architecture_to_json(jetson, file_path)


def load_from_json():
    file_path = str(os.path.join(str(get_project_root()), "output/architecture/jetson.json"))


save_as_json()
