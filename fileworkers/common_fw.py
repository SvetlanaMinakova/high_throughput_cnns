import os


def create_dir_if_does_not_exist(dir_path: str):
    """
    if the directory is not present, create it.
    :param dir_path: path to directory
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)