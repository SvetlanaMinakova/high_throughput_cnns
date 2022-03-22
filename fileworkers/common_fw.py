import os


def create_dir_if_does_not_exist(dir_path: str):
    """
    if the directory is not present, create it.
    :param dir_path: path to directory
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def clear_folder(directory_to_clear):
    """
    Recursively delete directory with files
    :param directory_to_clear: path to directory with files
    """
    if os.path.exists(directory_to_clear):
        for the_file in os.listdir(directory_to_clear):
            file_path = os.path.join(directory_to_clear, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                else:
                    clear_folder(file_path)
                    os.rmdir(file_path)
            except Exception as e:
                print(e)


def create_or_overwrite_dir(dir_path):
    """
    Delete and re-create directory
    :param dir_path: directory path
    """
    if os.path.exists(dir_path):
        clear_folder(dir_path)
    else:
        os.makedirs(dir_path)