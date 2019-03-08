from os import path

script_dir = path.dirname(path.realpath(__file__))
data_dir = path.normpath(path.join(script_dir, '..', '..', 'data'))


def get_data_dir():
    """
    Returns the path to the top-level data directory
    """
    return data_dir
