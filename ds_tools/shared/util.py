from os import path
import numpy as np
import tempfile
import json
import os

script_dir = path.dirname(path.realpath(__file__))
data_dir = path.normpath(path.join(script_dir, '..', '..', 'data'))
resource_dir = path.normpath(path.join(script_dir, '..', '..', 'resources'))


def get_data_dir():
    """
    Returns the path to the top-level data directory
    """
    return data_dir


def get_resource_dir():
    """
    Returns the path to the top level resource directory
    """
    return resource_dir


def get_temp_filename(name='srv-temp-file', extension=None):
    """
    Returns an absolute path to temp file with the specified name and extension.
    Filename format:
      - <name>_<rand-int>.<extension>

    :param name:
    :param extension:
    :return:
    """
    filename = '{}_{}'.format(name, np.random.randint(1000))
    if extension:
        filename = '{}.{}'.format(filename, extension)

    full_path = path.join(tempfile.gettempdir(), filename)
    return full_path


def ensure_dir(dir_path):
    if not path.exists(dir_path):
        os.makedirs(dir_path)


def save_dict(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)


def load_dict(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data


def save_intrinsics(file_path, cam_matrix=None, dist_coeffs=None, width=None, height=None):
    if cam_matrix is not None:
        assert cam_matrix.shape == (3, 3)
    if dist_coeffs is not None:
        dist_coeffs = dist_coeffs.flatten()
        assert dist_coeffs.shape == (5,)

    def prep_array(array):
        if array is None:
            return None
        return array.tolist()

    data = {
        'cam_matrix': prep_array(cam_matrix),
        'dist_coeffs': prep_array(dist_coeffs),
        'width': width,
        'height': height,
    }
    save_dict(file_path, data)


def load_intrinsics(file_path):
    data = load_dict(file_path)

    def prep_array(array):
        if array is None:
            return None
        return np.array(array)

    cam_matrix = prep_array(data.get('cam_matrix'))
    dist_coeffs = prep_array(data.get('dist_coeffs'))
    width = data.get('width')
    height = data.get('height')
    return cam_matrix, dist_coeffs, width, height


def save_extrinsics(file_path, transform=None, endoscope_markers=None):
    if endoscope_markers is not None:
        assert endoscope_markers.shape[0] == 3
    if transform is not None:
        assert transform.shape == (4, 4)

    def prep_array(array):
        if array is None:
            return None
        return array.tolist()

    data = {
        'transform': prep_array(transform),
        'endoscope_markers': prep_array(endoscope_markers),
    }
    save_dict(file_path, data)


def load_extrinsics(file_path):
    data = load_dict(file_path)

    def prep_array(array):
        if array is None:
            return None
        return np.array(array)

    transform = prep_array(data.get('transform'))
    endoscope_markers = prep_array(data.get('endoscope_markers'))
    return transform, endoscope_markers
