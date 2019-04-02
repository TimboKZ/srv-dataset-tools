import pandas as pd
from os import path
import numpy as np

# Our local modules
from ds_tools.shared import util


def extract_string_array_column(column_array, output_csv, number_count):
    string_count = len(column_array)

    mat = np.zeros((string_count, number_count), dtype=np.float32)
    for string_i in range(string_count):
        string = column_array[string_i]
        string_parts = string.strip().split()
        for num_i in range(number_count):
            mat[string_i, num_i] = np.round(np.float32(string_parts[num_i]), 6)

    np.savetxt(output_csv, mat, fmt='%.6f', delimiter=',')


def extract_individual_hands(input_kinematics_csv, output_dir):
    """
    Splits `DaVinciSiMemory.csv` from dvLogger kinematics data and stores them in a separate, nicely formatted CSV file.
    Note that the input for this script has to be a CSV file generated using tools from the following repo:
      - https://github.com/surgical-vision/davinci-video-to-kinematics-sync

    That is, the input should NOT be a `.issi` file but rather a `.csv` file generated from the `.issi`.
    """
    kinematics_df = pd.read_csv(input_kinematics_csv, header=0)
    values = kinematics_df.values

    # Clean up the data by throwing away unwanted columns.
    with open(input_kinematics_csv) as csv_file:
        header_line = csv_file.readline()
    headers = header_line.split(',')
    data_column_indices = []
    curr_index = -1
    for header in headers:
        curr_index += 1
        if header.startswith('header.') or header.startswith('data.'):
            data_column_indices.append(curr_index)
    if len(data_column_indices) != 15:
        raise ValueError('The provided doesn\'t have the right number of columns!')

    # Extract individual columns into separate `.csv` files
    extract_string_array_column(values[:, data_column_indices[5]], path.join(output_dir, 'joint_angles_ecm.csv'), 8)
    extract_string_array_column(values[:, data_column_indices[6]], path.join(output_dir, 'joint_angles_psm1.csv'), 8)
    extract_string_array_column(values[:, data_column_indices[7]], path.join(output_dir, 'joint_angles_psm2.csv'), 8)
    extract_string_array_column(values[:, data_column_indices[8]], path.join(output_dir, 'joint_angles_psm3.csv'), 8)
    extract_string_array_column(values[:, data_column_indices[10]], path.join(output_dir, 'pose_ecm.csv'), 12)
    extract_string_array_column(values[:, data_column_indices[11]], path.join(output_dir, 'pose_psm.csv'), 36)


def main():
    data_dir = util.get_data_dir()
    capture_dir = path.join(data_dir, 'placenta_phantom_capture', 'synced')
    # capture_dir = path.join(data_dir, 'prostate_surgery')

    input_kinematics_csv = path.join(capture_dir, 'DaVinciSiMemory.csv')
    output_dir = capture_dir

    extract_individual_hands(input_kinematics_csv, output_dir)


if __name__ == '__main__':
    main()
