import pandas as pd
from os import path
import numpy as np

script_dir = path.dirname(path.realpath(__file__))
data_dir = path.join(script_dir, '..', 'data')


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
    extract_string_array_column(kinematics_df.values[:, 6], path.join(output_dir, 'joint_angles_ecm.csv'), 8)
    extract_string_array_column(kinematics_df.values[:, 7], path.join(output_dir, 'joint_angles_psm1.csv'), 8)
    extract_string_array_column(kinematics_df.values[:, 8], path.join(output_dir, 'joint_angles_psm2.csv'), 8)
    extract_string_array_column(kinematics_df.values[:, 9], path.join(output_dir, 'joint_angles_psm3.csv'), 8)
    extract_string_array_column(kinematics_df.values[:, 11], path.join(output_dir, 'pose_ecm.csv'), 12)
    extract_string_array_column(kinematics_df.values[:, 12], path.join(output_dir, 'pose_psm.csv'), 36)


def main():
    input_kinematics_csv = path.join(data_dir, 'DaVinciSiMemory.csv')
    outpit_dir = data_dir
    extract_individual_hands(input_kinematics_csv, outpit_dir)


if __name__ == '__main__':
    main()
