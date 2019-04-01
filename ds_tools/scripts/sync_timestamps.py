import cv2
import numpy as np
import pandas as pd
from os import path

# Our local modules
from ds_tools.shared import util, cv_util


def main():
    data_dir = util.get_data_dir()
    capture_dir = path.join(data_dir, 'placenta_phantom_capture')
    video_0_path = path.join(capture_dir, 'EndoscopeImageMemory_0.avi')
    video_1_path = path.join(capture_dir, 'EndoscopeImageMemory_1.avi')
    video_csv_0_path = path.join(capture_dir, 'EndoscopeImageMemory_0.csv')
    video_csv_1_path = path.join(capture_dir, 'EndoscopeImageMemory_1.csv')
    kinematics_csv_path = path.join(capture_dir, 'DaVinciSiMemory.csv')
    output_path = path.join(capture_dir, 'synced')

    # Extract `header.timestamp` field for each video CSV
    timestamps_0 = pd.read_csv(video_csv_0_path)['header.timestamp']
    timestamps_1 = pd.read_csv(video_csv_1_path)['header.timestamp']

    # Load the whole kinematics CSV and store `header.timestamp` separately
    kinematics = pd.read_csv(kinematics_csv_path)
    kinematics_timestamps = kinematics['header.timestamp']

    #
    framerate = 30
    mus_per_frame = 1000 / framerate
    start_time = max(timestamps_0[0], timestamps_1[0])
    end_time = min(timestamps_0[len(timestamps_0) - 1], timestamps_1[len(timestamps_1) - 1])

    # Prepare containers for frame indices
    total_frames = int(np.ceil((end_time - start_time) / mus_per_frame))
    frames_0 = np.zeros((total_frames,))
    frames_1 = np.zeros((total_frames,))
    kinematics_indices = np.zeros((total_frames,))
    print('Final videos will have {} frames.'.format(total_frames))

    #
    frame_index = 0
    index_0 = int(0)
    index_1 = int(0)
    kinematics_index = int(0)
    current_time = start_time
    print('Synchronizing frame indices...')
    while current_time < end_time and frame_index < total_frames:

        # Find closest frame for first video
        while True:
            last_frame_diff = np.abs(current_time - timestamps_0[index_0])
            next_index = index_0 + 1
            if next_index >= len(timestamps_0):
                break
            frame_diff = np.abs(current_time - timestamps_0[next_index])
            if frame_diff < last_frame_diff:
                index_0 = next_index
            else:
                break

        # Find closest frame for second video
        while True:
            last_frame_diff = np.abs(current_time - timestamps_1[index_1])
            next_index = index_1 + 1
            if next_index >= len(timestamps_1):
                break
            frame_diff = np.abs(current_time - timestamps_1[next_index])
            if frame_diff < last_frame_diff:
                index_1 = next_index
            else:
                break

        # Find closest kinematics frame
        while True:
            last_frame_diff = np.abs(current_time - kinematics_timestamps[kinematics_index])
            next_index = kinematics_index + 1
            if next_index >= len(timestamps_1):
                break
            frame_diff = np.abs(current_time - kinematics_timestamps[next_index])
            if frame_diff < last_frame_diff:
                kinematics_index = next_index
            else:
                break

        # Store corresponding frames
        frames_0[frame_index] = index_0
        frames_1[frame_index] = index_1
        kinematics_indices[frame_index] = kinematics_index
        current_time += mus_per_frame
        frame_index += 1

    assert frame_index == total_frames
    print('Done!')

    # Make sure output dir exists
    util.ensure_dir(output_path)

    # Write out the synced kinematics CSV
    kinematics_output_path = path.join(output_path, path.basename(kinematics_csv_path))
    kinematics_output_dataframe = pd.DataFrame(columns=kinematics.columns.values)
    print('Kinematics CSV will be written to `{}`...'.format(kinematics_output_path))
    print('Preparing kinematics CSV...')
    for i in range(total_frames):
        print('\rProcessing row {} out of {}...'.format(i + 1, total_frames))
        kin_index = kinematics_indices[i]
        kinematics_output_dataframe.loc[i] = kinematics.loc[kin_index]
    kinematics_output_dataframe.to_csv(kinematics_output_path, header=True)
    print('Done!')

    video_0_source = cv2.VideoCapture(video_0_path)
    video_1_source = cv2.VideoCapture(video_1_path)

    video_0_output_path = path.join(output_path, path.basename(video_0_path))
    video_1_output_path = path.join(output_path, path.basename(video_1_path))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_0_output = cv2.VideoWriter(video_0_output_path, fourcc, framerate, cv_util.get_capture_size(video_0_source))
    video_1_output = cv2.VideoWriter(video_1_output_path, fourcc, framerate, cv_util.get_capture_size(video_1_source))
    print('Videos will be written to `{}` and `{}`...'.format(video_0_output_path, video_1_output_path))

    print('Preparing videos...')
    _, last_frame_0 = video_0_source.read()
    _, last_frame_1 = video_1_source.read()
    last_frame_index_0 = 0
    last_frame_index_1 = 0
    for i in range(total_frames):
        print('\rProcessing frame {} out of {}...'.format(i + 1, total_frames))

        target_frame_index_0 = frames_0[i]
        target_frame_index_1 = frames_1[i]

        # Skip until the desired frame in the first video
        if last_frame_index_0 != target_frame_index_0:
            backup_frame = None
            while last_frame_index_0 != target_frame_index_0 - 1:
                last_frame_index_0 += 1
                success, frame = video_0_source.read()
                if success:
                    backup_frame = frame

            last_frame_index_0 += 1
            success, frame = video_0_source.read()
            if success:
                last_frame_0 = frame
            elif backup_frame is not None:
                last_frame_0 = backup_frame

        # Skip until the desired frame in the second video
        if last_frame_index_1 != target_frame_index_1:
            backup_frame = None
            while last_frame_index_1 != target_frame_index_1 - 1:
                last_frame_index_1 += 1
                success, frame = video_1_source.read()
                if success:
                    backup_frame = frame

            last_frame_index_1 += 1
            success, frame = video_1_source.read()
            if success:
                last_frame_1 = frame
            elif backup_frame is not None:
                last_frame_1 = backup_frame

        # Write out the current frame
        video_0_output.write(last_frame_0)
        video_1_output.write(last_frame_1)
    print('')

    video_0_source.release()
    video_1_source.release()
    video_0_output.release()
    video_1_output.release()

    print('Done!')


if __name__ == "__main__":
    main()
