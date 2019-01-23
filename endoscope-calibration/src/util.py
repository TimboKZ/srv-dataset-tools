from os import path
import numpy as np
import cv2 as cv
import json
import sys

# Necessary to make sure this code works when imported into a Jupyter notebook
script_dir = path.dirname(path.realpath(__file__))
sys.path.append(script_dir)


def to_homog(cart):
    """
    :param cart: Cartesian coordinates as a `m x n` matrix, where `m` is the number
                 of dimensions and `n` is the number of points
    :return: Homogeneous coordinates as a `(m+1) x n` matrix
    """
    return np.concatenate((cart, np.ones((1, cart.shape[1]))), axis=0)


def to_cart(homog):
    """
    The opposite of `to_homog` function
    """
    m = homog.shape[0]
    return homog[0:m - 1, :] / np.tile([homog[m - 1, :]], (m - 1, 1))


def pick_frames(video_cap, frame_indices, convert_to_rgb=False):
    frame_count = len(frame_indices)
    frames = [None] * frame_count

    for i in range(frame_count):
        video_cap.set(cv.CAP_PROP_POS_FRAMES, frame_indices[i])
        ret, frame = video_cap.read()
        # TODO: Check ret
        if convert_to_rgb:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frames[i] = frame

    return frames


def pick_equidistant_frames(video_cap, frame_count, convert_to_rgb=False):
    total_frames = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
    step = total_frames // frame_count
    return pick_frames(video_cap, [i * step for i in range(frame_count)], convert_to_rgb=convert_to_rgb)


def apply_brightness_contrast(in_frame, brightness=0, contrast=1.0):
    out_frame = (in_frame + brightness) * contrast
    out_frame = np.clip(out_frame, 0, 255).astype(np.uint8)
    return out_frame


def save_intrinsics(file_path, cam_matrix, dist_coeffs):
    dist_coeffs = dist_coeffs.flatten()
    assert cam_matrix.shape == (3, 3)
    assert dist_coeffs.shape == (5,)

    data = {
        'cam_matrix': cam_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist()
    }
    with open(file_path, 'w') as file:
        json.dump(data, file)


def load_intrinsics(file_path):
    with open(file_path) as file:
        data = json.load(file)

    cam_matrix = np.array(data['cam_matrix'])
    dist_coeffs = np.array(data['dist_coeffs'])

    return cam_matrix, dist_coeffs


def undistort_line(line, cam_matrix, dist_coeffs, new_cam_matrix):
    cv_points = np.expand_dims(line.T, axis=0).astype(np.float32)
    new_line = cv.undistortPoints(cv_points,
                                  cam_matrix,
                                  dist_coeffs,
                                  P=new_cam_matrix
                                  )[0].T

    new_line[0, :] = new_line[0, :]
    new_line[1, :] = new_line[1, :]

    return new_line
