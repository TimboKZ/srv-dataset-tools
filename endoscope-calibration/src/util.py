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


def pick_equidistant_frames(video_cap, frame_count):
    total_frames = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
    step = total_frames // frame_count

    frames = [None] * frame_count

    curr_frame = 0
    for i in range(frame_count):
        video_cap.set(cv.CAP_PROP_POS_FRAMES, curr_frame)
        ret, frame = video_cap.read()
        # TODO: Check ret
        frames[i] = frame
        curr_frame += step

    return frames


def apply_brightness_contrast(in_frame, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        out_frame = cv.addWeighted(in_frame, alpha_b, in_frame, 0, gamma_b)
    else:
        out_frame = in_frame.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        out_frame = cv.addWeighted(out_frame, alpha_c, out_frame, 0, gamma_c)

    return out_frame


def save_intrinsics(file_path, cam_matrix, dist_coeffs):
    dist_coeffs = dist_coeffs.flatten()
    assert cam_matrix.shape == (3, 3)
    assert dist_coeffs.shape == (5,)

    json_data = {
        'cam_matrix': cam_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist()
    }
    with open(file_path, 'w') as file:
        json.dump(json_data, file)
