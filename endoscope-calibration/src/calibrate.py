from os import path
import numpy as np
import cv2 as cv
import sys

# Necessary to make sure this code works when imported into a Jupyter notebook
script_dir = path.dirname(path.realpath(__file__))
sys.path.append(script_dir)

# Our local modules
import util

simple_term_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def simple_preprocess(in_frame):
    gray = cv.cvtColor(in_frame, cv.COLOR_BGR2GRAY)
    return util.apply_brightness_contrast(gray, 80, 40)


def get_detected_chessboard_points(video_cap, cb_size, frames_with_pattern=30, term_criteria=simple_term_criteria,
                                   preprocess_func=simple_preprocess):
    video_cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    # Generate the plane with the chessboard (inner) corner points
    world_plane = np.zeros((cb_size[0] * cb_size[1], 3), np.float32)
    world_plane[:, :2] = np.mgrid[0:cb_size[0], 0:cb_size[1]].T.reshape(-1, 2)

    world_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.

    frame_count = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
    point_count = 0
    curr_frame = 0
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        curr_frame += 1
        pp_frame = preprocess_func(frame)
        ret, img_corners = cv.findChessboardCorners(pp_frame, cb_size, None)

        # If found, add object points, image points (after refining them)
        if ret:
            img_cb_corners = cv.cornerSubPix(pp_frame, img_corners, (11, 11), (-1, -1), term_criteria)
            world_points.append(world_plane)
            img_points.append(img_cb_corners)
            point_count += 1

        print('\rProcessing frame {} out of {}   (found pattern on {} frames, need {} more)'
              .format(curr_frame, frame_count, point_count, frames_with_pattern - point_count), end='')

        if point_count >= frames_with_pattern:
            print('')
            print('Found {} points, terminating the loop.'.format(frames_with_pattern))
            break

    return world_points, img_points


def calculate_camera_intrinsics(world_points, img_points, sample_frame):
    (ret,
     cam_matrix,
     dist_coeffs,
     rvecs,
     tvecs) = cv.calibrateCamera(world_points, img_points, sample_frame.shape[:2], None, None)
    return cam_matrix, dist_coeffs, rvecs, tvecs
