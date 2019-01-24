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
    return gray


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

        print('\rProcessed frame {} out of {}   (found pattern on {} frames, need {} more)'
              .format(curr_frame, frame_count, point_count, frames_with_pattern - point_count), end='')

        if point_count >= frames_with_pattern:
            print('')
            print('Found {} points, terminating the loop.'.format(frames_with_pattern))
            break

    return world_points, img_points


def calculate_camera_intrinsics(points_3d, points_2d, sample_frame):
    (ret,
     cam_matrix,
     dist_coeffs,
     rvecs,
     tvecs) = cv.calibrateCamera(points_3d, points_2d, sample_frame.shape[:2], None, None)
    return cam_matrix, dist_coeffs, rvecs, tvecs


def calc_best_homography(points_1_cart, points_2_cart):
    """
    Uses Direct Linear Transform (DLT) algorithm to calculate the best homography mapping
    `cart_points_1` to corresponding `cart_points_2`

    d - number of dimensions
    n - number of points

    :param points_1_cart: `d x n` matrix
    :param points_2_cart: `d x n` matrix
    :return:
    """
    n = points_1_cart.shape[1]
    points_1_hom = util.to_homog(points_1_cart)
    points_2_hom = util.to_homog(points_2_cart)

    A = np.zeros((2 * n, 9))
    for i in range(n):
        pt1 = points_1_hom[:, i]
        pt2 = points_2_hom[:, i]
        A[2 * i, 3:6] = -pt1
        A[2 * i, 6:9] = pt1 * pt2[1]
        A[2 * i + 1, 0:3] = pt1
        A[2 * i + 1, 6:9] = -pt1 * pt2[0]

    U, s, VT = np.linalg.svd(A)
    h = VT.T[:, -1]
    H = h.reshape(3, 3)
    return H


# Goal of function is to estimate pose of plane relative to camera (extrinsic matrix)
# given points in image xImCart, points in world XCart and intrinsic matrix K
def estimate_plane_pos(points_cart_3d, points_cart_2d, cam_matrix):
    img_points_hom = util.to_homog(points_cart_2d)
    cam_points_hom = np.linalg.inv(cam_matrix) @ img_points_hom
    H = calc_best_homography(points_cart_3d[0:2, :], util.to_cart(cam_points_hom))

    # Estimate first two columns of rotation matrix R from the first two
    # columns of H using the SVD
    R = np.zeros((3, 3))
    U, L, VT = np.linalg.svd(H[:, 0:2])
    R[:, 0:2] = U @ np.eye(3)[:, 0:2] @ VT

    # Estimate the third column of the rotation matrix by taking the cross
    # product of the first two columns
    R[:, 2] = np.cross(R[:, 0], R[:, 1])
    det = np.linalg.det(R)
    if det < 0:
        R[:, 2] *= -1

    # Estimate the translation t by finding the appropriate scaling factor k
    # and applying it to the third column of H
    sum_ = 0
    for i in range(3):
        for j in range(2):
            sum_ += H[i, j] / R[i, j]
    t = H[:, 2] / (sum_ / 6)

    # TO DO: Check whether t_z is negative - if it is then multiply t by -1 and
    # the first two columns of R by -1.
    if t[2] < 0:
        t *= -1
        R[:, 0] *= -1
        R[:, 1] *= -1

    return R, t
