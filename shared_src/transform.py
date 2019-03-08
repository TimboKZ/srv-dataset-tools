import numpy as np


def to_transform(rot_matrix=None, trans_vec=None):
    if rot_matrix is not None:
        assert rot_matrix.shape == (3, 3)
    if trans_vec is not None:
        assert trans_vec.size == 3

    T = np.eye(4)
    if rot_matrix is not None:
        T[0:3, 0:3] = rot_matrix
    if trans_vec is not None:
        T[0:3, 3] = trans_vec.flatten()
    return T


def from_transform(transform):
    assert transform.shape == (4, 4)

    R = transform[0:3, 0:3]
    t = transform[0:3, 3]
    return R, t


def euler_to_rot_matrix(theta):
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta[0]), -np.sin(theta[0])],
        [0, np.sin(theta[0]), np.cos(theta[0])]
    ])
    R_y = np.array([
        [np.cos(theta[1]), 0, np.sin(theta[1])],
        [0, 1, 0],
        [-np.sin(theta[1]), 0, np.cos(theta[1])]
    ])
    R_z = np.array([
        [np.cos(theta[2]), -np.sin(theta[2]), 0],
        [np.sin(theta[2]), np.cos(theta[2]), 0],
        [0, 0, 1]
    ])
    R = R_z @ R_y @ R_x
    return R


def is_rot_matrix(rot_matrix):
    identity_approx = rot_matrix.T @ rot_matrix
    identity = np.identity(3, dtype=rot_matrix.dtype)
    n = np.linalg.norm(identity - identity_approx)
    return n < 1e-6


def rot_matrix_to_euler(rot_matrix):
    assert (is_rot_matrix(rot_matrix))

    sy = np.sqrt(rot_matrix[0, 0] * rot_matrix[0, 0] + rot_matrix[1, 0] * rot_matrix[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.math.atan2(rot_matrix[2, 1], rot_matrix[2, 2])
        y = np.math.atan2(-rot_matrix[2, 0], sy)
        z = np.math.atan2(rot_matrix[1, 0], rot_matrix[0, 0])
    else:
        x = np.math.atan2(-rot_matrix[1, 2], rot_matrix[1, 1])
        y = np.math.atan2(-rot_matrix[2, 0], sy)
        z = 0

    return np.array([x, y, z])
