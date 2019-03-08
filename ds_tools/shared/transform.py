import numpy as np


def parse_dv_pose(dv_pose):
    """
    Breaks down a vector representing manipulator tip pose (as received from dvLogger data) into a rotation matrix
    and a translation vector.

    The order of elements is as follows (as seen in dvLogger data):
    [R11 R12 R13 tx R21 R22 R23 ty R31 R32 R33 tz]

    :param dv_pose: An array with 9 elements representing the manipulator pose.
    :return: A rotation matrix and a translation vector
    """
    dv_pose = dv_pose.flatten()
    assert len(dv_pose) == 12

    R = np.zeros((3, 3), dtype=np.float32)
    t = np.zeros((3,), dtype=np.float32)

    for i in range(3):
        R[i, :] = dv_pose[i * 4:i * 4 + 3]
        t[i] = dv_pose[i * 4 + 3]

    return R, t


def to_homog(cart):
    """
    :param cart: Cartesian coordinates as a `m x n` matrix, where `m` is the number
                 of dimensions and `n` is the number of points
    :return: Homogeneous coordinates as a `(m+1) x n` matrix
    """
    return np.vstack((cart, np.ones((1, cart.shape[1]))))


def to_cart(homog):
    """
    The opposite of `to_homog` function
    """
    m = homog.shape[0]
    return homog[0:m - 1, :] / np.tile([homog[m - 1, :]], (m - 1, 1))


def apply_transform(cart_points, transform):
    """
    Rotates the points around their centroid
    d - number of dimensions
    n - number of points

    :param cart_points: `d x n` array of cartesian points
    :param transform:
    :return:
    """
    assert transform.shape == (4, 4)

    homog_points = transform @ to_homog(cart_points)
    return to_cart(homog_points)


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
    return n < 1e-4


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
