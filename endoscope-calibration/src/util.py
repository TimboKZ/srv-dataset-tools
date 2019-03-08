import mpl_toolkits.mplot3d as mplot3d
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy.linalg as la
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
    return np.vstack((cart, np.ones((1, cart.shape[1]))))


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


def center_rot(points, rot_matrix):
    """
    Rotates the points around their centroid
    d - number of dimensions
    n - number of points

    :param points: `d x n` array of points
    :param rot_matrix: `d x d` rotation matrix
    :return:
    """

    center = np.mean(points, axis=1)
    result = rot_matrix @ (points.T - center).T
    result = (result.T + center).T
    return result


def undistort_line(line, cam_matrix, dist_coeffs, new_cam_matrix):
    cv_points = np.expand_dims(line.T, axis=0).astype(np.float32)
    new_line = cv.undistortPoints(cv_points,
                                  cam_matrix,
                                  dist_coeffs,
                                  P=new_cam_matrix
                                  )[0].T

    # new_line[0, :] = new_line[0, :]
    # new_line[1, :] = new_line[1, :]
    return new_line


def extract_marker_mesh_centroids(mesh):
    marker_meshes = mesh.split()
    centroids = [m.center_mass for m in marker_meshes]
    return centroids


def find_plane_normal(points):
    """
    d - number of dimensions
    n - number of points

    :param points: `d x n` array of points
    :return: normal vector of the best-fit plane through the points
    """
    mean = np.mean(points, axis=1)
    zero_centre = (points.T - mean.T).T
    U, s, VT = np.linalg.svd(zero_centre)
    normal = U[:, -1]
    return normal


def vecs_to_rot_mat(vec_a, vec_b):
    """
    Finds a `3 x 3` matrix that rotates vector A to vector B
    """
    adb = np.dot(vec_a, vec_b)
    acb = np.cross(vec_a, vec_b)
    bca = np.cross(vec_b, vec_a)

    acb_mag = np.linalg.norm(acb)

    G = np.eye(3)
    G[0, 0] = adb
    G[0, 1] = -acb_mag
    G[1, 0] = acb_mag
    G[1, 1] = adb

    rejec = vec_b - adb * vec_a
    norm_rejec = rejec / la.norm(rejec)

    F = np.vstack([vec_a, norm_rejec, bca]).T

    return F @ G @ la.inv(F)


def prepare_3d_plot(title=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig, ax


def draw_3d_points(ax, points, colour=None, size=10, connect=False, connect_colour='darkgrey',
                   fill=False, fill_colour='lightgray', alpha=0.4):
    """
    d - number of dimensions
    n - number of points

    :param points: `d x n` array of points
    :return:
    """
    n = points.shape[1]

    for i in range(n):
        x, y, z = points[:, i]
        col = None
        if colour is not None:
            col = colour if type(colour) is str else colour[i]
        ax.scatter(x, y, z, c=col, s=size)

    if connect:
        face = mplot3d.art3d.Poly3DCollection([points.T], color=connect_colour, facecolor=fill_colour, alpha=alpha)
        rgba = colors.to_rgba(fill_colour, alpha=alpha)
        face.set_facecolor(rgba)
        ax.add_collection3d(face)


def draw_3d_camera(ax, width, height, cam_matrix, T, z=20):
    """
    Given a 3D PyPlot axis, visualises the viewport of the
    specified camera.

    :param ax: PyPlot axis
    :param width: camera image width
    :param height: camera image height
    :param cam_matrix: `3 x 3` camera intrinsics matrix
    :param T: `4 x 4` transform matrix from camera coordinates to world coordinates
    :param z: depth at which to draw the image plane
    :return:
    """
    fx = cam_matrix[0, 0]
    fy = cam_matrix[1, 1]
    cx = cam_matrix[0, 2]
    cy = cam_matrix[1, 2]

    img_corners = np.array([
        [0, width, width, 0],
        [0, 0, height, height]
    ]).astype(float)
    img_corners[0, :] *= z
    img_corners[1, :] *= z

    cam_corners = img_corners.copy()
    cam_corners[0, :] -= cx * z
    cam_corners[0, :] /= fx
    cam_corners[1, :] -= cy * z
    cam_corners[1, :] /= fy
    cam_corners = np.vstack([cam_corners, np.repeat(z, 4)])

    cam_orig = np.zeros((3, 1))
    cam = apply_transform(cam_orig, T)
    # cam = R @ cam_orig + t
    world_corners = apply_transform(cam_corners, T)
    # world_corners = R @ cam_corners + t

    for i in range(4):
        corners = np.hstack([world_corners[:, [i, (i + 1) % 4]], cam])
        draw_3d_points(ax, corners, colour='black', connect=True,
                       connect_colour='black', fill=True, fill_colour='gray')


def draw_2d_points(ax, points, colour=None, connect=False, size=10):
    """
    d - number of dimensions
    n - number of points

    :param points: `d x n` array of points
    :return:
    """
    n = points.shape[1]

    if connect:
        pts = np.hstack([points, points[:, 0].reshape(2, 1)])
        ax.plot(pts[0, :], pts[1, :])

    for i in range(n):
        x, y = points[:, i]
        col = ''
        if colour is not None:
            col = colour if type(colour) is str else colour[i]
        ax.plot(x, y, color=col, marker='+', markersize=size)
        ax.plot(x, y, color=col, marker='x', markersize=size)


def draw_3d_plane(ax, width, height, normal, point=None, centre=None):
    d = 0
    if point is not None:
        d = -point.dot(normal)
    xs, ys = np.meshgrid(np.arange(-5, 5).astype(float), np.arange(-5, 5).astype(float))
    xs *= width
    ys *= height
    zs = (-normal[0] * xs - normal[1] * ys - d) * 1. / normal[2]

    xs -= np.mean(xs)
    ys -= np.mean(ys)
    zs -= np.mean(zs)

    if centre is not None:
        xs += centre[0]
        ys += centre[1]
        zs += centre[2]

    ax.plot_surface(xs, ys, zs, alpha=0.2)


def main():
    np.set_printoptions(precision=4, suppress=True)

    points = np.array([[-32.01155603, -11.87633243, 33.31950959, 37.93984299, -2.32206122, -25.0494029],
                       [16.61770692, 18.74683867, 24.66148177, -12.85169279, -22.38952365, -24.78481092],
                       [15.19164244, 15.21246656, 15.80259675, -12.87110754, -16.83638323, -16.49921497]])

    normal = find_plane_normal(points)
    front = np.array([1, 0, 0])
    rot = vecs_to_rot_mat(normal, front)

    print(la.det(rot))

    new_normal = find_plane_normal(rot @ points)

    print('normal =', normal)
    print('rot @ normal =', rot @ normal)
    print('new_normal =', new_normal)

    print(repr(rot @ points))


if __name__ == '__main__':
    main()
