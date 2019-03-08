import mpl_toolkits.mplot3d as mplot3d
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

# Our local modules
import ds_tools.shared.transform as tf


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
    cam = tf.apply_transform(cam_orig, T)
    # cam = R @ cam_orig + t
    world_corners = tf.apply_transform(cam_corners, T)
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
