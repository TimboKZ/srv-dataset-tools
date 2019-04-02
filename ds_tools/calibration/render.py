from direct.gui.OnscreenImage import OnscreenImage
from panda3d.core import *
from time import sleep
from os import path
import numpy as np
import cv2 as cv
import tempfile
import os

# Our local modules
from ds_tools.shared.base_render_app import BaseRenderApp
import ds_tools.shared.transform as tf
from ds_tools.shared import cv_util
from ds_tools.shared import util


class RenderApp(BaseRenderApp):

    def __init__(self, width, height, headless=False):
        BaseRenderApp.__init__(self, title='Calibration render', width=width, height=height, headless=headless)

        # self.hpr_adjustment = LVecBase3f(39, 127, 20)
        # self.pos_adjustment = LVecBase3f(-1, -2, 0.5)
        self.hpr_adjustment = LVecBase3f(0)
        self.pos_adjustment = LVecBase3f(0)

        self.onscreen_image = None
        self.onscreen_image_path = None

    def init_scene(self, R, t, cam_matrix, phantom_model_path, endoscope_markers_path=None, onscreen_image_path=None):
        render = self.render

        self.onscreen_image_path = onscreen_image_path

        # Enable render shaders
        render.setShaderAuto()

        # Add a point light to where the origin of the camera
        plight = PointLight('plight')
        plight.setColor(VBase4(0.2, 0.2, 0.2, 1))
        plight.setAttenuation((1, 0, 0.00001))
        plnp = self.main_camera_parent.attachNewNode(plight)
        render.setLight(plnp)

        # Load phantom model and set material
        phantom_model = self.loader.loadModel(phantom_model_path)
        phantom_model.reparentTo(render)

        phantom_material = Material()
        phantom_material.setShininess(20.0)
        phantom_material.setAmbient((0, 0, 0, 1))
        phantom_material.setDiffuse((1, 0, 0, 1))
        phantom_model.setMaterial(phantom_material)

        # Load endoscope markers - these are optional because they might not even be visible in the shot, and they don't
        # give us any useful information about the scene.
        if endoscope_markers_path:
            markers_model = self.loader.loadModel(endoscope_markers_path)
            markers_model.reparentTo(render)

        self.disableMouse()

        # Set camera pose - not that we need to add 90 to pitch for the camera to be aligned correctly.
        euler_degrees = np.rad2deg(tf.rot_matrix_to_euler(R))
        fixed_euler_degrees = [self.hpr_adjustment[0], self.hpr_adjustment[1], self.hpr_adjustment[2]] + euler_degrees
        fixed_pos = [self.pos_adjustment[0], self.pos_adjustment[1], self.pos_adjustment[2]] + t

        fixed_euler_degrees = [-33.073097229003906, 152.8730010986328, 204.61900329589844]
        fixed_pos = [-254.94598388671875, -169.5260009765625, -115.77900695800781]

        self.main_camera_parent.setHpr(*fixed_euler_degrees)
        self.main_camera_parent.setPos(*fixed_pos)

        # Set camera focal length
        f_x, f_y = cam_matrix[0, 0], cam_matrix[1, 1]
        focal_length = f_x
        self.main_lens.setFilmSize(self.width, self.height)
        self.main_lens.setFocalLength(focal_length)

        # fov_x = 2 * np.arctan(self.width / (2 * f_x))
        # fov_x = np.rad2deg(fov_x)
        # fov_y = 2 * np.arctan(self.height / (2 * f_y))
        # fov_y = np.rad2deg(fov_y)
        # self.main_lens.setFov(fov_x, fov_y)

        print('FOV:', self.main_lens.getFov())

        magnitude = 0.5
        self.accept('u', self.adjust_camera_hpr, [LVecBase3f(+magnitude, 0, 0)])
        self.accept('i', self.adjust_camera_hpr, [LVecBase3f(-magnitude, 0, 0)])
        self.accept('j', self.adjust_camera_hpr, [LVecBase3f(0, +magnitude, 0)])
        self.accept('k', self.adjust_camera_hpr, [LVecBase3f(0, -magnitude, 0)])
        self.accept('n', self.adjust_camera_hpr, [LVecBase3f(0, 0, +magnitude)])
        self.accept('m', self.adjust_camera_hpr, [LVecBase3f(0, 0, -magnitude)])

        magnitude = 1
        self.accept('a', self.adjust_camera_pos, [LVecBase3f(+magnitude, 0, 0)])
        self.accept('d', self.adjust_camera_pos, [LVecBase3f(-magnitude, 0, 0)])
        self.accept('q', self.adjust_camera_pos, [LVecBase3f(0, +magnitude, 0)])
        self.accept('e', self.adjust_camera_pos, [LVecBase3f(0, -magnitude, 0)])
        self.accept('w', self.adjust_camera_pos, [LVecBase3f(0, 0, +magnitude)])
        self.accept('s', self.adjust_camera_pos, [LVecBase3f(0, 0, -magnitude)])

        self.accept('space', self.toggle_onscreen_image)
        self.accept('v', self.report_camera_params)

        self.report_camera_params()

    def report_camera_params(self):
        hpr = self.main_camera_parent.getHpr()
        pos = self.main_camera_parent.getPos()
        print('Camera HPR:', [hpr[0], hpr[1], hpr[2]])
        print('Camera pos:', [pos[0], pos[1], pos[2]])
        print('Camera film size:', self.main_lens.getFilmSize())
        print('Camera focal length:', self.main_lens.getFocalLength())

    def toggle_onscreen_image(self):
        if self.onscreen_image_path is None:
            return

        if self.onscreen_image is None:
            size_ratio = self.width / float(self.height)
            onscreen_image = OnscreenImage(image=self.onscreen_image_path,
                                           scale=(size_ratio, 1, 1))
            onscreen_image.setTransparency(TransparencyAttrib.MAlpha)
            self.onscreen_image = onscreen_image
        else:
            self.onscreen_image.destroy()
            self.onscreen_image = None

    def adjust_camera_hpr(self, hpr):
        old_hpr = self.main_camera_parent.getHpr()
        new_hpr = old_hpr + hpr
        self.main_camera_parent.setHpr(*new_hpr)

        self.hpr_adjustment += hpr
        print('HPR adjustment:', self.hpr_adjustment)

    def adjust_camera_pos(self, pos):
        old_pos = self.main_camera_parent.getPos()
        new_pos = old_pos + pos
        self.main_camera_parent.setPos(*new_pos)

        self.pos_adjustment += pos
        print('Pos adjustment:', self.pos_adjustment)


def generate_render(width, height, T_cam_to_world, cam_matrix, phantom_model_path, endoscope_markers_path):
    # Initialise the render
    R, t = tf.from_transform(T_cam_to_world)
    app = RenderApp(width=width, height=height, headless=True)
    app.init_scene(R=R, t=t, cam_matrix=cam_matrix, phantom_model_path=phantom_model_path,
                   endoscope_markers_path=endoscope_markers_path)

    # Render a couple of frames to make sure all changes are visible
    app.graphicsEngine.render_frame()
    app.graphicsEngine.render_frame()
    sleep(0.5)
    app.graphicsEngine.render_frame()
    app.graphicsEngine.render_frame()
    app.graphicsEngine.render_frame()
    app.graphicsEngine.render_frame()
    app.graphicsEngine.render_frame()
    app.graphicsEngine.render_frame()

    # Save file into a temp directory
    temp_screenshot_file = path.join(tempfile.gettempdir(), 'src-temp-capture-{}.png'.format(np.random.randint(1000)))
    app.screenshot(temp_screenshot_file, defaultFilename=False)

    # Shutdown the graphics engine
    app.shutdown()
    app.destroy()

    # Load the temp file using OpenCV and delete the original file
    image = cv.imread(temp_screenshot_file)
    os.remove(temp_screenshot_file)
    return image


def main():
    # This function is here for testing purposes
    width, height = 720, 576
    data_dir = util.get_data_dir()
    capture_dir = path.join(data_dir, 'placenta_phantom_capture')
    phantom_model_path = path.join(capture_dir, 'placenta_mesh.stl')
    endoscope_markers_path = path.join(capture_dir, 'endo_markers.stl')
    T_cam_to_world = np.array([
        [-9.68176662e-01, 1.67576317e-01, - 1.85882030e-01, - 2.29945901e+02],
        [3.18028233e-02, - 6.54331106e-01, - 7.55539135e-01, - 1.73525540e+02],
        [-2.48238860e-01, - 7.37406931e-01, 6.28178706e-01, - 1.28278633e+02],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    R, t = tf.from_transform(T_cam_to_world)
    cam_matrix = np.array([
        [773.25415039, 0., 384.09088309],
        [0., 842.88543701, 276.08463295],
        [0., 0., 1.],
    ])

    test_headless = False
    if test_headless:
        screenshot = generate_render(width, height, T_cam_to_world,
                                     cam_matrix, phantom_model_path, endoscope_markers_path)
        cv.imshow("Calibration render preview", screenshot)
        print(screenshot.shape)
        cv_util.wait_for_esc()
    else:
        app = RenderApp(width=width, height=height)
        app.init_scene(R=R, t=t, cam_matrix=cam_matrix, phantom_model_path=phantom_model_path,
                       endoscope_markers_path=endoscope_markers_path,
                       onscreen_image_path=path.join(util.get_resource_dir(), 'placenta_phantom_images',
                                                     '5_screenshot.png'))
        app.run()


if __name__ == '__main__':
    main()
