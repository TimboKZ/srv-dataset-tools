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


# Use SPACE key to toggle on-screen image
# Use U,I,J,K,N,M keys to translate the camera
# Use Q,W,E,A,S,D keys to rotate the camera
# Use V key to report camera pose


class RenderApp(BaseRenderApp):

    def __init__(self, width, height, headless=False):
        BaseRenderApp.__init__(self, title='Calibration render', width=width, height=height, headless=headless)

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
        self.toggleTexture()

        # Load endoscope markers - these are optional because they might not even be visible in the shot, and they don't
        # give us any useful information about the scene.
        if endoscope_markers_path:
            markers_model = self.loader.loadModel(endoscope_markers_path)
            markers_model.reparentTo(render)

        # Extract camera intrinsics from camera matrix
        fx = cam_matrix[0, 0]
        fy = cam_matrix[1, 1]
        cx = cam_matrix[0, 2]
        cy = cam_matrix[1, 2]

        # Set camera pose - not that we need to add 90 to pitch for the camera to be aligned correctly.
        fixed_pos = [self.pos_adjustment[0], self.pos_adjustment[1], self.pos_adjustment[2]] + t

        # Visualise frustum of the camera
        z = 20
        img_corners = np.array([
            [0, self.width, self.width, 0],
            [0, 0, self.height, self.height]
        ]).astype(float)
        img_corners[0, :] *= z
        img_corners[1, :] *= z
        cam_corners = img_corners.copy()
        cam_corners[0, :] -= cx * z
        cam_corners[0, :] /= fx
        cam_corners[1, :] -= cy * z
        cam_corners[1, :] /= fy
        cam_corners = np.vstack([cam_corners, np.repeat(z, 4)])
        T = tf.to_transform(R, t)
        world_corners = tf.apply_transform(cam_corners, T)
        cube = render.attachNewNode("centre_node")
        pos = np.mean(world_corners, axis=1)
        cube.setPos(*pos)
        cube2 = render.attachNewNode("top_node")
        pos = np.mean(world_corners[:, :2], axis=1)
        cube2.setPos(*pos)
        proj = render.attachNewNode(LensNode('proj'))
        proj.setPos(*fixed_pos)
        proj.lookAt(cube)

        # Calculate FOV and roll of the camera
        fov_x = 2 * np.arctan(self.width / (2 * fx))
        fov_x = np.rad2deg(fov_x)
        fov_y = 2 * np.arctan(self.height / (2 * fy))
        fov_y = np.rad2deg(fov_y)
        pos1 = cube.getPos(proj)
        pos2 = cube2.getPos(proj)
        pos1[1], pos2[1] = 0, 0
        correct_up = (pos2 - pos1).normalized()
        angle = np.rad2deg(np.arcsin(-correct_up[0]))

        # Apply camera parameters
        self.disableMouse()
        self.main_camera_parent.setPos(*fixed_pos)
        self.main_camera_parent.lookAt(cube)
        self.main_camera_parent.setR(-angle)
        self.main_lens.setFov(fov_x, fov_y)
        self.main_lens.setNear(0.5)

        magnitude = 3
        self.accept('u', self.adjust_camera_hpr, [LVecBase3f(+magnitude, 0, 0)])
        self.accept('i', self.adjust_camera_hpr, [LVecBase3f(-magnitude, 0, 0)])
        self.accept('j', self.adjust_camera_hpr, [LVecBase3f(0, +magnitude, 0)])
        self.accept('k', self.adjust_camera_hpr, [LVecBase3f(0, -magnitude, 0)])
        self.accept('n', self.adjust_camera_hpr, [LVecBase3f(0, 0, +magnitude)])
        self.accept('m', self.adjust_camera_hpr, [LVecBase3f(0, 0, -magnitude)])

        magnitude = 3
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
        print('\n')
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
    capture_dir = path.join(data_dir, 'placenta_phantom', 'placenta_scene')
    phantom_model_path = path.join(capture_dir, 'placenta_mesh.stl')
    endoscope_markers_path = path.join(capture_dir, 'endo_markers.stl')
    onscreen_image_path = path.join(capture_dir, 'scan_6_rectified.png')

    phantom_scene_json = path.join(data_dir, 'phantom_scene.json')
    phantom_scene = util.load_dict(phantom_scene_json)
    T_cam_to_world = np.array(phantom_scene['cam_to_world_transform'])
    cam_matrix = np.array(phantom_scene['camera_matrix'])
    R, t = tf.from_transform(T_cam_to_world)

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
                       onscreen_image_path=onscreen_image_path)
        app.run()


if __name__ == '__main__':
    main()
