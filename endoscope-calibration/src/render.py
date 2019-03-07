from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from time import sleep
from os import path
import numpy as np
import cv2 as cv
import tempfile
import sys
import os

# Necessary to make sure this code works when imported into a Jupyter notebook
script_dir = path.dirname(path.realpath(__file__))
sys.path.append(script_dir)

# Our local modules
import transform as tf


class RenderApp(ShowBase):

    def __init__(self, width, height, headless=False):
        # if headless:
        #     ShowBase.__init__(self, windowType='offscreen')
        # else:
        #     ShowBase.__init__(self)
        ShowBase.__init__(self)

        self.width = width
        self.height = height

        # if headless:
        #     self.win.setSize(self.width, self.height)
        # else:
        #     props = WindowProperties()
        #     props.setTitle("Endoscope Position Preview")
        #     props.setSize(self.width, self.height)
        #     props.setFixedSize(True)
        #     self.win.requestProperties(props)
        # TODO: In headless mode, the window size is incorrect - try to fix it to have the correct size.
        props = WindowProperties()
        props.setTitle("3D render")
        props.setSize(self.width, self.height)
        props.setFixedSize(True)
        self.win.requestProperties(props)

        self.realCamera = None

    def init_scene(self, R, t, cam_matrix, phantom_model_path, endoscope_markers_path=None):
        render = self.render

        # dlight = DirectionalLight('dlight')
        # dlight.setColor(VBase4(1, 1, 1, 1) * 0.5)
        # dlnp = render.attachNewNode(dlight)
        # dlnp.setHpr(0, 60, 0)
        # render.setLight(dlnp)

        # Enable render shaders
        render.setShaderAuto()

        # Add a point light to where the origin of the camera
        plight = PointLight('plight')
        plight.setColor(VBase4(0.2, 0.2, 0.2, 1))
        plight.setAttenuation((1, 0, 0.001))
        plight.setShadowCaster(True, 512, 512)
        plnp = render.attachNewNode(plight)
        plnp.setPos(t[0], t[1], t[2])
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
        fixed_euler_degrees = np.array([0., 90., 0.]) + euler_degrees
        self.camera.setHpr(fixed_euler_degrees[0], fixed_euler_degrees[1], fixed_euler_degrees[2])
        self.camera.setPos(t[0], t[1], t[2])

        # Find camera object and set aspect ratio and focal length
        for obj in self.camera.getChildren():
            self.realCamera = obj.node()
        lens = self.realCamera.getLens()
        aspect_ratio = self.width / self.height
        f_x, f_y = cam_matrix[0, 0], cam_matrix[1, 1]
        # TODO: Determine a better (correct) formula for the focal length
        # focal_length = (f_x / self.width + f_y / self.height) / 2
        focal_length = f_x / self.width
        lens.setAspectRatio(aspect_ratio)
        lens.setFocalLength(focal_length)


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
    width, height = 1280, 1024
    phantom_model_path = '../data/endo_phantom/mesh2_phantom.stl'
    endoscope_markers_path = '../data/endo_phantom/mesh1_endoscope_markers.stl'
    T_cam_to_world = np.array([
        [4.95669255e-01, 8.67596708e-01, -3.98489853e-02, -9.31481022e+01],
        [-8.68225266e-01, 4.96161724e-01, 2.90365040e-03, -1.21905998e+02],
        [2.22907387e-02, 3.31586456e-02, 9.99201495e-01, -1.02540324e+02],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    R, t = tf.from_transform(T_cam_to_world)
    cam_matrix = np.array([
        [321.73571777, 0., 651.79768073],
        [0., 339.54953003, 514.0065655],
        [0., 0., 1.]
    ])

    app = RenderApp(width=width, height=height)
    app.init_scene(R=R, t=t, cam_matrix=cam_matrix, phantom_model_path=phantom_model_path,
                   endoscope_markers_path=endoscope_markers_path)
    app.run()


if __name__ == '__main__':
    main()
