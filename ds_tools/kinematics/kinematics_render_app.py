from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
import numpy as np
import sys

# These names are imported by other scripts so that they can dispatch events to the render app
LoadFrameEventName = 'load-frame-event'
ShutdownEventName = 'shutdown-event'

# Our local modules
from ds_tools.shared import transform as tf


class KinematicsRenderApp(ShowBase):

    def __init__(self, pose_ecm, pose_psm):
        ShowBase.__init__(self)

        self.pose_ecm = pose_ecm
        self.pose_psm = pose_psm

        props = WindowProperties()
        props.setTitle("ECM kinematics visualiser")
        self.win.requestProperties(props)

        self.realCamera = None

        self.ecm = None
        self.ecm_cube = None
        self.psms = [None] * 3

        self.angle = 0

    def init_scene(self):
        render = self.render

        # Find camera object and set aspect ratio and focal length
        for obj in self.camera.getChildren():
            self.realCamera = obj.node()
        lens = self.realCamera.getLens()
        lens.setFocalLength(2)

        # Enable render shaders
        render.setShaderAuto()

        # Create ECM and PSM objects
        self.ecm = NodePath("ecm")
        self.ecm.reparent_to(render)

        self.ecm_cube = self.loader.loadModel('models/box')
        self.ecm_cube.reparent_to(self.ecm)
        self.ecm_cube.setScale(0.1)
        self.ecm_cube.setPos(0, 0, 0)

        for i in range(3):
            psm = self.loader.loadModel('models/box')
            psm.setScale(0.05)
            psm.reparent_to(self.ecm)
            self.psms[i] = psm

        # Hide the inactive PSM for the time being
        self.psms[2].hide()

        # Set materials for visible objects
        self.toggleTexture()
        self.ecm.setColor(1, 0, 0, 1)
        self.psms[0].setColor(0, 1, 0, 1)
        self.psms[1].setColor(0, 0, 1, 1)

        # Add an ambient light to make things brighter
        alight = AmbientLight('alight')
        alight.setColor(VBase4(0.2, 0.2, 0.2, 1))
        alnp = render.attachNewNode(alight)
        render.setLight(alnp)

        # Add a point light to the tip of ECM
        plight = PointLight('plight')
        plight.setColor(VBase4(0.2, 0.2, 0.2, 1))
        plight.setAttenuation((1, 0, 0.001))
        # plight.setShadowCaster(True, 512, 512)
        plnp = render.attachNewNode(plight)
        plnp.reparentTo(self.ecm)
        plnp.setPos(0, 0, 0)
        render.setLight(plnp)

        # Set camera position
        self.trackball.node().setPos(0, 2, 0)

        # Setup event listeners
        self.accept(LoadFrameEventName, self.load_frame)
        self.accept(ShutdownEventName, self.shutdown_and_destroy)

        # Move the camera to where the endoscope would be
        # self.disableMouse()
        # self.camera.reparentTo(self.ecm)
        # self.camera.setPos(0, 0, 0)
        # self.camera.setHpr(440, 0, 0)

    def load_frame(self, frame):
        self.apply_transform(self.ecm, self.pose_ecm[frame])
        for i in range(3):
            transform_array = self.pose_psm[frame, i * 12:(i + 1) * 12]
            self.apply_transform(self.psms[i], transform_array)

        # self.angle += 1
        # self.camera.setHpr(self.angle, 0, 0)

    def apply_transform(self, node, transform_array):
        assert len(transform_array) == 12

        R, t = tf.parse_dv_pose(transform_array)
        euler_degrees = np.rad2deg(tf.rot_matrix_to_euler(R))
        node.setHpr(euler_degrees[0], euler_degrees[1], euler_degrees[2])
        node.setPos(t[0], t[1], t[2])

    def shutdown_and_destroy(self):
        self.shutdown()
        self.destroy()


def main():
    arr = np.array([
        0.962770, 0.256150, -0.086365, 0.009218,
        0.261360, -0.800530, 0.539290, 0.011552,
        0.068999, -0.541790, -0.837680, -0.172370
    ])
    R, t = tf.parse_dv_pose(arr)
    euler_degrees = np.rad2deg(tf.rot_matrix_to_euler(R))

    # This function is here for testing purposes
    # app = EcmRenderApp()
    # app.init_scene()
    #
    # def multithreading_events():
    #     uno = np.ones((8,))
    #     do = np.ones((8,)) * 2
    #
    #     sleep(5)
    #
    #     # TODO: Add some frame testing code ghy
    #
    # start_new_thread(multithreading_events, ())
    #
    # app.run()
    pass


if __name__ == '__main__':
    main()
