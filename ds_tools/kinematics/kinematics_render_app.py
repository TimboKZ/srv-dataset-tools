from panda3d.core import *
import numpy as np

# These names are imported by other scripts so that they can dispatch events to the render app
LoadFrameEventName = 'load-frame-event'
ShutdownEventName = 'shutdown-event'

# Our local modules
from ds_tools.shared.base_render_app import BaseRenderApp
from ds_tools.shared import transform as tf


class KinematicsRenderApp(BaseRenderApp):

    def __init__(self, width=None, height=None, pose_ecm=None, pose_psm=None):
        BaseRenderApp.__init__(self, title='ECM kinematics visualiser', width=width, height=height)

        self.pose_ecm = pose_ecm
        self.pose_psm = pose_psm

        self.ecm = None
        self.ecm_cube = None
        self.psms = [None] * 3

        self.skybox = None
        self.angle = np.array([0, 90, 0])

    def init_scene(self):
        render = self.render

        # Set camera near clip and focal length
        self.main_lens.setNear(0.01)
        self.main_lens.setFocalLength(1)

        # Enable render shaders
        render.setShaderAuto()

        # Create ECM and PSM objects
        self.ecm = NodePath("ecm")
        self.ecm.reparentTo(render)

        # Visualise endoscope manipulator
        ecm_box = self.loader.loadModel('models/box')
        ecm_box.setScale(0.01)
        ecm_box.reparentTo(self.ecm)

        # Bind camera to the endoscope
        # self.disableMouse()
        # self.main_camera_parent.reparentTo(self.ecm)
        # self.camera.setHpr(self.angle[0], self.angle[1], self.angle[2])

        # Setup skybox
        self.skybox = self.loader.loadModel('models/box')
        self.skybox.setBin('background', 0)
        self.skybox.setDepthWrite(False)
        self.skybox.setCompass()
        # self.skybox.reparentTo(self.main_camera_parent)

        # Visualise tool manipulators
        for i in range(3):
            psm = self.loader.loadModel('models/box')
            psm.setScale(0.01)
            psm.reparentTo(self.ecm)
            self.psms[i] = psm

        # Set materials for visible objects
        self.toggleTexture()
        ecm_box.setColor(1, 0, 0, 1)
        self.psms[0].setColor(0, 1, 0, 1)
        self.psms[1].setColor(0, 0, 1, 1)
        self.psms[2].setColor(1, 1, 0, 1)

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
        # self.trackball.node().setPos(0, 2, 0)

        # Setup event listeners
        self.accept(LoadFrameEventName, self.load_frame)
        self.accept(ShutdownEventName, self.shutdown_and_destroy)

    def load_frame(self, frame):
        if self.pose_ecm is not None:
            print('Frame {} Pose ECM:'.format(frame))
            self.apply_pose(self.ecm, self.pose_ecm[frame], log=True)
            print('')

        if self.pose_psm is not None:
            for i in range(3):
                transform_array = self.pose_psm[frame, i * 12:(i + 1) * 12]
                self.apply_pose(self.psms[i], transform_array)

    @staticmethod
    def apply_pose(node, pose_array, log=False):
        assert len(pose_array) == 12

        # Check if kinematics data is available, do nothing if it's not.
        if np.sum(pose_array) == 0:
            return

        R, t = tf.parse_dv_pose(pose_array)
        euler_degrees = np.rad2deg(tf.rot_matrix_to_euler(R))

        if log:
            print('Hpr:', euler_degrees)
            print('Pos:', t)

        node.setHpr(*euler_degrees)
        node.setPos(*t)

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
    #     # TODO: Add some frame testing code
    #
    # start_new_thread(multithreading_events, ())
    #
    # app.run()


if __name__ == '__main__':
    main()
