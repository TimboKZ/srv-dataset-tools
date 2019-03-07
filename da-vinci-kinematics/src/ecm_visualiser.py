from direct.showbase.ShowBase import ShowBase
from _thread import start_new_thread
from panda3d.core import *
import transform as tf
from time import sleep
from os import path
import numpy as np

script_dir = path.dirname(path.realpath(__file__))

LoadFrameEventName = 'load-frame-event'
ShutdownEventName = 'shutdown-event'


# Our local modules
class EcmRenderApp(ShowBase):

    def __init__(self, pose_ecm, pose_psm):
        ShowBase.__init__(self)

        self.pose_ecm = pose_ecm
        self.pose_psm = pose_psm

        props = WindowProperties()
        props.setTitle("ECM kinematics visualiser")
        self.win.requestProperties(props)

        self.realCamera = None

        self.ecm = None
        self.psms = [None] * 3

    def init_scene(self):
        render = self.render

        # Enable render shaders
        render.setShaderAuto()

        # Create ECM and PSM objects
        self.ecm = self.loader.loadModel('models/box')
        self.ecm.reparent_to(render)
        for i in range(3):
            psm = self.loader.loadModel('models/box')
            psm.setScale(0.5)
            psm.reparent_to(self.ecm)
            self.psms[i] = psm

        # Hide the inactive PSM for the time being
        self.psms[2].hide()

        # Set camera position
        self.trackball.node().setPos(0, 60, 0)

        # Setup event listeners
        self.accept(LoadFrameEventName, self.load_frame)
        self.accept(ShutdownEventName, self.shutdown_and_destroy)

        # self.disableMouse()
        #
        # # Set camera pose - not that we need to add 90 to pitch for the camera to be aligned correctly.
        # self.camera.setHpr(fixed_euler_degrees[0], fixed_euler_degrees[1], fixed_euler_degrees[2])
        # self.camera.setPos(t[0], t[1], t[2])
        #
        # # Find camera object and set aspect ratio and focal length
        # for obj in self.camera.getChildren():
        #     self.realCamera = obj.node()
        # lens = self.realCamera.getLens()
        # lens.setAspectRatio(aspect_ratio)
        # lens.setFocalLength(focal_length)

    def load_frame(self, frame):
        self.apply_transform(self.ecm, self.pose_ecm[frame])
        for i in range(3):
            transform_array = self.pose_psm[frame, i * 12:(i + 1) * 12]
            self.apply_transform(self.psms[i], transform_array)
        pass

    def apply_transform(self, node, transform_array):
        assert len(transform_array) == 12
        t = transform_array[0:3]
        R = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            start = 3 + i * 3
            end = start + 3
            R[i, :] = transform_array[start:end]

        # euler_degrees = np.rad2deg(tf.rot_matrix_to_euler(R))

        # node.setHpr(euler_degrees[0], euler_degrees[1], euler_degrees[2])
        node.setPos(t[0], t[1], t[2])

    def shutdown_and_destroy(self):
        self.shutdown()
        self.destroy()


def main():
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
