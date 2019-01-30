from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import *
import numpy as np


class RenderApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self, )  # windowType='offscreen')

        # Load the environment model.
        self.scene = self.loader.loadModel("models/environment")
        # Reparent the model to render.
        self.scene.reparentTo(self.render)

        self.temp = 0

        # Apply scale and position transforms on the model.
        # self.scene.setScale(0.25, 0.25, 0.25)
        # self.scene.setPos(-8, 42, 0)

    def init_scene(self):
        render = self.render

        dlight = DirectionalLight('dlight')
        dlight.setColor(VBase4(0.8, 0.8, 0.5, 1))
        dlnp = render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        render.setLight(dlnp)

        alight = AmbientLight('alight')
        alight.setColor(VBase4(0.2, 0.2, 0.2, 1))
        alnp = render.attachNewNode(alight)
        render.setLight(alnp)

        phantom_model = self.loader.loadModel('../data/endo_phantom/mesh2_phantom.stl')
        phantom_model.reparentTo(render)

        # print(phantom_model.getPos())

        markers_model = self.loader.loadModel('../data/endo_phantom/mesh1_endoscope_markers.stl')
        markers_model.reparentTo(render)

        self.taskMgr.add(self.spin_camera_task, "SpinCameraTask")

    def spin_camera_task(self, task):
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (np.pi / 180.0)
        # self.camera.setPos(0, 0, task.time / 10)
        # self.camera.setPos(20 * np.sin(angleRadians), -20.0 * np.cos(angleRadians), 3)
        # self.camera.setHpr(angleDegrees, 0, 0)
        return Task.cont


def main():
    app = RenderApp()
    app.init_scene()
    app.run()
    # app.graphicsEngine.render_frame()
    # app.graphicsEngine.render_frame()
    # file_name = Filename('whatever.png')
    # app.win.saveScreenshot(file_name)


if __name__ == '__main__':
    main()
