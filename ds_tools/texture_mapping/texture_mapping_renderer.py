from pandac.PandaModules import ProjectionScreen
from panda3d.core import *
from os import path
import cv2 as cv

# Our local modules
from ds_tools.shared.base_render_app import BaseRenderApp
from ds_tools.shared import util


class TextureMappingRenderApp(BaseRenderApp):
    ShaderMode_3D = 0
    ShaderMode_Texture = 1

    def __init__(self, width=None, height=None, headless=False):
        BaseRenderApp.__init__(self, title='Texture mapping visualiser', width=width, height=height, headless=headless)

        self.model = None

        resource_dir = util.get_resource_dir()
        vertex_path = path.join(resource_dir, 'texture_uv_shader.vert')
        fragment_path = path.join(resource_dir, 'texture_uv_shader.frag')

        self.shaderMode = self.ShaderMode_3D
        self.textureMode = 0
        self.uvShader = Shader.load(Shader.SL_GLSL, vertex=vertex_path, fragment=fragment_path)

    def init_scene(self, model_path, texture_path, camera_image_path, camera_pos, camera_hpr):
        render = self.render
        self.setBackgroundColor(1, 1, 1, 1)

        # Set camera near clip and focal length
        self.main_lens.setNear(0.01)
        self.main_lens.setFocalLength(1)

        # Put the camera into correct state
        # self.disableMouse()
        # self.camera.setPos(camera_pos[0], camera_pos[1], camera_pos[2])
        # self.camera.setHpr(camera_hpr[0], camera_hpr[1], camera_hpr[2])

        # Load the model
        self.model = self.loader.loadModel(model_path, noCache=True)
        self.model.reparentTo(render)

        # Explicitly apply the model texture to the model
        model_tex = self.loader.loadTexture(texture_path)
        self.model.setTexture(model_tex)

        # Import the camera image
        camera_image_cv = cv.imread(camera_image_path)
        camera_image_height = camera_image_cv.shape[0]
        camera_image_width = camera_image_cv.shape[1]
        camera_image_texture = self.loader.loadTexture(camera_image_path)
        # camera_image_texture.setWrapU(Texture.WMClamp)
        # camera_image_texture.setWrapV(Texture.WMClamp)

        # Pass the camera image to the shader
        self.model.setShaderInput('ModelTexture', model_tex)
        self.model.setShaderInput('ProjectionTexture', camera_image_texture)

        # Create a projection camera in the specified location
        proj_buffer = self.win.makeTextureBuffer('Projection Render', camera_image_width, camera_image_height)
        proj_buffer.setSort(-100)
        projection_camera_np = self.makeCamera(proj_buffer)
        projection_camera_np.reparentTo(render)
        projection_camera_np.setPos(camera_pos[0], camera_pos[1], camera_pos[2])
        projection_camera_np.setHpr(camera_hpr[0], camera_hpr[1], camera_hpr[2])

        # Set the lens parameters
        projection_camera = projection_camera_np.node()
        projection_lens = projection_camera.getLens()
        projection_lens.setNear(0.01)
        projection_lens.setFocalLength(1)

        # Send the details of projection camera to the shader
        self.model.setShaderInput('proj_ModelViewMatrix', render.getMat(projection_camera_np))
        self.model.setShaderInput('proj_ProjectionMatrix', projection_lens.getProjectionMat())
        self.update_shader_state()
        self.model.setShader(self.uvShader)

        self.taskMgr.add(self.update_shader_state, 'UpdateShaderStateTask')
        self.accept('space', self.toggle_shader_mode)

        # Enable render shaders
        # render.setShaderAuto()

    def toggle_shader_mode(self):
        if self.shaderMode == self.ShaderMode_3D:
            self.shaderMode = self.ShaderMode_Texture
        else:
            self.shaderMode = self.ShaderMode_3D

    def update_shader_state(self, task=None):
        self.model.setShaderInput('ShaderMode', self.shaderMode)
        self.model.setShaderInput('TextureMode', self.textureMode)

        if task is not None:
            return task.cont


def main():
    resource_dir = util.get_resource_dir()
    model_path = path.join(resource_dir, 'woodbox.obj')
    texture_path = path.join(resource_dir, 'woodbox.jpg')
    camera_image_path = path.join(resource_dir, 'box_image.png')
    camera_pos = [-5, -5, 5]
    camera_hpr = [-45, -33, 0]

    camera_image = cv.imread(camera_image_path)
    camera_image_height = camera_image.shape[0]
    camera_image_width = camera_image.shape[1]

    renderer = TextureMappingRenderApp(width=camera_image_width, height=camera_image_height, headless=False)
    renderer.init_scene(model_path=model_path,
                        texture_path=texture_path,
                        camera_image_path=camera_image_path,
                        camera_pos=camera_pos,
                        camera_hpr=camera_hpr)

    renderer.run()
    # screenshot = renderer.capture_screenshot()
    # # cv.imwrite(path.join(resource_dir, 'box_image.png'), screenshot)
    # cv.imshow('Screenshot', screenshot)
    # cv.waitKey()

    renderer.shutdown_and_destroy()


if __name__ == '__main__':
    main()
