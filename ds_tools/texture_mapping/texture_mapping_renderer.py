from panda3d.core import *
from os import path
import numpy as np
import cv2 as cv

# Our local modules
from ds_tools.shared.base_render_app import BaseRenderApp
from ds_tools.shared import transform as tf
from ds_tools.shared import util


class TextureMappingRenderApp(BaseRenderApp):
    ShaderViewMode_3D = 0
    ShaderViewMode_Texture = 1

    ShaderTextureMode_Default = 0
    ShaderTextureMode_Projection = 1
    ShaderTextureMode_Normal = 2
    ShaderTextureMode_Mask = 3
    ShaderTextureMode_END = 4

    def __init__(self, width=None, height=None, headless=False):
        BaseRenderApp.__init__(self, title='Texture mapping visualiser', width=width, height=height, headless=headless)

        self.model = None

        resource_dir = util.get_resource_dir()
        vertex_path = path.join(resource_dir, '3d_assets', 'texture_uv_shader.vert')
        fragment_path = path.join(resource_dir, '3d_assets', 'texture_uv_shader.frag')

        self.shader_view_mode = self.ShaderViewMode_3D
        self.shader_texture_mode = self.ShaderTextureMode_Projection
        self.uvShader = Shader.load(Shader.SL_GLSL, vertex=vertex_path, fragment=fragment_path)

        self.proj_buffer = None
        self.tex_buffer = None
        self.tex_buffer_texture = None

        self.screenshot_count = 0
        self.texture_width = None
        self.texture_height = None

        # self.default_bg = [0.0, 0.0, 0.0, 1.0]
        self.default_bg = [1.0, 1.0, 1.0, 1.0]

    def init_scene(self, model_path, texture_path):
        render = self.render
        self.setBackgroundColor(*self.default_bg)

        # Set camera near clip and focal length
        self.main_lens.setNear(0.01)
        self.main_lens.setFocalLength(1)

        # Load the model
        self.model = self.loader.loadModel(model_path, noCache=True)
        self.model.reparentTo(render)

        # Explicitly apply the model texture to the model
        texture_cv = cv.imread(texture_path)
        self.texture_height, self.texture_width = texture_cv.shape[:2]
        model_tex = self.loader.loadTexture(texture_path)
        self.model.setTexture(model_tex)

        # Setup the shader with basic parameters
        self.update_shader_state()
        self.model.setShaderInput('ModelTexture', model_tex)
        self.model.setShader(self.uvShader)

        # Create a buffer for projection images. This will be resized as requested in `self.update_projection()`.
        self.proj_buffer = self.win.makeTextureBuffer('Projection Render', 800, 600)
        self.proj_buffer.setSort(100)

        # Prepare the buffer that texture in UV-space will be rendered to
        self.tex_buffer = self.win.makeTextureBuffer('Texture Render', self.texture_width, self.texture_height)
        self.tex_buffer.setSort(-100)
        self.tex_buffer_texture = self.tex_buffer.getTexture()
        self.makeCamera(self.tex_buffer)

        # Setup per-frame tasks and listeners
        self.taskMgr.add(self.update_shader_state, 'UpdateShaderStateTask')
        self.accept('space', self.toggle_shader_mode)
        self.accept('c', self.toggle_shader)
        self.accept('b', self.capture_camera_screenshot)

    def update_projection(self, camera_image_path, camera_pos, camera_hpr):
        render = self.render

        # Import the camera image
        camera_image_cv = cv.imread(camera_image_path)
        height, width = camera_image_cv.shape[:2]
        camera_image_texture = self.loader.loadTexture(camera_image_path)

        # Create a projection camera in the specified location
        self.proj_buffer.setSize(width, height)
        projection_camera_np = self.makeCamera(self.proj_buffer)
        projection_camera_np.reparentTo(render)
        projection_camera_np.setPos(*camera_pos)
        projection_camera_np.setHpr(*camera_hpr)

        box = self.loader.loadModel('models/box')
        box.reparentTo(self.main_camera_np)
        box.setScale(0.01)
        box.setPos(0, 2, 0)

        box_pos = box.getNetTransform().getPos()
        cam_pos = self.main_camera_parent.getPos()
        normal = (box_pos - cam_pos).normalized()
        print('\nPanda3d normal:', normal)

        # Move actual render camera to the same position
        self.disableMouse()
        self.main_camera_parent.setPos(*camera_pos)
        self.main_camera_parent.setHpr(*camera_hpr)

        box_pos = box.getNetTransform().getPos()
        cam_pos = self.main_camera_parent.getPos()
        normal = (box_pos - cam_pos).normalized()
        print('\nPanda3d normal:', normal)

        # Set the lens parameters
        projection_camera = projection_camera_np.node()
        projection_lens = projection_camera.getLens()
        projection_lens.setNear(0.01)
        projection_lens.setFocalLength(1)

        # Set shader parameters
        self.model.setShaderInput('ProjectionTexture', camera_image_texture)
        self.model.setShaderInput('proj_ModelViewMatrix', render.getMat(projection_camera_np))
        self.model.setShaderInput('proj_ProjectionMatrix', projection_lens.getProjectionMat())

        # Delete all unnecessary
        projection_camera_np.removeNode()

        # Visualise the normal
        R = tf.euler_to_rot_matrix(np.deg2rad(camera_hpr))
        front = np.array(LVector3f.forward())
        camera_normal = (R @ front).flatten()
        print('Calculated normal:', camera_normal)

        # Rotate the normal
        np_normal = np.array(normal)
        print('Rot:', np.linalg.inv(R) @ np_normal)

    def toggle_shader(self):
        self.shader_texture_mode = (self.shader_texture_mode + 1) % self.ShaderTextureMode_END

        if self.shader_texture_mode == self.ShaderTextureMode_Mask:
            black_bg = [0.0, 0.0, 0.0, 1.0]
            self.setBackgroundColor(*black_bg)
        else:
            self.setBackgroundColor(*self.default_bg)

    def toggle_shader_mode(self):
        if self.shader_view_mode == self.ShaderViewMode_3D:
            self.shader_view_mode = self.ShaderViewMode_Texture
            if not self.headless:
                wp = WindowProperties()
                wp.setSize(self.texture_width, self.texture_height)
                self.win.requestProperties(wp)

        else:
            self.shader_view_mode = self.ShaderViewMode_3D
            if not self.headless:
                wp = WindowProperties()
                wp.setSize(self.width, self.height)
                self.win.requestProperties(wp)

    def update_shader_state(self, task=None):
        self.model.setShaderInput('ShaderViewMode', self.shader_view_mode)
        self.model.setShaderInput('ShaderTextureMode', self.shader_texture_mode)

        if task is not None:
            return task.cont

    def capture_shader_texture(self, texture_mode):

        # Set shader state for the capture
        self.model.setShaderInput('ShaderViewMode', self.ShaderTextureMode_Projection)
        self.model.setShaderInput('ShaderTextureMode', texture_mode)

        if texture_mode == self.ShaderTextureMode_Mask:
            black_bg = [0.0, 0.0, 0.0, 1.0]
            self.setBackgroundColor(*black_bg)
        else:
            self.setBackgroundColor(*self.default_bg)

        # Capture the screenshot
        texture_capture = self.capture_screenshot(source=self.tex_buffer)

        # Revert shader state to its original values
        self.update_shader_state()

        return texture_capture

    def capture_camera_screenshot(self):
        print('\nScreenshot:')
        print('  - Cam pos:', self.camera.getPos())
        print('  - Cam hpr:', self.camera.getHpr())

        screenshot = self.capture_screenshot()
        save_path = path.join(util.get_resource_dir(), '{}_screenshot.png'.format(self.screenshot_count))
        cv.imwrite(save_path, screenshot)
        self.screenshot_count += 1


def main():
    resource_dir = util.get_resource_dir()
    model_path = path.join(resource_dir, '3d_assets', 'woodbox.obj')
    texture_path = path.join(resource_dir, '3d_assets', 'woodbox.png')
    camera_image_path = path.join(resource_dir, 'box_screenshots', '{}_screenshot.png')
    capture_data_json_path = path.join(resource_dir, 'capture_data.json')

    # Load capture data JSON
    capture_json = util.load_dict(capture_data_json_path)
    camera_pos = capture_json['camera_pos']
    camera_hpr = capture_json['camera_hpr']
    camera_normal = capture_json['camera_normal']

    # camera_normals = []
    # for i in range(len(camera_hpr)):
    #     R = tf.euler_to_rot_matrix(np.deg2rad(camera_hpr[i]))
    #     front = np.zeros((3, 1))
    #     front[2] = 1.0
    #     camera_normal = (R @ front).flatten()
    #     camera_normals.append(camera_normal.tolist())
    #
    # my_dict = {'camera_pos': camera_pos, 'camera_hpr': camera_hpr, 'camera_normal': camera_normals}
    # util.save_dict(path.join(resource_dir, 'capture_data.json'), my_dict)
    # return

    tex_mode = False
    renderer = None
    if tex_mode:
        texture_cv = cv.imread(texture_path)
        texture_height, texture_width = texture_cv.shape[:2]
        renderer = TextureMappingRenderApp(width=texture_width, height=texture_height, headless=True)
    else:
        renderer = TextureMappingRenderApp(width=720, height=576, headless=False)

    renderer.init_scene(model_path=model_path,
                        texture_path=texture_path)

    base_capture_path = path.join(resource_dir, 'texture_capture', 'base_{}.png')
    texture_capture_path = path.join(resource_dir, 'texture_capture', '{}_{}.png')

    def capture_texture(texture_type, name, index=None):
        texture_capture = renderer.capture_shader_texture(texture_type)
        if index is not None:
            save_path = texture_capture_path.format(index, name)
        else:
            save_path = base_capture_path.format(name)
        cv.imwrite(save_path, texture_capture)

    for i in range(3, 4):
        renderer.update_projection(camera_image_path=camera_image_path.format(i),
                                   camera_pos=camera_pos[i],
                                   camera_hpr=camera_hpr[i])

        if tex_mode:
            if i == 0:
                capture_texture(renderer.ShaderTextureMode_Default, 'texture')
                capture_texture(renderer.ShaderTextureMode_Normal, 'normal')
                capture_texture(renderer.ShaderTextureMode_Mask, 'mask')

            capture_texture(renderer.ShaderTextureMode_Projection, 'projection', i)

    if not tex_mode:
        renderer.run()

    renderer.shutdown_and_destroy()


if __name__ == '__main__':
    main()
