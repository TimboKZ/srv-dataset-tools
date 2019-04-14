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
    ShaderTextureMode_Visibility = 4
    ShaderTextureMode_Frustum = 5
    ShaderTextureMode_Light = 6
    ShaderTextureMode_END = 7

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

        self.default_bg = [1.0, 1.0, 1.0, 1.0]

        self.last_projection_camera_normal = None

        self.camera_film_size = None
        self.camera_focal_length = None

    def init_scene(self, model_path, texture_path,
                   camera_film_size=None,
                   camera_focal_length=None,
                   normal_map_path=None):
        render = self.render
        self.setBackgroundColor(*self.default_bg)

        # Record film size and focal length of capture
        self.camera_film_size = camera_film_size
        self.camera_focal_length = camera_focal_length

        # Set camera near clip and focal length
        self.main_lens.setNear(0.01)
        self.main_lens.setFocalLength(1)

        # Load the model
        self.model = self.loader.loadModel(model_path, noCache=True)
        self.model.reparentTo(render)

        # Find the actual geom object of the model
        model_geom = self.model
        while len(model_geom.children) != 0:
            if len(model_geom.children) > 1:
                print('[WARN] Model has more than one child!')
            model_geom = model_geom.children[0]
            model_geom.setPos(0, 0, 0)
            model_geom.setHpr(0, 0, 0)

        # Remove all textures from the model
        model_geom.clearTexture()
        # model_geom.clearMaterial()

        # Prepare a material for the model
        model_mat = model_geom.getMaterial()
        model_mat.setShininess(10.0)

        # Add normal map to the texture if one was provided
        if normal_map_path is not None:
            normal_tex = self.loader.loadTexture(normal_map_path)
            normal_ts = TextureStage('model_normal_ts')
            normal_ts.setMode(TextureStage.MNormal)
            # model_geom.setTexture(normal_ts, normal_tex)

        # Explicitly apply the model texture to the model
        texture_cv = cv.imread(texture_path)
        self.texture_height, self.texture_width = texture_cv.shape[:2]
        model_tex = self.loader.loadTexture(texture_path)
        model_ts = TextureStage('model_ts')
        model_ts.setMode(TextureStage.MDecal)
        # model_geom.setTexture(model_ts, model_tex)

        # Enable shaders on model geometry
        # model_geom.setShaderAuto()

        # Attach a point light to the camera
        point_power = 0.5
        point_light = PointLight('point_light')
        point_light.setColor(VBase4(point_power, point_power, point_power, 1))
        point_light_np = self.main_camera_np.attachNewNode(point_light)
        point_light_np.setPos(0, 0, 0)
        render.setLight(point_light_np)

        # Add an ambient light to the scene
        ambient_power = 0.3
        ambient_light = AmbientLight('ambient_light')
        ambient_light.setColor(VBase4(ambient_power, ambient_power, ambient_power, 1))
        ambient_light_np = render.attachNewNode(ambient_light)
        render.setLight(ambient_light_np)

        # Setup the shader with basic parameters
        self.update_shader_state()
        self.model.setShaderInput('ModelTexture', model_tex)
        self.model.setShader(self.uvShader)
        # render.setShaderAuto()

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

        # Optionally add noise (to demonstrate how error changes)
        noise = 0
        if noise > 0:
            camera_hpr += np.random.normal(0, noise, size=3)

        # Create a projection camera in the specified location
        self.proj_buffer.setSize(width, height)
        projection_camera_np = self.makeCamera(self.proj_buffer)
        projection_camera_np.reparentTo(render)
        projection_camera_np.setPos(*camera_pos)
        projection_camera_np.setHpr(*camera_hpr)

        normal_box = self.loader.loadModel('models/box')
        normal_box.reparentTo(projection_camera_np)
        normal_box.setScale(0.0001)
        normal_box.setPos(0, 2, 0)

        normal_box_pos = normal_box.getNetTransform().getPos()
        cam_pos = projection_camera_np.getPos()
        normal = (normal_box_pos - cam_pos).normalized()
        self.last_projection_camera_normal = normal

        # Move actual render camera to the same position
        # self.disableMouse()
        self.main_camera_np.setPos(*camera_pos)
        self.main_camera_np.setHpr(*camera_hpr)

        # Set the lens parameters
        projection_camera = projection_camera_np.node()
        projection_lens = projection_camera.getLens()
        projection_lens.setNear(0.01)
        projection_lens.setFocalLength(1)

        if self.camera_film_size:
            projection_lens.setFilmSize(*self.camera_film_size)
            self.main_lens.setFilmSize(*self.camera_film_size)

        if self.camera_focal_length:
            projection_lens.setFocalLength(self.camera_focal_length)
            self.main_lens.setFocalLength(self.camera_focal_length)

        print('Camera focal length:', projection_lens.getFocalLength())
        print('Camera film size:', projection_lens.getFilmSize())

        # # Uncomment to visualise the camera
        # proj = render.attachNewNode(LensNode('proj'))
        # proj.node().setLens(projection_lens)
        # proj.node().showFrustum()
        # proj.find('frustum').setColor(1, 0, 0, 1)
        # camModel = self.loader.loadModel('camera.egg')
        # camModel.reparentTo(proj)
        # proj.reparentTo(render)
        # proj.setPos(*projection_camera_np.getNetTransform().getPos())
        # proj.setHpr(*projection_camera_np.getNetTransform().getHpr())

        # Set shader parameters
        self.model.setShaderInput('ProjectionTexture', camera_image_texture)
        self.model.setShaderInput('proj_ModelViewMatrix', self.model.getMat(projection_camera_np))
        self.model.setShaderInput('proj_ProjectionMatrix', projection_lens.getProjectionMat())
        self.model.setShaderInput('LightPos', projection_camera_np.getNetTransform().getPos())

        # Delete all unnecessary nodes and objects
        projection_camera_np.removeNode()

    def toggle_shader(self):
        self.shader_texture_mode = (self.shader_texture_mode + 1) % self.ShaderTextureMode_END

        if self.shader_texture_mode == self.ShaderTextureMode_Mask:
            black_bg = [0.0, 0.0, 0.0, 1.0]
            self.setBackgroundColor(*black_bg)
        else:
            self.setBackgroundColor(*self.default_bg)

        print('Shader texture mode:', self.shader_texture_mode)

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

        # print('')
        # print(self.camera.getPos())
        # print(self.camera.getHpr())

        if task is not None:
            return task.cont

    def capture_shader_texture(self, texture_mode):

        # Set shader state for the capture
        self.model.setShaderInput('ShaderViewMode', self.ShaderTextureMode_Projection)
        self.model.setShaderInput('ShaderTextureMode', texture_mode)

        modes_that_need_black_bg = [self.ShaderTextureMode_Mask, self.ShaderTextureMode_Visibility]
        if texture_mode in modes_that_need_black_bg:
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
    data_dir = util.get_data_dir()
    resource_dir = util.get_resource_dir()

    model_path = path.join(data_dir, 'placenta_phantom_capture', 'placenta_mesh.obj')
    texture_path = path.join(data_dir, 'placenta_phantom_capture', 'texture.png')
    normal_map_path = None
    camera_image_path = path.join(resource_dir, 'placenta_phantom_images', '{}_screenshot.png')
    capture_data_json_path = path.join(resource_dir, 'placenta_phantom_images', 'capture_data.json')
    capture_folder_name = 'placenta_phantom_capture'
    base_capture_path = path.join(resource_dir, capture_folder_name, 'base_{}.png')
    texture_capture_path = path.join(resource_dir, capture_folder_name, '{}_{}.png')

    # model_path = path.join(resource_dir, '3d_assets', 'heart.egg')
    # model_path = path.join(resource_dir, '3d_assets', 'heart.obj')
    # texture_path = path.join(resource_dir, '3d_assets', 'T_heart_base3.png')
    # normal_map_path = path.join(resource_dir, '3d_assets', 'T_heart_n.png')
    # camera_image_path = path.join(resource_dir, 'heart_screenshots', '{}_screenshot.png')
    # capture_data_json_path = path.join(resource_dir, 'heart_screenshots', 'capture_data.json')
    # capture_folder_name = 'heart_texture_capture'
    # base_capture_path = path.join(resource_dir, capture_folder_name, 'base_{}.png')
    # texture_capture_path = path.join(resource_dir, capture_folder_name, '{}_{}.png')

    # Load capture data JSON
    capture_json = util.load_dict(capture_data_json_path)
    camera_film_size = capture_json.get('camera_film_size')
    camera_focal_length = capture_json.get('camera_focal_length')
    camera_pos = capture_json['camera_pos']
    camera_hpr = capture_json['camera_hpr']
    camera_normals = []

    # When tex_mode is `true` the app captures reprojected textures in headless mode. When it is false, the app runs
    # in foreground and lets you explore the model.
    tex_mode = True

    renderer = None
    if tex_mode:
        texture_cv = cv.imread(texture_path)
        texture_height, texture_width = texture_cv.shape[:2]
        renderer = TextureMappingRenderApp(width=texture_width, height=texture_height, headless=True)
    else:
        renderer = TextureMappingRenderApp(width=720, height=576, headless=False)

    renderer.init_scene(model_path=model_path,
                        texture_path=texture_path,
                        normal_map_path=normal_map_path,
                        camera_film_size=camera_film_size,
                        camera_focal_length=camera_focal_length)

    def capture_texture(texture_type, name, index=None):
        texture_capture = renderer.capture_shader_texture(texture_type)
        if index is not None:
            save_path = texture_capture_path.format(index, name)
        else:
            save_path = base_capture_path.format(name)
        cv.imwrite(save_path, texture_capture)

    for i in [1]:  # range(len(camera_pos)):
        print('Processing screenshot {}...'.format(i))

        renderer.update_projection(camera_image_path=camera_image_path.format(i),
                                   camera_pos=camera_pos[i],
                                   camera_hpr=camera_hpr[i])

        # Extract camera normal for this screenshot
        camera_normal = renderer.last_projection_camera_normal
        camera_normals.append([camera_normal[0], camera_normal[1], camera_normal[2]])

        if tex_mode:
            if i == 0:
                capture_texture(renderer.ShaderTextureMode_Default, 'texture')
                capture_texture(renderer.ShaderTextureMode_Normal, 'normal')
                capture_texture(renderer.ShaderTextureMode_Mask, 'mask')

            capture_texture(renderer.ShaderTextureMode_Projection, 'projection', i)
            capture_texture(renderer.ShaderTextureMode_Visibility, 'visibility', i)
            capture_texture(renderer.ShaderTextureMode_Frustum, 'frustum', i)
            capture_texture(renderer.ShaderTextureMode_Light, 'light', i)

    if not tex_mode:
        renderer.run()

    # Save camera normals (they will be used for texture reconstruction)
    capture_json['camera_normal'] = camera_normals
    util.save_dict(capture_data_json_path, capture_json)

    renderer.shutdown_and_destroy()


if __name__ == '__main__':
    main()
