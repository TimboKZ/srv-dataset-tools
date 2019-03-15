from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from time import sleep
from os import path
import cv2 as cv
import os

# Our local modules
from ds_tools.shared import util


class BaseRenderApp(ShowBase):

    def __init__(self, title='Panda3d App', width=None, height=None, headless=False):
        self.title = title
        self.width = width
        self.height = height
        self.headless = headless

        if self.width and self.height:
            loadPrcFileData('', 'win-size {} {}'.format(self.width, self.height))

        if self.headless:
            ShowBase.__init__(self, windowType='offscreen')
            self.headless_buffer = self.win
        else:
            ShowBase.__init__(self)
            props = WindowProperties()
            props.setTitle(self.title)
            if self.width and self.height:
                props.setSize(self.width, self.height)
                props.setFixedSize(True)
            self.win.requestProperties(props)

        # Setup references to cameras and lenses
        self.main_camera_parent = self.camera
        for obj in self.camera.getChildren():
            self.main_camera_nodepath = obj
        self.main_camera = self.main_camera_nodepath.node()
        self.main_lens = self.main_camera.getLens()

        # Set default aspect ratio for the camera
        if self.width and self.height:
            aspect_ratio = self.width / self.height
            self.main_lens.setAspectRatio(aspect_ratio)

    def capture_screenshot(self):
        # Render a couple of frames to make sure all changes are visible
        self.graphicsEngine.render_frame()
        self.graphicsEngine.render_frame()

        # Save file into a temp directory
        temp_screenshot_file = util.get_temp_filename(name='src-temp-capture', extension='png')

        def attempt_screenshot():
            if self.headless:
                source = self.headless_buffer
                self.screenshot(temp_screenshot_file, source=source, defaultFilename=False)
            else:
                self.screenshot(temp_screenshot_file, defaultFilename=False)

        # Try to take a screenshot
        if not self.headless:
            print('Be warned - screenshots only work consistently in headless mode. In other modes you might '
                  'experience the issue where the screenshot file is never generated.')
        attempt_screenshot()

        # Load the temp file using OpenCV and delete the original file
        screenshot = cv.imread(temp_screenshot_file)
        os.remove(temp_screenshot_file)
        return screenshot

    def shutdown_and_destroy(self):
        self.shutdown()
        self.destroy()
