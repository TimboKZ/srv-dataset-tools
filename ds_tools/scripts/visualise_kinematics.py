from _thread import start_new_thread
from os import path
import numpy as np
import cv2 as cv

# Our local modules
from ds_tools.kinematics.kinematics_render_app import KinematicsRenderApp, LoadFrameEventName, ShutdownEventName
from ds_tools.shared import util, cv_util


def load_numpy_csv(csv_path):
    return np.loadtxt(csv_path, delimiter=',', dtype=np.float32)


def init_video_viewer(video_capture, ecm_renderer):
    WindowName = 'Kinematics Viz'
    TrackbarName = 'Frame'

    def video_viewer_routine():
        frame_count = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
        last_loop_frame = 0

        def on_trackbar_change(frame):
            ecm_renderer.messenger.send(LoadFrameEventName, [frame])

            if last_loop_frame != frame:
                video_capture.set(cv.CAP_PROP_POS_FRAMES, frame)
                err, img = video_capture.read()
                cv.imshow(WindowName, img)

        cv.namedWindow(WindowName)
        cv.moveWindow(WindowName, 100, 100)
        cv.createTrackbar(TrackbarName, WindowName, 0, frame_count - 1, on_trackbar_change)

        on_trackbar_change(1)

        paused = True

        while video_capture.isOpened():
            if paused:
                key = cv.waitKey(10) & 0xff
                if key == 32:
                    paused = False
                elif key == 27:
                    if ecm_renderer:
                        ecm_renderer.messenger.send(ShutdownEventName)
                    break
                else:
                    continue

            err, img = video_capture.read()
            cv.imshow(WindowName, img)

            curr_frame = int(video_capture.get(cv.CAP_PROP_POS_FRAMES))
            last_loop_frame = curr_frame
            cv.setTrackbarPos(TrackbarName, WindowName, curr_frame)

            key = cv.waitKey(10) & 0xff
            if key == 27:
                if ecm_renderer:
                    ecm_renderer.messenger.send(ShutdownEventName)
                break
            elif key == 32:
                paused = True

        cv.destroyAllWindows()
        video_capture.release()

    start_new_thread(video_viewer_routine, ())


def main():
    # Tweak these paths to match your data
    data_dir = util.get_data_dir()
    capture_dir = path.join(data_dir, 'new_phantom_capture_p1')
    video_path = path.join(capture_dir, 'EndoscopeImageMemory_0_small.mp4')
    pose_ecm = load_numpy_csv(path.join(capture_dir, 'pose_ecm.csv'))
    pose_psm = load_numpy_csv(path.join(capture_dir, 'pose_psm.csv'))

    # Prepare video capture
    cap = cv.VideoCapture(video_path)
    width, height = cv_util.get_capture_size(cap)

    # Setup the 3D renderer
    ecm_renderer = KinematicsRenderApp(width=width, height=height,
                                       pose_ecm=pose_ecm, pose_psm=pose_psm)
    ecm_renderer.init_scene()

    # Start the video viewer loop in a separate thread
    init_video_viewer(cap, ecm_renderer)

    # Start the 3D render loop in the main thread
    ecm_renderer.run()


if __name__ == '__main__':
    main()
