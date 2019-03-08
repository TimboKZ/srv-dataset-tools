from _thread import start_new_thread
from os import path
import numpy as np
import cv2 as cv

# Our local modules
from ds_tools.kinematics.kinematics_render_app import KinematicsRenderApp, LoadFrameEventName, ShutdownEventName
from ds_tools.shared import util


def load_numpy_csv(csv_path):
    return np.loadtxt(csv_path, delimiter=',', dtype=np.float32)


def init_video_viewer(video_path, ecm_renderer):
    WindowName = 'Kinematics Viz'
    TrackbarName = 'Frame'

    def video_viewer_routine():
        cap = cv.VideoCapture(video_path)
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        last_loop_frame = 0

        def on_trackbar_change(frame):
            ecm_renderer.messenger.send(LoadFrameEventName, [frame])

            if last_loop_frame != frame:
                cap.set(cv.CAP_PROP_POS_FRAMES, frame)
                err, img = cap.read()
                cv.imshow(WindowName, img)

        cv.namedWindow(WindowName)
        cv.moveWindow(WindowName, 100, 100)
        cv.createTrackbar(TrackbarName, WindowName, 0, frame_count - 1, on_trackbar_change)

        on_trackbar_change(1)

        paused = True

        while cap.isOpened():
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

            err, img = cap.read()
            cv.imshow(WindowName, img)

            curr_frame = int(cap.get(cv.CAP_PROP_POS_FRAMES))
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
        cap.release()

    start_new_thread(video_viewer_routine, ())


def main():
    data_dir = util.get_data_dir()
    video_path = path.join(data_dir, 'EndoscopeImageMemory_0_small.avi')
    pose_ecm = load_numpy_csv(path.join(data_dir, 'pose_ecm.csv'))
    pose_psm = load_numpy_csv(path.join(data_dir, 'pose_psm.csv'))

    # Setup the 3D renderer
    ecm_renderer = KinematicsRenderApp(pose_ecm, pose_psm)
    ecm_renderer.init_scene()

    # Start the video viewer loop in a separate thread
    init_video_viewer(video_path, ecm_renderer)

    # Start the 3D render loop in the main thread
    ecm_renderer.run()


if __name__ == '__main__':
    main()
