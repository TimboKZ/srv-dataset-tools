from _thread import start_new_thread
from os import path
import numpy as np
import cv2 as cv

# Our local modules
from ds_tools.kinematics.kinematics_render_app import KinematicsRenderApp, LoadFrameEventName, ShutdownEventName
from ds_tools.shared import util, cv_util


def load_numpy_csv(csv_path):
    return np.loadtxt(csv_path, delimiter=',', dtype=np.float32)


def init_video_viewer(left_video_capture, ecm_renderer, right_video_capture=None, snapshot_dir=None):
    LeftWindowName = 'Kinematics Viz Left'
    RightWindowName = 'Kinematics Viz Right'
    TrackbarName = 'Frame'

    def video_viewer_routine():
        frame_count = int(left_video_capture.get(cv.CAP_PROP_FRAME_COUNT))
        last_loop_frame = 0

        last_left_image = None
        last_right_image = None

        def on_trackbar_change(frame):
            nonlocal last_loop_frame
            nonlocal last_left_image
            nonlocal last_right_image

            ecm_renderer.messenger.send(LoadFrameEventName, [frame])

            if last_loop_frame != frame:
                last_loop_frame = frame

                left_video_capture.set(cv.CAP_PROP_POS_FRAMES, frame)
                err, img_left = left_video_capture.read()
                cv.imshow(LeftWindowName, img_left)
                last_left_image = img_left

                if right_video_capture is not None:
                    right_video_capture.set(cv.CAP_PROP_POS_FRAMES, frame)
                    err, img_right = right_video_capture.read()
                    cv.imshow(RightWindowName, img_right)
                    last_right_image = img_right

        cv.namedWindow(LeftWindowName)
        cv.moveWindow(LeftWindowName, 100, 100)
        cv.createTrackbar(TrackbarName, LeftWindowName, 0, frame_count - 1, on_trackbar_change)

        if right_video_capture is not None:
            cv.namedWindow(RightWindowName)
            cv.moveWindow(RightWindowName, 100 + 500, 100)

        on_trackbar_change(1)

        paused = True

        while left_video_capture.isOpened():
            if paused:
                key = cv.waitKey(10) & 0xff
                if key == 32:
                    paused = False
                elif key == 27:
                    if ecm_renderer:
                        ecm_renderer.messenger.send(ShutdownEventName)
                    break
                elif key == 98:
                    if snapshot_dir is not None:
                        filename = path.join(snapshot_dir, 'frame_{}_{{}}.png'.format(last_loop_frame))
                        if last_left_image is not None:
                            cv.imwrite(filename.format('left'), last_left_image)
                        if last_right_image is not None:
                            cv.imwrite(filename.format('right'), last_right_image)
                else:
                    continue

            err, img_left = left_video_capture.read()
            cv.imshow(LeftWindowName, img_left)
            last_left_image = img_left

            if right_video_capture is not None:
                err, img_right = right_video_capture.read()
                cv.imshow(RightWindowName, img_right)
                last_right_image = img_right

            curr_frame = int(left_video_capture.get(cv.CAP_PROP_POS_FRAMES))
            last_loop_frame = curr_frame
            cv.setTrackbarPos(TrackbarName, LeftWindowName, curr_frame)

            key = cv.waitKey(10) & 0xff
            if key == 27:
                if ecm_renderer:
                    ecm_renderer.messenger.send(ShutdownEventName)
                break
            elif key == 32:
                paused = True

        cv.destroyAllWindows()
        left_video_capture.release()
        right_video_capture.release()

    start_new_thread(video_viewer_routine, ())


def main():
    # Tweak these paths to match your data
    data_dir = util.get_data_dir()
    capture_dir = path.join(data_dir, 'placenta_phantom_capture', 'synced')
    snapshot_dir = path.join(data_dir, 'placenta_phantom_snapshots')
    left_video_path = path.join(capture_dir, 'EndoscopeImageMemory_0.avi')
    right_video_path = path.join(capture_dir, 'EndoscopeImageMemory_1.avi')
    pose_ecm = load_numpy_csv(path.join(capture_dir, 'pose_ecm.csv'))
    pose_psm = load_numpy_csv(path.join(capture_dir, 'pose_psm.csv'))

    # Prepare video capture
    left_cap = cv.VideoCapture(left_video_path)
    right_cap = cv.VideoCapture(right_video_path)
    width, height = cv_util.get_capture_size(left_cap)

    # Setup the 3D renderer
    ecm_renderer = KinematicsRenderApp(width=width, height=height,
                                       pose_ecm=pose_ecm, pose_psm=pose_psm)
    ecm_renderer.init_scene()

    # Start the video viewer loop in a separate thread
    init_video_viewer(left_cap, ecm_renderer, right_video_capture=right_cap, snapshot_dir=snapshot_dir)

    # Start the 3D render loop in the main thread
    ecm_renderer.run()


if __name__ == '__main__':
    main()
