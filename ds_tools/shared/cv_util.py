from os import path
import numpy as np
import cv2 as cv
import os


def expand_2d_to_3d(mat_2d, last_dim_size):
    assert len(mat_2d.shape) == 2
    return np.repeat(mat_2d[:, :, np.newaxis], last_dim_size, axis=2)


def get_capture_size(cap):
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    return width, height


def wait_for_esc(interval=0):
    """
    Forces OpenCV to wait for the Escape key
    :param interval:
    :return:
    """
    while True:
        key = cv.waitKey(interval) & 0xff
        if key == 27:
            break


def apply_brightness_contrast(in_frame, brightness=0, contrast=1.0):
    out_frame = (in_frame + brightness) * contrast
    out_frame = np.clip(out_frame, 0, 255).astype(np.uint8)
    return out_frame


def pick_frames(video_cap, frame_indices, convert_to_rgb=False):
    frame_count = len(frame_indices)
    frames = [None] * frame_count

    for i in range(frame_count):
        video_cap.set(cv.CAP_PROP_POS_FRAMES, frame_indices[i])
        ret, frame = video_cap.read()
        # TODO: Check ret
        if convert_to_rgb:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frames[i] = frame

    return frames


def pick_equidistant_frames(video_cap, frame_count, convert_to_rgb=False):
    total_frames = int(video_cap.get(cv.CAP_PROP_FRAME_COUNT))
    step = total_frames // frame_count
    return pick_frames(video_cap, [i * step for i in range(frame_count)], convert_to_rgb=convert_to_rgb)


def load_all_images(folder_path, convert_to_rgb=False):
    all_images = sorted(os.listdir(folder_path))
    image_count = len(all_images)

    images = []

    for i in range(image_count):
        file = all_images[i]
        image = cv.imread(path.join(folder_path, file))
        if convert_to_rgb:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        images.append(image)

    return images
