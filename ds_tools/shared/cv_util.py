import cv2 as cv


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
