import cv2 as cv


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
