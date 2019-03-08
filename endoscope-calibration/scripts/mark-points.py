from os import path
import cv2 as cv
import sys

script_dir = path.dirname(path.realpath(__file__))
sys.path.append(path.join(script_dir, '..', 'src'))

data_dir = path.join(script_dir, '..', 'data')

# Script for manually marking points and lines on images.
#
# Click in relevant places of the image, then press 'e' to start a new line.
# Once you're done, press `g` to print all lines to console in Python friendly
# format. This way, you can just copy & paste console output into Jupyter.


if __name__ == '__main__':

    import util

    img = None
    load_image = True

    if load_image:
        image_name = 'image-3.png'
        img = cv.imread(path.join(data_dir, image_name))
    else:
        video_name = 'handheld-endoscope-calibration-full.avi'
        frame_index = 2069
        cap = cv.VideoCapture(path.join(data_dir, video_name))
        img = util.pick_frames(cap, [frame_index])[0]

    lines = []
    points_x = []
    points_y = []


    def click(event, x, y, flags, param):
        if event != cv.EVENT_LBUTTONDOWN:
            return

        points_x.append(x)
        points_y.append(y)


    def save_line():
        global points_x, points_y, img

        line = [points_x, points_y]
        lines.append(line)
        for i in range(len(points_x) - 1):
            start = (points_x[i], points_y[i])
            end = (points_x[i + 1], points_y[i + 1])
            cv.line(img, start, end, (255, 0, 0), 2)

        points_x = []
        points_y = []


    cv.namedWindow("image")
    cv.setMouseCallback("image", click)

    while True:
        cv.imshow('image', img)
        key = cv.waitKey(50)
        if key == ord('e'):
            save_line()

        if key == ord('g'):
            if len(points_x) > 0:
                save_line()

            for line in lines:
                print('[{},'.format(line[0]))
                print(' {}],'.format(line[1]))
