from os import path
import cv2 as cv

# Our local modules
from ds_tools.shared import util, cv_util


# Script for manually marking points and lines on images.
#
# Click in relevant places of the image, then press 'e' to start a new line.
# Once you're done, press `g` to print all lines to console in Python friendly
# format. This way, you can just copy & paste console output into Jupyter.
#
# If you only want mark one line:
#   1. Start the script.
#   2. Click on relevant points in the image.
#   3. Press `e` to create a line.
#   4. Press `g` to print out the points in console.

def main():
    load_image = True
    data_dir = util.get_data_dir()

    if load_image:
        image_path = path.join(data_dir, 'placenta_phantom_calib', 'scan_3.png')
        img = cv.imread(image_path)
    else:
        video_name = 'handheld-endoscope-calibration-full.avi'
        frame_index = 2069
        cap = cv.VideoCapture(path.join(data_dir, video_name))
        img = cv_util.pick_frames(cap, [frame_index])[0]

    lines = []
    points_x = []
    points_y = []

    def click(event, x, y, flags, param):
        if event != cv.EVENT_LBUTTONDOWN:
            return

        points_x.append(x)
        points_y.append(y)

    def save_line(points_x, points_y, img):
        line = [points_x, points_y]
        lines.append(line)
        for i in range(len(points_x) - 1):
            start = (points_x[i], points_y[i])
            end = (points_x[i + 1], points_y[i + 1])
            cv.line(img, start, end, (255, 0, 0), 2)

    cv.namedWindow("image")
    cv.setMouseCallback("image", click)

    while True:
        cv.imshow('image', img)
        key = cv.waitKey(50)
        if key == ord('e'):
            save_line(points_x, points_y, img)
            points_x = []
            points_y = []

        if key == ord('g'):
            if len(points_x) > 0:
                save_line(points_x, points_y, img)
                points_x = []
                points_y = []

            for line in lines:
                print('[{},'.format(line[0]))
                print(' {}],'.format(line[1]))


if __name__ == '__main__':
    main()
