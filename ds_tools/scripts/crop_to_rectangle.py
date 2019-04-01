from os import path
import cv2 as cv
import os

# Our local modules
from ds_tools.shared import util


def main():
    """
    This script loads all images from the specified directory, crops them to the specified rectangle, and writes them
    to the output directory.
    :return:
    """

    input_dir = path.join(util.get_data_dir(), 'placenta_phantom_capture', 'uncropped')
    output_dir = path.join(util.get_data_dir(), 'placenta_phantom_capture', 'calib_0')

    input_width, input_height = 720, 576
    expected_shape = (input_height, input_width)

    x1, y1 = 11, 4
    x2, y2 = 714, 575

    all_image_files = os.listdir(input_dir)
    for image_file in all_image_files:
        image = cv.imread(path.join(input_dir, image_file))
        assert image.shape[:2] == expected_shape

        image = image[y1:y2 + 1, x1:x2 + 1]
        cv.imwrite(path.join(output_dir, image_file), image)


if __name__ == '__main__':
    main()
