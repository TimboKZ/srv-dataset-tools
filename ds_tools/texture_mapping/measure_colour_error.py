from os import path
import numpy as np
import cv2 as cv

# Our local modules
from ds_tools.shared import util, cv_util


def measure_error(true_texture_path, reconst_texture_path, mask_path, output_path):
    true_tex = cv.imread(true_texture_path)
    rec_tex = cv.imread(reconst_texture_path)
    mask_bool = (cv.imread(mask_path, cv.IMREAD_GRAYSCALE) > 125)
    mask = mask_bool.astype(int)
    mask_rgb = cv_util.expand_2d_to_3d(mask, 3)

    true_tex_processed = true_tex[:, :, :3].astype(float) * mask_rgb
    true_tex_processed -= np.mean(true_tex_processed, axis=(0, 1))

    rec_tex_processed = rec_tex[:, :, :3].astype(float) * mask_rgb
    rec_tex_processed -= np.mean(rec_tex_processed, axis=(0, 1))

    diff = np.abs(true_tex_processed - rec_tex_processed)
    avg_diff = np.sum(diff, axis=2) / 3.0

    avg_diff_masked = avg_diff * mask

    np_masked = np.ma.array(avg_diff_masked, mask=np.invert(mask_bool))
    mean_error = np.round(np.ma.mean(np_masked), 2)
    median_error = np.round(np.ma.median(np_masked), 2)
    std_error = np.round(np.sqrt(np.ma.var(np_masked)), 2)
    print('Error mean/median/s.d.:  {}  /  {}  /  {}'.format(mean_error, median_error, std_error))

    avg_diff_img = cv_util.expand_2d_to_3d(avg_diff_masked.astype(int), 3)
    cv.imwrite(output_path, avg_diff_img)


def main():
    capture_path = path.join(util.get_resource_dir(), 'heart_texture_capture')
    true_texture_path = path.join(capture_path, 'base_texture.png')
    reconst_texture_path = path.join(capture_path, 'base_no_luma.png')
    mask_path = path.join(capture_path, 'base_confidence.png')

    output_path = path.join(capture_path, 'base_difference.png')

    measure_error(true_texture_path, reconst_texture_path, mask_path, output_path)


if __name__ == '__main__':
    main()
