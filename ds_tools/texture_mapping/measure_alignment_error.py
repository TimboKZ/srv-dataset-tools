from itertools import combinations
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
    variance_error = np.round(np.ma.var(np_masked), 2)
    print('Error mean/variance:  {}  /  {}  /  {}'.format(mean_error, median_error, variance_error))

    avg_diff_img = cv_util.expand_2d_to_3d(avg_diff_masked.astype(int), 3)
    cv.imwrite(output_path, avg_diff_img)


def compute_alignment_error_between(proj_A, mask_A, proj_B, mask_B):
    e = 0
    n = 0
    h, w = proj_A.shape
    mask = np.logical_and(mask_A, mask_B)
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue

            diff = np.abs(float(proj_A[y, x]) - float(proj_B[y, x]))
            e += diff
            n += 1

    return e, n


def measure_alignment_error(projection_path_template, confidence_path_template, output_path_template, total_images):
    all_projections = []
    all_masks = []
    for i in range(total_images):
        projection = cv.imread(projection_path_template.format(i), cv.IMREAD_GRAYSCALE)
        projection = cv.Laplacian(projection, cv.CV_8UC1)
        projection = cv.blur(projection, (3, 3))
        cv.imwrite(output_path_template.format(i), projection)
        all_projections.append(projection)

        mask = (cv.imread(confidence_path_template.format(i), cv.IMREAD_GRAYSCALE) > 10)
        all_masks.append(mask)

    total_e = 0.0
    total_n = 0.0

    for i, j in combinations(range(total_images), 2):
        proj_A = all_projections[i]
        mask_A = all_masks[i]
        proj_B = all_projections[j]
        mask_B = all_masks[j]
        e, n = compute_alignment_error_between(proj_A, mask_A, proj_B, mask_B)
        total_e += e
        total_n += n

    alignment_error = np.round(total_e / total_n, 2)
    print('Alignment error:', alignment_error)


def main():
    capture_path = path.join(util.get_resource_dir(), 'heart_texture_capture')
    projection_path_template = path.join(capture_path, '{}_projection.png')
    confidence_path_template = path.join(capture_path, '{}_confidence.png')
    output_path_template = path.join(capture_path, '{}_edges.png')
    total_images = 10

    measure_alignment_error(projection_path_template, confidence_path_template, output_path_template, total_images)


if __name__ == '__main__':
    main()
