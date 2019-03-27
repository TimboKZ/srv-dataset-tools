import numpy.linalg as la
from os import path
import numpy as np
import cv2 as cv

# Our local modules
from ds_tools.shared import transform as tf
from ds_tools.shared import util, cv_util


def merge_textures(capture_data_json_path, texture_dir, texture_count):
    # Prepare paths
    base_capture_path = path.join(texture_dir, 'base_{}.png')
    texture_capture_path = path.join(texture_dir, '{}_{}.png')

    # Load capture data JSON
    capture_json = util.load_dict(capture_data_json_path)
    all_camera_pos = capture_json['camera_pos']
    all_camera_hpr = capture_json['camera_hpr']
    all_camera_normal = capture_json['camera_normal']

    # Load all textures
    base_texture = cv.imread(base_capture_path.format('texture'))
    base_normal = cv.imread(base_capture_path.format('normal'))
    base_mask_img = cv.imread(base_capture_path.format('mask'), cv.IMREAD_GRAYSCALE)
    base_mask_inv_img = 255 - base_mask_img
    projections = []
    for i in range(texture_count):
        projection = cv.imread(texture_capture_path.format(i, 'projection'))
        projections.append(projection)

    # Extract basic information about the textures
    height, width = base_texture.shape[:2]
    base_mask = base_mask_img > 125

    # Transform normal map from [0, 255] to [-1.0, 1.0]
    base_normal = cv.cvtColor(base_normal, cv.COLOR_BGR2RGB)
    base_normal = (base_normal.astype(np.float32) / 255.0) * 2.0 - 1
    base_normal_sizes = cv_util.expand_2d_to_3d(la.norm(base_normal, axis=2), 3)
    base_normal /= base_normal_sizes

    # Compute confidence scores for all pixels
    confidences = np.zeros((texture_count, height, width))
    for i in range(texture_count):
        camera_normal = np.array(all_camera_normal[i])

        # Compute incidence angle for each pixel in the image
        incidence_angles = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                if not base_mask[y, x]:
                    continue

                pixel_normal = base_normal[y, x, :]
                incidence_angles[y, x] = 180 - tf.angle_between(pixel_normal, camera_normal, normalize=False)

        # Compute the confidence scores
        confidence = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                if not base_mask[y, x]:
                    continue

                angle = incidence_angles[y, x]
                confidence[y, x] = 1.0 - angle / 90 if angle < 90 else 0

        confidences[i] = confidence
        confidence_img = cv.cvtColor(confidence * 255.0, cv.COLOR_GRAY2BGR)
        cv.imwrite(texture_capture_path.format(i, 'confidence'), confidence_img)

    # Normalize confidences
    confidences /= np.sum(confidences, axis=0) + np.finfo(np.float32).eps

    # Combine all textures using confidence scores image
    combined_image = np.zeros((height, width, 3))
    for i in range(texture_count):
        combined_image += projections[i] * cv_util.expand_2d_to_3d(confidences[i], 3)

    # Write final image
    final_image = combined_image + cv_util.expand_2d_to_3d(base_mask_inv_img, 3)
    cv.imwrite(base_capture_path.format('final'), final_image)


def main():
    resource_dir = util.get_resource_dir()
    json_path = path.join(resource_dir, 'heart_screenshots', 'capture_data.json')
    capture_path = path.join(resource_dir, 'heart_texture_capture')

    merge_textures(json_path, capture_path, 4)


if __name__ == '__main__':
    main()
