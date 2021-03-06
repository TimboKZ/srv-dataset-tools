import numpy.linalg as la
from os import path
import numpy as np
import cv2 as cv

# Our local modules
from ds_tools.shared import transform as tf
from ds_tools.shared import util, cv_util


def merge_textures(capture_data_json_path, texture_dir, max_texture_count=99999):
    # Prepare paths
    base_capture_path = path.join(texture_dir, 'base_{}.png')
    texture_capture_path = path.join(texture_dir, '{}_{}.png')

    #############################################
    #
    # Load relevant capture data
    #
    #############################################

    # Load capture data JSON
    capture_json = util.load_dict(capture_data_json_path)
    all_camera_pos = capture_json['camera_pos']
    texture_count = min(max_texture_count, len(all_camera_pos))

    # Load all textures
    projections = []
    projections_yuv = []
    visibility_maps = []
    frustums = []
    light_maps = []
    for i in range(texture_count):
        projection = cv.imread(texture_capture_path.format(i, 'projection'))
        projections.append(projection)

        projection_yuv = cv.cvtColor(projection[:, :, :3], cv.COLOR_BGR2YUV)
        projections_yuv.append(projection_yuv)

        frustum = cv.imread(texture_capture_path.format(i, 'frustum'))
        frustum = cv.cvtColor(frustum[:, :, :3], cv.COLOR_BGR2RGB)
        frustum = (frustum.astype(float) / 255.0) * 2.0 - 1.0
        frustums.append(frustum)

        light_map = cv.imread(texture_capture_path.format(i, 'light'))
        light_map = cv.cvtColor(light_map[:, :, :3], cv.COLOR_BGR2RGB)
        light_map = (light_map.astype(float) / 255.0) * 2.0 - 1.0
        light_map_sizes = cv_util.expand_2d_to_3d(la.norm(light_map, axis=2), 3)
        light_map /= light_map_sizes
        light_maps.append(light_map)

        visibility_map = cv.imread(texture_capture_path.format(i, 'visibility'), cv.IMREAD_GRAYSCALE)
        visibility_maps.append(visibility_map > 125)

    base_normal = cv.imread(base_capture_path.format('normal'))
    base_texture_mask_img = cv.imread(base_capture_path.format('mask'), cv.IMREAD_GRAYSCALE)
    base_texture_mask_inv_img = 255 - base_texture_mask_img

    # Extract basic information about the textures
    height, width = base_normal.shape[:2]
    texture_mask = base_texture_mask_img > 125

    # Combine visibility maps for individual projections into a single global visibility map
    base_visibility_map = np.zeros((height, width), dtype=bool)
    for visibility_map in visibility_maps:
        base_visibility_map = np.logical_or(base_visibility_map, visibility_map)

    # Transform normal map from [0, 255] to [-1.0, 1.0]
    base_normal = cv.cvtColor(base_normal, cv.COLOR_BGR2RGB)
    base_normal = (base_normal.astype(np.float32) / 255.0) * 2.0 - 1
    base_normal_sizes = cv_util.expand_2d_to_3d(la.norm(base_normal, axis=2), 3)
    base_normal /= base_normal_sizes

    #############################################
    #
    # Prepare confidences for all projections
    #
    #############################################

    # Compute confidence scores for all pixels
    confidences = np.zeros((texture_count, height, width))
    for i in range(texture_count):
        print('Processing {}...'.format(i))
        local_mask = np.logical_and(texture_mask, visibility_maps[i])
        projection_yuv = projections_yuv[i]
        frustum = frustums[i]
        light_map = light_maps[i]

        # Compute incidence angle for each pixel in the image
        incidence_angles = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                if not local_mask[y, x]:
                    continue

                pixel_normal = base_normal[y, x, :]
                vertex_to_light = light_map[y, x, :]
                incidence_angles[y, x] = tf.angle_between(pixel_normal, vertex_to_light, normalize=False)

        # Compute the confidence scores
        confidence_mat = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                if not local_mask[y, x]:
                    continue

                confidence = 0.0

                # Determine base confidence based on the angle between camera normal and surface normal
                angle = incidence_angles[y, x]
                if angle < 90:
                    perfect_angle = 30
                    absolute_angle_diff = np.abs(angle - perfect_angle)
                    confidence = 1.0 - absolute_angle_diff / (90.0 - perfect_angle)

                # Try to remove specular highlights based on intensity
                luma_Y = projection_yuv[y, x, 0]
                intensity_threshold = 50.0
                intensity_range = 255.0 - intensity_threshold
                if luma_Y > intensity_threshold:
                    intensity_diff = luma_Y - intensity_threshold
                    intensity_score = 1.0 - intensity_diff / intensity_range
                    confidence *= intensity_score ** 3

                # Decay confidence next to borders
                border_buffer = 0.7
                for frus_dim in frustum[y, x, :2]:
                    frus_abs = np.abs(frus_dim) - border_buffer
                    if frus_abs > 0:
                        border_score = 1.0 - frus_abs / (1.0 - border_buffer)
                        confidence *= border_score ** 2

                confidence_mat[y, x] = confidence

        confidences[i] = confidence_mat
        confidence_img = cv.cvtColor(confidence_mat * 255.0, cv.COLOR_GRAY2BGR)
        cv.imwrite(texture_capture_path.format(i, 'confidence'), confidence_img)

    # Normalize confidences
    confidences_sum = np.sum(confidences, axis=0)
    confidences /= confidences_sum + np.finfo(np.float32).eps

    #############################################
    #
    # Generate and save complete texture
    #
    #############################################

    # Combine all textures using confidence scores image
    combined_image = np.zeros((height, width, 3))
    for i in range(texture_count):
        combined_image += projections[i] * cv_util.expand_2d_to_3d(confidences[i], 3)

    # Inpaint pixels around the texture to avoid white edges during Blender texture preview
    eroded_mask = base_texture_mask_img.copy()
    eroded_mask = cv.dilate(eroded_mask, np.ones((5, 5), np.uint8))
    inpaint_mask = eroded_mask * base_texture_mask_inv_img

    # Write final image
    final_image = combined_image + cv_util.expand_2d_to_3d(base_texture_mask_inv_img, 3)
    final_image = cv.inpaint(final_image.astype(np.uint8), inpaint_mask, 3, cv.INPAINT_TELEA)

    cv.imwrite(base_capture_path.format('final'), final_image)

    # Write combined visibility
    base_visibility_map_img = cv_util.expand_2d_to_3d(base_visibility_map.astype(int) * 255, 3)
    cv.imwrite(base_capture_path.format('visibility'), base_visibility_map_img)

    # Write mask with white for all pixels with non-zero confidence
    confidence_sum_img = cv_util.expand_2d_to_3d((confidences_sum > 0).astype(int) * 255, 3)
    cv.imwrite(base_capture_path.format('confidence'), confidence_sum_img)


def main():
    resource_dir = util.get_resource_dir()
    json_path = path.join(resource_dir, 'placenta_images', 'capture_data.json')
    capture_path = path.join(resource_dir, 'placenta_texture')

    merge_textures(json_path, capture_path)


if __name__ == '__main__':
    main()
