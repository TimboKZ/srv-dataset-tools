from matplotlib import pyplot as plt
from os import path
import numpy as np
import cv2 as cv

# Our local modules
from ds_tools.shared import util


def find_sift_correspondence(src_img, dest_img, visualize=False):
    src_img = cv.equalizeHist(src_img)
    dest_img = cv.equalizeHist(dest_img)

    sift = cv.xfeatures2d.SIFT_create()
    src_keypoints, src_descs = sift.detectAndCompute(src_img, None)
    dest_keypoints, dest_descs = sift.detectAndCompute(dest_img, None)

    FLANN_INDEX_KDTREE = 0
    flann = cv.FlannBasedMatcher({'algorithm': FLANN_INDEX_KDTREE, 'trees': 5}, {'checks': 50})
    all_matches = flann.knnMatch(src_descs, dest_descs, k=2)

    matches = []
    for a, b in all_matches:
        if a.distance < 0.7 * b.distance:
            matches.append(a)

    print('Found {} matches!'.format(len(matches)))

    if visualize:
        src_points = np.float32([src_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_points = np.float32([dest_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)
        img3 = cv.drawMatches(src_img, src_keypoints, dest_img, dest_keypoints, matches, None, **draw_params)
        plt.imshow(img3, 'gray')
        plt.show(figsize=(10, 6), dpi=80)

    src_points = []
    dest_points = []
    for match in matches:
        src_keypoint = src_keypoints[match.queryIdx]
        dest_keypoint = src_keypoints[match.trainIdx]
        src_points.append(src_keypoint.pt)
        dest_points.append(dest_keypoint.pt)

    return np.array(src_points), np.array(dest_points)


def main():
    resource_dir = util.get_resource_dir()
    capture_dir = path.join(resource_dir, 'placenta_images')
    capture_data_json = path.join(capture_dir, 'capture_data.json')
    image_path_template = path.join(capture_dir, '{}_screenshot.png')

    # Each pair will be checked for matching features. The camera pose for the first picture will remain unchanged,
    # while the camera pose for second image will be tweaked to reduce reprojection error.
    pairs = [
        (0, 1)
    ]

    # Load all images and data
    capture_data = util.load_dict(capture_data_json)
    all_camera_pos = np.array(capture_data['camera_pos'])
    all_camera_hpr = np.array(capture_data['camera_hpr'])
    cam_matrix = np.array(capture_data['camera_matrix'])
    images = []
    for i in range(len(all_camera_pos)):
        gray_img = cv.imread(image_path_template.format(i), cv.IMREAD_GRAYSCALE)
        images.append(gray_img)

    # Process all pairs and improve alignment
    for src_ind, dest_ind in pairs:
        src_img = images[src_ind]
        dest_img = images[dest_ind]
        src_points, dest_points = find_sift_correspondence(src_img, dest_img, visualize=True)
        E, mask = cv.findEssentialMat(src_points, dest_points, cam_matrix)
        R1, R2, t = cv.decomposeEssentialMat(E)
        print(R1)


if __name__ == '__main__':
    main()
