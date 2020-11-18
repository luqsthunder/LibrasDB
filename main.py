# %%
from libras_classifiers.librasdb_loaders import DBLoader2NPY
import pandas as pd
import numpy as np
import cv2 as cv
import os

left_front_data = pd.read_csv('v1098_barely_front_view_left_person.csv')
left_side_data = pd.read_csv('v0180_side_view_left_person.csv')

left_front_data = left_front_data.applymap(DBLoader2NPY.parse_npy_vec_str)
left_side_data = left_side_data.applymap(DBLoader2NPY.parse_npy_vec_str)
left_side_data.frame = left_front_data.frame


# %%
all_joints_to_use = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
                     'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 'RAnkle', 'LHip',
                     'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar'
                     ]

threshold = 0.60
all_frames = left_front_data.frame.unique().tolist()
# all_frames = [all_frames[0]] + [all_frames[len(all_frames) // 2]] + [all_frames[len(all_frames) - 2]]

matches = []

for frame in all_frames:
    row_front = left_front_data[left_front_data['frame'] == frame]
    row_front = row_front[all_joints_to_use].values[0]
    row_front = [x if x[2] >= threshold else None for x in row_front]

    row_side = left_side_data[left_side_data['frame'] == frame]
    row_side = row_side[all_joints_to_use].values[0]
    row_side = [x if x[2] >= threshold else None for x in row_side]

    for front, side in zip(row_front, row_side):
        if front is not None and side is not None:
            matches.append((front[:2], side[:2]))

    if len(matches) >= 16:
        break


if len(matches) >= 16:
    camera_mat = np.identity(3)

    essential, inliers = cv.findEssentialMat(np.array([x[0] for x in matches]), np.array([x[1] for x in matches]),
                                             method=cv.RANSAC, cameraMatrix=camera_mat)
    retval, R, t, mask2 = cv.recoverPose(essential, np.array([x[0] for x in matches]),
                                         np.array([x[1] for x in matches]),
                                         mask=inliers)#, cameraMatrix=camera_mat)

    diag = np.ones((3, 3))
    projection1 = np.zeros((3, 4))
    projection1[:3, :3] = diag

    projection2 = np.zeros((3, 4))
    projection2[:3, :3] = R
    projection2[:, 3:] = t

    inliers_pts1 = []
    inliers_pts2 = []
    for it in range(inliers.shape[0]):
        if inliers[it] != 0:
            inliers_pts1.append(matches[it][0].tolist())
            inliers_pts2.append(matches[it][1].tolist())

    points1u = cv.undistortPoints(np.array(inliers_pts1), cameraMatrix=camera_mat, distCoeffs=None, R=R)
    points2u = cv.undistortPoints(np.array(inliers_pts2), cameraMatrix=camera_mat, distCoeffs=None, R=R)

    for frame in all_frames:
        row_front = left_front_data[left_front_data['frame'] == frame]
        row_front = row_front[all_joints_to_use].values[0]
        row_front = [x if x[2] >= threshold else None for x in row_front]

        row_side = left_side_data[left_side_data['frame'] == frame]
        row_side = row_side[all_joints_to_use].values[0]
        row_side = [x if x[2] >= threshold else None for x in row_side]

        matches = []
        for front, side in zip(row_front, row_side):
            if front is not None and side is not None:
                matches.append((front[:2], side[:2]))

        points3d = cv.triangulatePoints(projection1, projection2,
                                        np.array([x[0].tolist() for x in matches]).reshape(len(matches), 1, 2),
                                        np.array([x[1].tolist() for x in matches]).reshape(len(matches), 1, 2))

        cv.v

# cv::findEssentialMat (InputArray points1, InputArray points2, double focal=1.0, Point2d pp=Point2d(0, 0),
#                       int method=RANSAC, double prob=0.999, double threshold=1.0, OutputArray mask=noArray())

