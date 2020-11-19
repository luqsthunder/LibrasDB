# %%
from libras_classifiers.librasdb_loaders import DBLoader2NPY
import pandas as pd
import numpy as np
import cv2 as cv
from tqdm import tqdm

left_front_data = pd.read_csv('v1098_barely_front_view_right_person.csv')
left_side_data = pd.read_csv('v0180_side_view_right_person.csv')

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

    dic = {
        k: [] for k in left_front_data.keys()
    }

    for frame in tqdm(all_frames):
        row_front = left_front_data[left_front_data['frame'] == frame]
        row_front = row_front.drop(columns=['frame', 'Unnamed: 0'])
        row_front = row_front.applymap(lambda x: np.array([0.0, 0.0, 0.0]) if isinstance(x, float) else x)
        row_front = [(x.values[0] if x.values[0][2] >= threshold else None, key)
                     for key, x in row_front.iteritems()]

        row_side = left_side_data[left_side_data['frame'] == frame]
        row_side = row_side.drop(columns=['frame', 'Unnamed: 0'])
        row_side = row_side.applymap(lambda x: np.array([0.0, 0.0, 0.0]) if isinstance(x, float) else x)
        row_side = [(x.values[0] if x.values[0][2] >= threshold else None, key)
                    for key, x in row_side.iteritems()]

        matches = []
        dic['frame'].append(frame)
        for front, side in zip(row_front, row_side):
            if front[0] is not None and side[0] is not None:
                #matches.append((front[0][:2], side[0][:2]))
                points3d = cv.triangulatePoints(projection1, projection2,
                                                front[0][:2].reshape(1, 1, 2),
                                                side[0][:2].reshape(1, 1, 2))
                dic[front[1]].append(points3d)
            else:
                dic[front[1]].append(None)

# cv::findEssentialMat (InputArray points1, InputArray points2, double focal=1.0, Point2d pp=Point2d(0, 0),
#                       int method=RANSAC, double prob=0.999, double threshold=1.0, OutputArray mask=noArray())
