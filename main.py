# %%
# https://colab.research.google.com/drive/1iIyU6DItu96V1omHqWMinvQlKL0UWR1w?usp=sharing
from libras_classifiers.librasdb_loaders import DBLoader2NPY
import pandas as pd
import numpy as np
import cv2 as cv
from tqdm import tqdm
import random

person = "left"
version = "2"

front_data_path = "v1098_barely_front_view_" + person + "_person.csv"
side_data_path = "v0180_side_view_" + person + "_person.csv"
destination_path = r"C:/csv Libras/" + person + "_person_3d_" + version + ".csv"

left_front_data = pd.read_csv(front_data_path)
df = pd.DataFrame(left_front_data)
left_side_data = pd.read_csv(side_data_path)

left_front_data = left_front_data.applymap(DBLoader2NPY.parse_npy_vec_str)
left_side_data = left_side_data.applymap(DBLoader2NPY.parse_npy_vec_str)
left_side_data.frame = left_front_data.frame


# %%
all_joints_to_save = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
]

# all_joints_to_use = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar', 'left_Wrist', 'left_ThumbMetacarpal', 'left_ThumbProximal', 'left_ThumbMiddle', 'left_ThumbDistal', 'left_IndexFingerMetacarpal', 'left_IndexFingerProximal', 'left_IndexFingerMiddle', 'left_IndexFingerDistal', 'left_MiddleFingerMetacarpal', 'left_MiddleFingerProximal', 'left_MiddleFingerMiddle', 'left_MiddleFingerDistal', 'left_RingFingerMetacarpal', 'left_RingFingerProximal', 'left_RingFingerMiddle', 'left_RingFingerDistal', 'left_LittleFingerMetacarpal', 'left_LittleFingerProximal', 'left_LittleFingerMiddle', 'left_LittleFingerDistal', 'right_Wrist', 'right_ThumbMetacarpal', 'right_ThumbProximal', 'right_ThumbMiddle', 'right_ThumbDistal', 'right_IndexFingerMetacarpal', 'right_IndexFingerProximal', 'right_IndexFingerMiddle', 'right_IndexFingerDistal', 'right_MiddleFingerMetacarpal', 'right_MiddleFingerProximal', 'right_MiddleFingerMiddle', 'right_MiddleFingerDistal', 'right_RingFingerMetacarpal', 'right_RingFingerProximal', 'right_RingFingerMiddle', 'right_RingFingerDistal', 'right_LittleFingerMetacarpal', 'right_LittleFingerProximal', 'right_LittleFingerMiddle', 'right_LittleFingerDistal']

all_joints_to_use = ["RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist"]

threshold = 0.60
all_frames = left_front_data.frame.unique().tolist()
# all_frames = [all_frames[0]] + [all_frames[len(all_frames) // 2]] + [all_frames[len(all_frames) - 2]]

matches = []

for i in range(0, len(all_frames), 5):
    i += random.randrange(-2, 2)
    row_front = left_front_data[left_front_data["frame"] == all_frames[i]]
    row_front = row_front[all_joints_to_use].values[0]
    row_front = [x if x[2] >= threshold else None for x in row_front]

    row_side = left_side_data[left_side_data["frame"] == all_frames[i]]
    row_side = row_side[all_joints_to_use].values[0]
    row_side = [x if x[2] >= threshold else None for x in row_side]

    for front, side in zip(row_front, row_side):
        if front is not None and side is not None:
            matches.append((front[:2], side[:2]))

    if len(matches) >= 100:
        break
print(len(matches))

if len(matches) >= 100:
    camera_mat = np.identity(3)
    fx, fy = 1, 1
    camera_mat[0][0] = fx
    camera_mat[1][1] = fy

    essential, inliers = cv.findEssentialMat(
        np.array([x[0] for x in matches]),
        np.array([x[1] for x in matches]),
        method=cv.RANSAC,
        cameraMatrix=camera_mat,
    )
    retval, R, t, mask2 = cv.recoverPose(
        essential,
        np.array([x[0] for x in matches]),
        np.array([x[1] for x in matches]),
        mask=inliers,
    )  # , cameraMatrix=camera_mat)

    projection1 = np.zeros((3, 4))
    projection1[:3, :3] = camera_mat

    projection2 = np.zeros((3, 4))
    projection2[:3, :3] = R
    projection2[:, 3:] = t

    dic = {k: [] for k in left_front_data.keys()}

    for frame in tqdm(all_frames):
        row_front = left_front_data[left_front_data["frame"] == frame]
        row_front = row_front.drop(columns=["frame", "Unnamed: 0"])
        row_front = row_front.applymap(
            lambda x: np.array([0.0, 0.0, 0.0]) if isinstance(x, float) else x
        )
        row_front = [
            (x.values[0] if x.values[0][2] >= threshold else None, key)
            for key, x in row_front.iteritems()
        ]

        row_side = left_side_data[left_side_data["frame"] == frame]
        row_side = row_side.drop(columns=["frame", "Unnamed: 0"])
        row_side = row_side.applymap(
            lambda x: np.array([0.0, 0.0, 0.0]) if isinstance(x, float) else x
        )
        row_side = [
            (x.values[0] if x.values[0][2] >= threshold else None, key)
            for key, x in row_side.iteritems()
        ]

        matches = []
        dic["frame"].append(frame)
        for front, side in zip(row_front, row_side):
            if front[0] is not None and side[0] is not None:
                # points1 = cv.undistortPoints(front[0][:2].reshape(1, 1, 2),camera_mat)
                points3d = cv.triangulatePoints(
                    projection1,
                    projection2,
                    front[0][:2].reshape(1, 1, 2),
                    side[0][:2].reshape(1, 1, 2),
                )
                dic[front[1]].append(points3d.T[0])
            else:
                dic[front[1]].append(None)


del dic["Unnamed: 0"]
df = pd.DataFrame(dic)
df.to_csv(destination_path)
# cv::findEssentialMat (InputArray points1, InputArray points2, double focal=1.0, Point2d pp=Point2d(0, 0),
#                       int method=RANSAC, double prob=0.999, double threshold=1.0, OutputArray mask=noArray())
