import os
import numpy as np
import sys
import cv2 as cv
import time
import matplotlib.pyplot as plt
import pandas as pd
% matplotlib
inline
sys.path.append('openpose/build/python')
import copy
from openpose import pyopenpose as op
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
from matplotlib.backends.backend_pdf import PdfPages


HAND_PARTS = { "Wrist":                 0, "ThumbMetacarpal":         1,
               "ThumbProximal":         2, "ThumbMiddle":             3,
               "ThumbDistal":           4, "IndexFingerMetacarpal":   5,
               "IndexFingerProximal":   6, "IndexFingerMiddle":       7,
               "IndexFingerDistal":     8, "MiddleFingerMetacarpal":  9,
               "MiddleFingerProximal": 10, "MiddleFingerMiddle":     11,
               "MiddleFingerDistal":   12, "RingFingerMetacarpal":   13,
               "RingFingerProximal":   14, "RingFingerMiddle":       15,
               "RingFingerDistal":     16, "LittleFingerMetacarpal": 17,
               "LittleFingerProximal": 18, "LittleFingerMiddle":     19,
               "LittleFingerDistal":   20
             }

HAND_PAIRS = [ ["Wrist",                  "ThumbMetacarpal"],
               ["ThumbMetacarpal",        "ThumbProximal"],
               ["ThumbProximal",          "ThumbMiddle"],
               ["ThumbMiddle",            "ThumbDistal"],
               ["Wrist",                  "IndexFingerMetacarpal"],
               ["IndexFingerMetacarpal",  "IndexFingerProximal"],
               ["IndexFingerProximal",    "IndexFingerMiddle"],
               ["IndexFingerMiddle",      "IndexFingerDistal"],
               ["Wrist",                  "MiddleFingerMetacarpal"],
               ["MiddleFingerMetacarpal", "MiddleFingerProximal"],
               ["MiddleFingerProximal",   "MiddleFingerMiddle"],
               ["MiddleFingerMiddle",     "MiddleFingerDistal"],
               ["Wrist",                  "RingFingerMetacarpal"],
               ["RingFingerMetacarpal",   "RingFingerProximal"],
               ["RingFingerProximal",     "RingFingerMiddle"],
               ["RingFingerMiddle",       "RingFingerDistal"],
               ["Wrist",                  "LittleFingerMetacarpal"],
               ["LittleFingerMetacarpal", "LittleFingerProximal"],
               ["LittleFingerProximal",   "LittleFingerMiddle"],
               ["LittleFingerMiddle",     "LittleFingerDistal"]
             ]

BODY_PARTS = { "Nose":    0, "Neck":       1, "RShoulder":    2, "RElbow":  3,
               "RWrist":  4, "LShoulder":  5, "LElbow":       6, "LWrist":  7,
               "RHip":    8, "RKnee":      9, "RAnkle":      10, "LHip":   11,
               "LKnee":  12, "LAnkle":    13, "REye":        14, "LEye":   15,
               "REar":   16, "LEar":      17, "Background":  18
             }
INV_BODY_PARTS = {v: k for k, v in BODY_PARTS.items()}

POSE_PAIRS = [ ["Neck",      "RShoulder"],
               ["Neck",      "LShoulder"],
               ["RShoulder", "RElbow"],
               ["RElbow",    "RWrist"],
               ["LShoulder", "LElbow"],
               ["LElbow",    "LWrist"],
               ["Neck",      "RHip"],
               ["RHip",      "RKnee"],
               ["RKnee",     "RAnkle"],
               ["Neck",      "LHip"],
               ["LHip",      "LKnee"],
               ["LKnee",     "LAnkle"],
               ["Neck",      "Nose"],
               ["Nose",      "REye"],
               ["REye",      "REar"],
               ["Nose",      "LEye"],
               ["LEye",      "LEar"]
             ]
print(INV_BODY_PARTS )

def convert_xypose_to_dir_angle(first_pose, second_pose, third_pose):
    """
    Parameters
    ----------

    Returns
    -------
    """

    first_vec = np.array(second_pose) - np.array(first_pose)
    second_vec = np.array(third_pose) - np.array(second_pose)

    vecs_norm = (np.linalg.norm(np.array(first_vec)) *
                 np.linalg.norm(np.array(second_vec)))

    angle = np.arccos(np.dot(first_vec, second_vec) / vecs_norm)

    angle_sign = np.sign(first_vec[0] * second_vec[1] -
                         first_vec[1] * second_vec[0])
    return angle_sign * angle


def make_xy_centroid(body_pose, pose_acc_min_threshold=0.2):
    """
    Parameters
    ----------

    Returns
    -------
    """

    x_points = [x[0] for x in body_pose
                if x[2] > pose_acc_min_threshold and x[0] > 1 and x[1] > 1]
    y_points = [x[1] for x in body_pose
                if x[2] > pose_acc_min_threshold and x[0] > 1 and x[1] > 1]

    return np.array([sum(x_points) / len(x_points),
                     sum(y_points) / len(y_points)])


def track_by_centroid(curr_persons_list, person_centroid,
                      ret_new_centroid=False):
    """
    as listas devem conter um dicionario nesse formato: {
        id: int
        centroid: np.array 2D
    }
    Parameters
    ----------

    Returns
    -------
    """
    correct_id = -1
    curr_person_dist = 9999999
    new_centroid = None

    for c_person in curr_persons_list:
        centroids_dist = \
            np.linalg.norm(person_centroid - c_person['centroid'])
        if centroids_dist < curr_person_dist:
            curr_person_dist = centroids_dist
            correct_id = c_person['id']
            new_centroid = c_person['centroid']

    return correct_id if not ret_new_centroid else correct_id, new_centroid


def make_hands_list(dt):
    """
    Parameters
    ----------

    Returns
    -------
    """

    hands_list = []
    # isso tem cara de ser 2 mão x N pessoas
    for it, persons in enumerate(dt.handKeypoints[0]):
        hand = {
            'id': it,
            'centroid': make_xy_centroid(dt.handKeypoints[0][it])}
        hands_list.append(hand)
    return hands_list


def make_persons_list(dt):
    """
    Parameters
    ----------

    Returns
    -------
    """

    persons_list = []
    for it, persons in enumerate(dt.poseKeypoints):
        person = {
            'id': it,
            'centroid': make_xy_centroid(dt.poseKeypoints[it])}
        persons_list.append(person)
    return persons_list


def calculate_angle_by_joints(dt, person, first_joint_id, second_joint_id,
                              third_joint_id):
    """
    Parameters
    ----------

    Returns
    -------
    """

    first = dt.poseKeypoints[person, first_joint_id][:2]
    second = dt.poseKeypoints[person, second_joint_id][:2]
    third = dt.poseKeypoints[person, third_joint_id][:2]

    angle = convert_xypose_to_dir_angle(first, second, third)

    return angle


def update_anglepose_df(df, dt, angles_list, headers):
    pose_dict = dict(zip(headers, angles_list))
    return df.append(pose_dict)


def update_xypose_df(df, datum, person, body_joints,
                     hand_joints=None, head_joints=None,
                     use_acc_treshold=False, acc_treshold=0.3):
    """
    Parameters
    ----------

    Returns
    -------
    """

    pose_dic = {
        c: datum.poseKeypoints[person, BODY_PARTS[c]][: 2]
        for it, c in enumerate(body_joints)
        if (not use_acc_treshold) or
           datum.poseKeypoints[person, BODY_PARTS[c]][2] > acc_treshold
    }
    if hand_joints is not None:
        pose_dic.update({
            (c + 'r'): datum.handKeypoints[0][person, HAND_PARTS[c]][: 2]
            for it, c in enumerate(hand_joints)
            if (not use_acc_treshold) or
               datum.handKeypoints[0][person, HAND_PARTS[c]][2] > acc_treshold
        })

        pose_dic.update({
            (c + 'l'): datum.handKeypoints[1][person, HAND_PARTS[c]][: 2]
            for it, c in enumerate(hand_joints)
            if (not use_acc_treshold) or
               datum.handKeypoints[1][person, HAND_PARTS[c]][2] > acc_treshold
        })

    # if head_joints is not None:

    return df.append(pose_dic, ignore_index=False)


def update_r_hands_list(dt, list_persons):
    """
    Parameters
    ----------

    Returns
    -------
    """

    curr_r_hands_list = make_hands_list(dt)

    r_hands = []
    for old_person in list_persons:
        correct_id, centroid = track_by_centroid(curr_r_hands_list,
                                                 old_person['centroid'],
                                                 ret_new_centroid=True)
        r_hands.append({
            'id': correct_id,
            'centroid': centroid
        })

    return r_hands


def update_person_list(dt, old_list_persons):
    """
    Atualiza a lista de pessoas baseado no centroid anterior e atual.

    Nessa função é associado o centroid mais proximo da lista velha com os
    novos calculados.

    Parameters
    ----------
    dt: Openpose Datum
        parametro que contem as poses fornecidas pelo openpose.

    old_list_persons: list
        Lista de dicionarios com cada dicionario no
        formato {id: int, centroid: (x,y)}.

    Returns
    -------
    lista de pessoas com os seus respectivos id corrigidos e seus centroides
    atuais.
    """

    curr_person_list = make_persons_list(dt)

    persons = []
    for old_person in old_list_persons:
        correct_id, centroid = track_by_centroid(curr_person_list,
                                                 old_person['centroid'],
                                                 ret_new_centroid=True)
        persons.append({
            'id': correct_id,
            'centroid': centroid
        })

    return persons


df_cols_angles = ['person', 'frame']
for it, pair in enumerate(POSE_PAIRS[:7]):
    first = INV_BODY_PARTS[BODY_PARTS[pair[0]] - 1]
    second = pair[0]
    third = pair[1]

    df_cols_angles.append(first + '-' + second + '-' + third)
print(df_cols_angles)

df_cols_xy = ['person', 'frame', "Nose", "Neck", "RShoulder", "RElbow",
              "RWrist", "LShoulder", "LElbow", "LWrist", "RHip"]

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "openpose/models/"
params["face"] = True
params["hand"] = True

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

person_last_list = None
right_hands_last_list = None

db_path = './drive/My Drive/librasdb-pose-window-500ms/'

for sign_dir in tqdm(os.listdir(db_path)):

    sign_name = copy.copy(sign_dir)
    sign_dir = os.path.join(db_path, sign_dir)

    for file_path in os.listdir(sign_dir):

        file_name = copy.copy(file_path)
        file_path = os.path.join(sign_dir, file_path)

        anglepose_df = pd.DataFrame(columns=df_cols_angles)
        xypose_df = pd.DataFrame(columns=df_cols_xy)
        single_video_ref = cv.VideoCapture(file_path)

        last_frame_pos = single_video_ref.get(cv.CAP_PROP_FRAME_COUNT)
        video_finished_correctly = True
        while True:
            ret, frame = single_video_ref.read()
            frame_pos = single_video_ref.get(cv.CAP_PROP_POS_FRAMES)
            if not ret:
                video_finished_correctly = frame_pos == (last_frame_pos - 1)
                break

            datum = op.Datum()

            imageToProcess = frame
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop([datum])

            if len(datum.handKeypoints) > 1:
                person_last_list = update_person_list(datum, person_last_list) \
                    if person_last_list is not None \
                    else make_persons_list(datum)

                r_hands_list = update_r_hands_list(datum, person_last_list)

                for it, person_tracked in enumerate(person_last_list):
                    person = person_tracked['id']
                    if it == 0:
                        centroid = person_tracked['centroid']
                        cv.circle(imageToProcess, tuple(centroid), 5,
                                  (128, 0, 128), -1)
                        r_hand_id = r_hands_list[0]['id']
                        centroid = r_hands_list[0]['centroid']
                        cv.circle(imageToProcess, tuple(centroid), 5,
                                  (128, 0, 128), -1)
                        centroid = make_xy_centroid(datum.handKeypoints[1][r_hand_id])
                        cv.circle(imageToProcess, tuple(centroid), 5,
                                  (128, 0, 128), -1)
                    angles_pose = []

                    for it, pair in enumerate(POSE_PAIRS[:7]):
                        first = BODY_PARTS[pair[0]] - 1
                        second = BODY_PARTS[pair[0]]
                        third = BODY_PARTS[pair[1]]

                        # checagem se tem pouca acuracia.

                        angle = calculate_angle_by_joints(datum, person, first,
                                                          second, third)
                        second = datum.poseKeypoints[person, second][:2]
                        third = datum.poseKeypoints[person, third][:2]

                        angles_pose.append(angle)

                    xypose_df = update_xypose_df(xypose_df, datum, person,
                                                 df_cols_xy)
                    anglepose_df = update_anglepose_df(anglepose_df, datum,
                                                       angles_pose,
                                                       df_cols_angles)

            plt.imshow(imageToProcess)
            plt.show()

        if not video_finished_correctly:
            print(last_frame_pos, frame_pos, file_path)
        else:
            file_name = file_name.split('.')[0] + '.csv'
            print(file_name)
            # if not os.path.exists(db_path + sign_name + '/xy/'):
            #     os.makedirs(db_path + sign_name + '/xy/')
            # xypose_df.to_csv(db_path + sign_name + '/xy/' + file_name)
            #
            # if not os.path.exists(db_path + sign_name + '/angle/'):
            #     os.makedirs(db_path + sign_name + '/angle/')
            # anglepose_df.to_csv(db_path + sign_name + '/angle/' + file_name)

        single_video_ref.release()