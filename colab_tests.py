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

    if len(x_points) == 0:
        return np.array([])

    return np.array([sum(x_points) / len(x_points),
                     sum(y_points) / len(y_points)])


def track_by_centroid(curr_persons_list, person_centroid,
                      ret_new_centroid=False, ret_dist=False):
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
        if c_person['centroid'].size == 0:
            continue

        centroids_dist = \
            np.linalg.norm(person_centroid - c_person['centroid'])
        if centroids_dist < curr_person_dist:
            curr_person_dist = centroids_dist
            correct_id = c_person['id']
            new_centroid = c_person['centroid']

    if ret_dist and ret_new_centroid:
        return correct_id, new_centroid, curr_person_dist
    if ret_dist and not ret_new_centroid:
        return correct_id, curr_person_dist

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
        centroid = make_xy_centroid(dt.handKeypoints[0][it])

        hand = {
            'id': it,
            'centroid': centroid,
            'failed': False
        }
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
            'centroid': make_xy_centroid(dt.poseKeypoints[it]),
            'failed': False
        }
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


def update_anglepose_df(df, angles_list, headers, person_id, frame):
    pose_dict = dict(zip(headers, angles_list))
    pose_dict.update({
        'person': person_id,
        'frame': frame
    })
    return df.append(pose_dict, ignore_index=True)


def update_xypose_df(df, datum, person, body_joints,
                     person_id, frame,
                     hand_joints=None, head_joints=None,
                     use_acc_treshold=False, acc_treshold=0.3):
    """
    Parameters
    ----------

    Returns
    -------
    """

    pose_dic = {
        'person': person_id,
        'frame': frame
    }
    body_parts_dict = {}
    for c in body_joints:
        try:
            if (not use_acc_treshold) or \
                    datum.poseKeypoints[person, BODY_PARTS[c]][
                        2] > acc_treshold:
                body_parts_dict.update(
                    {c: datum.poseKeypoints[person, BODY_PARTS[c]][: 2]})
            else:
                body_parts_dict.update({c: None})
        except IndexError:
            body_parts_dict.update({c: None})

    pose_dic.update(body_parts_dict)
    if hand_joints is not None:
        pose_dic.update({
            (c + 'r'): datum.handKeypoints[0][person, HAND_PARTS[c]][: 2] if (
                                                                                 not use_acc_treshold) or
                                                                             datum.handKeypoints[
                                                                                 0][
                                                                                 person,
                                                                                 HAND_PARTS[
                                                                                     c]][
                                                                                 2] > acc_treshold else None
            for c in hand_joints
        })

        pose_dic.update({
            (c + 'l'): datum.handKeypoints[1][person, HAND_PARTS[c]][: 2] if (
                                                                                 not use_acc_treshold) or
                                                                             datum.handKeypoints[
                                                                                 1][
                                                                                 person,
                                                                                 HAND_PARTS[
                                                                                     c]][
                                                                                 2] > acc_treshold else None
            for c in hand_joints
        })

    # if head_joints is not None:

    return df.append(pose_dic, ignore_index=True)


def update_r_hands_list(dt, list_persons, dist_threshold=120):
    """
    Parameters
    ----------

    Returns
    -------
    """

    curr_r_hands_list = make_hands_list(dt)

    r_hands = []
    for old_person in list_persons:
        failed_to_track = False
        correct_id, centroid, dist = track_by_centroid(curr_r_hands_list,
                                                       old_person['centroid'],
                                                       ret_new_centroid=True,
                                                       ret_dist=True)
        if dist > dist_threshold:
            centroid = old_person['centroid']
            correct_id = old_person['id']
            failed_to_track = True

        r_hands.append({
            'id': correct_id,
            'centroid': centroid,
            'failed': failed_to_track
        })

    return r_hands


def update_person_list(dt, old_list_persons, dist_threshold=30):
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
        failed_to_track = False
        correct_id, centroid, dist = track_by_centroid(curr_person_list,
                                                       old_person['centroid'],
                                                       ret_new_centroid=True,
                                                       ret_dist=True)
        if dist > dist_threshold:
            centroid = old_person['centroid']
            correct_id = old_person['id']
            failed_to_track = True

        persons.append({
            'id': correct_id,
            'centroid': centroid,
            'failed': failed_to_track
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

db_path = './drive/My Drive/librasdb-videos-window-500ms/'
acc_threshold = 0.2

for sign_dir in tqdm(os.listdir(db_path)):
    sign_name = copy.copy(sign_dir)
    sign_dir = os.path.join(db_path, sign_dir)
    for file_path in tqdm(os.listdir(sign_dir),
                          desc='Sign {}'.format(sign_name)):
        file_name = copy.copy(file_path)
        file_path = os.path.join(sign_dir, file_path)

        if not os.path.isfile(file_path):
            continue

        anglepose_df = pd.DataFrame(columns=df_cols_angles)
        xypose_df = pd.DataFrame(columns=df_cols_xy)
        single_video_ref = cv.VideoCapture(file_path)

        last_frame_pos = single_video_ref.get(cv.CAP_PROP_FRAME_COUNT)
        video_finished_correctly = True
        while True:
            ret, frame = single_video_ref.read()
            frame_pos = single_video_ref.get(cv.CAP_PROP_POS_FRAMES)
            if not ret:
                video_finished_correctly = frame_pos == last_frame_pos
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
                    angles_pose = []

                    for it, pair in enumerate(POSE_PAIRS[:7]):
                        first = BODY_PARTS[pair[0]] - 1
                        second = BODY_PARTS[pair[0]]
                        third = BODY_PARTS[pair[1]]

                        # TODO:
                        # -  adicionar Nan as posições com baixa acuracia e a
                        #    toda linha caso não tenha achado o esqueledo
                        try:
                            angle = calculate_angle_by_joints(datum, person,
                                                              first, second,
                                                              third)
                        except IndexError:
                            angles_pose.append(None)
                            continue

                        first = datum.poseKeypoints[person, first]
                        second = datum.poseKeypoints[person, second]
                        third = datum.poseKeypoints[person, third]

                        if first[2] > acc_threshold and \
                                second[2] > acc_threshold and \
                                third[2] > acc_threshold and \
                                person_tracked['failed']:
                            angles_pose.append(angle)
                        else:
                            angles_pose.append(None)

                    if not person_tracked['failed']:
                        xypose_df = update_xypose_df(xypose_df, datum, person,
                                                     df_cols_xy[2:],
                                                     person_tracked['id'],
                                                     frame_pos)

                    anglepose_df = update_anglepose_df(anglepose_df,
                                                       angles_pose,
                                                       df_cols_angles[2:],
                                                       person_tracked['id'],
                                                       frame_pos)

        if not video_finished_correctly:
            print(last_frame_pos, frame_pos, file_path)
        else:
            file_name = file_name.split('.')[0] + '.csv'
            if not os.path.exists(db_path + sign_name + '/no_hands-xy/'):
                os.makedirs(db_path + sign_name + '/no_hands-xy/')
            xypose_df.to_csv(db_path + sign_name + '/no_hands-xy/' + file_name)

            if not os.path.exists(db_path + sign_name + '/no_hands-angle/'):
                os.makedirs(db_path + sign_name + '/no_hands-angle/')
            anglepose_df.to_csv(
                db_path + sign_name + '/no_hands-angle/' + file_name)

        single_video_ref.release()