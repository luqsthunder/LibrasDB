from libras_classifiers.librasdb_loaders import DBLoader2NPY
from pose_extractor.all_parts import *

from copy import copy
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

import os
import pandas as pd
import numpy as np
import multiprocessing


def convert_xypose_to_dir_angle(first_pose, second_pose, third_pose):
    """
    Parameters
    ----------

    Returns
    -------
    """
    try:
        first_vec = np.array(second_pose) - np.array(first_pose)
        second_vec = np.array(third_pose) - np.array(second_pose)

        first_vec = first_vec / np.linalg.norm(first_vec)
        second_vec = second_vec / np.linalg.norm(second_vec)

        #vecs_norm = (np.linalg.norm(np.array(first_vec)) *
        #             np.linalg.norm(np.array(second_vec)))

        #angle = np.arccos(np.dot(first_vec, second_vec) / vecs_norm)
        angle = np.arccos(np.dot(first_vec, second_vec))

        angle_sign = np.sign(first_vec[0] * second_vec[1] -
                             first_vec[1] * second_vec[0])
    except BaseException as e:
        #print(e)
        #error_str = f'f pose: {first_pose}, s pose: {second_pose}, t pose: {third_pose}\n'\
        #            f'f vec: {first_vec}, s vec: {second_vec}. Dot: {np.dot(first_vec, second_vec)}. Angle: {angle}'
        #raise RuntimeError(error_str)
        return None

    return angle_sign * angle


def make_angle_df_from_xy(sample: pd.DataFrame, no_hands=False, pbar: tqdm = None, sample_name: str = None):
    """
    Convert a XY-pose pd.DataFrame to a angle-pose pd.DataFrame.

    Using the pose_extractor.all_parts.HAND_ANGLE_PAIR and pose_extractor.all_parts.BODY_ANGLE_PAIR it's calculated
    the directional angle between the joints. Those joints was made using opencv openpose dicts and openpose output
    file.

    Parameters
    ----------
    sample: pd.DataFrame
        XY-pose DataFrame
    no_hands: bool
        Indicates if will use hands.
    pbar: tqdm
       Progress bar.

    sample_name: str
        Sample name to put in progress bar description.

    Returns
    -------
        Angle-pose pd.DataFrame.
    """

    def make_angle_pose(r, parts):
        """

        Parameters
        ----------
        r: row containing pose data
        parts

        Returns
        -------

        """
        angle = None
        if (not ((0 in r[parts[0]]) or r[parts[0]] is None)) and \
           (not ((0 in r[parts[1]]) or r[parts[1]] is None)) and \
           (not ((0 in r[parts[2]]) or r[parts[2]] is None)):
            angle = convert_xypose_to_dir_angle(r[parts[0]][:2], r[parts[1]][:2], r[parts[2]][:2])

        return angle

    df_cols = ['frame'] + ['-'.join(x) for x in BODY_ANGLE_PAIRS] + ['l-' + ('-'.join(x)) for x in HAND_ANGLE_PAIRS] + \
              ['r-' + ('-'.join(x)) for x in HAND_ANGLE_PAIRS]
    pose_angle_df = pd.DataFrame()
    m_sample = sample.copy()
    if pbar is not None:
        pbar.reset(total=m_sample.shape[0])
        if sample_name is not None:
            pbar.set_description(f'{sample_name}')

    for row in m_sample.iterrows():
        row = row[1]
        angle_data = {}
        angle_data.update({'frame': [row.frame]})
        for angle_part in BODY_ANGLE_PAIRS:
            angle_part_name = '-'.join(angle_part)
            angle_pose = make_angle_pose(row, angle_part)

            angle_data.update({angle_part_name: [angle_pose]})
        if not no_hands:
            for angle_part in HAND_ANGLE_PAIRS:
                left_hand_angle_part = ['l-' + x for x in angle_part]
                right_hand_angle_part = ['r-' + x for x in angle_part]

                left_angle_part_name = '-'.join(left_hand_angle_part)
                right_angle_part_name = '-'.join(right_hand_angle_part)

                left_angle_pose = make_angle_pose(row, left_hand_angle_part)

                angle_data.update({left_angle_part_name: [left_angle_pose]})

                right_angle_pose = make_angle_pose(row, right_hand_angle_part)
                angle_data.update({right_angle_part_name: [right_angle_pose]})

        pose_angle_df = pose_angle_df.append(pd.DataFrame(data=angle_data), ignore_index=True)

        if pbar is not None:
            pbar.update(1)
            pbar.refresh()

    return pose_angle_df


def convert_all_samples_xy_2_angle(db_path, no_hands=False):
    """
    Convert all samples in xy folder to angle.

    Parameters
    ----------
    db_path
    no_hands

    Returns
    -------

    """

    dir_before_samples_name = ('no_hands' if no_hands else 'hands') + '-angle'

    amount_samples = count_samples_in_database(db_path, no_hands)
    folder_pbar = tqdm(position=1)
    folder_pbar.reset(total=amount_samples)
    db_generator = yield_all_db_samples(db_path, no_hands)

    amount_workers = multiprocessing.cpu_count() // 2
    thread_executor = \
        ThreadPoolExecutor(max_workers=amount_workers*2)

    sample_pbars = [tqdm(position=(x + 1)) for x in range(1, 3)]

    for _ in range(0, amount_samples, amount_workers):

        # sample_xy1, class_name1, sample_name1 = next(db_generator)
        # sample_xy2, class_name2, sample_name2 = next(db_generator)
        # sample_xy3, class_name3, sample_name3 = next(db_generator)
        # sample_xy4, class_name4, sample_name4 = next(db_generator)

        data_list = [next(db_generator) for _ in range(amount_workers)]
        folder_pbar.set_description(data_list[0][1])

        futures = [
            thread_executor.submit(make_angle_df_from_xy, x[0], x[1], sample_pbars[it], x[2])
            for it, x in enumerate(data_list) if not os.path.exists(os.path.join(db_path, data_list[it][1],
                                                                                 dir_before_samples_name,
                                                                                 data_list[it][2]))
        ]

        all_done = [False] * len(futures)
        while not all(all_done):
            all_done = [f.done() for f in futures]

        res = [f.result() for f in futures]

        for it, r in enumerate(res):
            if r is None:
                continue
            sample_path = os.path.join(db_path, data_list[it][1], dir_before_samples_name, data_list[it][2])
            r.to_csv(sample_path)

        # sample_angle_df = make_angle_df_from_xy(sample_xy, no_hands, pbar=sample_pbar, sample_name=sample_name)
        # sample_path = os.path.join(db_path,  class_name, dir_before_samples_name, sample_name)
        # sample_angle_df.to_csv(sample_path)
        folder_pbar.update(amount_workers)
        folder_pbar.refresh()


def count_samples_in_database(db_path, no_hands=False, xy=True):
    amount_samples = 0
    dir_before_samples_name = 'no_hands' if no_hands else 'hands'
    dir_before_samples_name += '-xy' if xy else '-angle'

    for sample_dir in os.listdir(db_path):
        sample_dir = os.path.join(db_path, sample_dir, dir_before_samples_name)
        amount_samples += len(os.listdir(sample_dir))

    return amount_samples


def yield_all_db_samples(db_path, no_hands=False, xy=True):
    """
    Returns each sample in database. Works as generator.

    Parameters
    ----------
    db_path: str
        path to where is database.

    no_hands: bool
        boolean for use samples with hands or not.

    xy: bool
        Indicates if samples to load are XY-pose samples.

    Yields
    -------
        Each sample. And they are a pd.DataFrame. If sample was not a CSV will be considered an error so None will be
        yielded.
    """

    dir_before_samples_name = 'no_hands' if no_hands else 'hands'
    dir_before_samples_name += '-xy' if xy else '-angle'
    for sample_dir in os.listdir(db_path):
        class_name = sample_dir
        sample_dir = os.path.join(db_path, sample_dir, dir_before_samples_name)

        for sample_path in os.listdir(sample_dir):
            sample_name = sample_path
            sample_path = os.path.join(sample_dir, sample_path)
            if '.csv' in sample_path:
                sample = pd.read_csv(sample_path)
                sample = sample.applymap(DBLoader2NPY.parse_npy_vec_str)
                yield sample, class_name, sample_name
            else:
                yield None, class_name, sample_name


if __name__ == '__main__':
    np.seterr(all='raise')
    convert_all_samples_xy_2_angle('../libras-db-folders')
