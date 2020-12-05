from libras_classifiers.librasdb_loaders import DBLoader2NPY
from pose_extractor.all_parts import *

from tqdm.auto import tqdm

import os
import pandas as pd
import numpy as np


def convert_xypose_to_dir_angle(first_pose, second_pose, third_pose):
    """
    Converte as três poses XY em angulos.

    Calcula o angulo entre as três poses. com os vetores first_pose->second_pose, second_pose->third_pose.

    Parameters
    ----------
    first_pose: np.array
        np.array([x, y]) com as coordenadas da imagem.

    second_pose: np.array
        np.array([x, y]) com as coordenadas da imagem.

    third_pose: np.array
        np.array([x, y]) com as coordenadas da imagem.

    Returns
    -------
        Angulo direcionado entre as poses. E caso aconteça algum erro na execução None sera retornado.

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


def make_angle_df_from_xy(sample: pd.DataFrame, no_hands=False, pbar: tqdm = None, sample_name: str = None,
                          body_angles=None, hand_angles=None):
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
        if isinstance(r[parts[0]], float) or isinstance(r[parts[1]], float) or isinstance(r[parts[1]], float):
            return None

        if (not ((0 in r[parts[0]]) or r[parts[0]] is None or any(np.isnan(r[parts[0]])) )) and \
           (not ((0 in r[parts[1]]) or r[parts[1]] is None or any(np.isnan(r[parts[1]])) )) and \
           (not ((0 in r[parts[2]]) or r[parts[2]] is None or any(np.isnan(r[parts[2]])) )):
            angle = convert_xypose_to_dir_angle(r[parts[0]][:2], r[parts[1]][:2], r[parts[2]][:2])

        return angle

    pose_angle_df = pd.DataFrame()
    if pbar is not None:
        pbar.reset(total=sample.shape[0])
        if sample_name is not None:
            pbar.set_description(f'{sample_name}')

    for row in sample.iterrows():
        row = row[1]
        angle_data = {}
        angle_data.update({'frame': [row.frame]})
        if body_angles is not None:
            for angle_part in body_angles:
                angle_part_name = '-'.join(angle_part)
                angle_pose = make_angle_pose(row, angle_part)

                angle_data.update({angle_part_name: [angle_pose]})
        if not no_hands and hand_angles is not None:
            for angle_part in hand_angles:
                left_hand_angle_part = ['left-' + x for x in angle_part]
                right_hand_angle_part = ['right-' + x for x in angle_part]

                left_angle_part_name = '-'.join(left_hand_angle_part)
                right_angle_part_name = '-'.join(right_hand_angle_part)

                left_angle_pose = make_angle_pose(row, left_hand_angle_part)

                angle_data.update({left_angle_part_name: [left_angle_pose]})

                right_angle_pose = make_angle_pose(row, right_hand_angle_part)
                angle_data.update({right_angle_part_name: [right_angle_pose]})

        pose_angle_df = pose_angle_df.append(pd.DataFrame(data=angle_data), ignore_index=True)

        if pbar is not None:
            pbar.update(1)
            # pbar.refresh()

    return pose_angle_df


def convert_all_samples_xy_2_angle(db_path, no_hands=False, custom_dir=None):
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

    sample_pbar = tqdm(position=2)

    amount_samples = count_samples_in_database(db_path, no_hands, custom_dir=custom_dir)
    folder_pbar = tqdm(position=1)
    folder_pbar.reset(total=amount_samples)

    body_angle_pairs_to_7 = [[1, 2, 3], [2, 3, 4], [1, 5, 6], [5, 6, 7], [2, 1, 5]]
    body_angle_pairs_to_7 = [[INV_BODY_PARTS[x[0]], INV_BODY_PARTS[x[1]], INV_BODY_PARTS[x[2]]]
                             for x in body_angle_pairs_to_7]
    hand_angle_pair = [[0, 2, 4], [0, 6, 8], [0, 10, 12], [0, 14, 16], [0, 18, 20]]
    hand_angle_pair = [[INV_HAND_PARTS[x[0]], INV_HAND_PARTS[x[1]], INV_HAND_PARTS[x[2]]]
                       for x in hand_angle_pair]

    for sample_xy, class_name, sample_name in yield_all_db_samples(db_path, no_hands):

        signs_info = sample_name.split('---')
        v_part = signs_info[
            0]  # signs_info[3] + signs_info[4] if 'Inventário Nacional de Libras' in sample_name else signs_info[3]
        only_sample_name = f'{v_part}---{signs_info[1]}---{signs_info[2]}---{signs_info[3]}---{signs_info[-1][0]}' \
                           f'---sample-angle.csv'

        sample_path = os.path.join(db_path, class_name, dir_before_samples_name, only_sample_name)

        folder_path = os.path.join(db_path, class_name, dir_before_samples_name)
        os.makedirs(folder_path, exist_ok=True)

        if os.path.exists(sample_path):
            folder_pbar.update(1)
            # folder_pbar.refresh()
            continue

        folder_pbar.set_description(class_name)

        sample_angle_df = make_angle_df_from_xy(sample_xy, False, pbar=sample_pbar, sample_name=only_sample_name,
                                                body_angles=body_angle_pairs_to_7, hand_angles=hand_angle_pair)

        sample_angle_df.to_csv(sample_path)

        folder_pbar.update(1)
        # folder_pbar.refresh()


def count_samples_in_database(db_path, no_hands=False, xy=True, custom_dir=None):
    amount_samples = 0
    dir_before_samples_name = 'no_hands' if no_hands else 'hands'
    dir_before_samples_name += '-xy' if xy else '-angle'

    if custom_dir is not None:
        dir_before_samples_name = custom_dir

    for sample_dir in os.listdir(db_path):
        sample_dir = os.path.join(db_path, sample_dir, dir_before_samples_name)
        samples_at_dir = list(filter(lambda x: '.csv' in x, os.listdir(sample_dir)))

        amount_samples += len(samples_at_dir)

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

    # dir_before_samples_name = 'no_hands' if no_hands else 'hands'
    # dir_before_samples_name += '-xy' if xy else '-angle'
    # for sample_dir in os.listdir(db_path):
    #     class_name = sample_dir
    #     sample_dir = os.path.join(db_path, sample_dir, dir_before_samples_name)
    #
    #     for sample_path in os.listdir(sample_dir):
    #         sample_name = sample_path
    #         sample_path = os.path.join(sample_dir, sample_path)
    #         if '.csv' in sample_path:
    #             sample = pd.read_csv(sample_path)
    #             sample = sample.applymap(DBLoader2NPY.parse_npy_vec_str)
    #             yield sample, class_name, sample_name
    #         else:
    #             yield None, class_name, sample_name
    samples_path, class_dirs = DBLoader2NPY.read_all_db_folders(
        db_path=db_path, only_that_classes=None, angle_or_xy='xy-hands', custom_internal_dir=''
    )

    for samples in samples_path:

        xy_sample = pd.read_csv(samples[0])
        xy_sample = xy_sample.applymap(DBLoader2NPY.parse_npy_vec_str)
        cls_name = class_dirs[samples[1]].replace('\\', '/').split('/')[-1]
        sample_name = samples[0].replace('\\', '/').split('/')[-1]

        yield xy_sample, cls_name, sample_name


if __name__ == '__main__':
    np.seterr(all='raise')
    convert_all_samples_xy_2_angle('/home/usuario/Documents/clean_sign_db_front_view', custom_dir='')
