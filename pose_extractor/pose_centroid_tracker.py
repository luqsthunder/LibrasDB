import numpy as np
import pandas as pd
import cv2 as cv
import os
from copy import copy
from tqdm import tqdm
from pose_extractor.all_parts import BODY_PARTS_NAMES, HAND_PARTS_NAMES
from pose_extractor.openpose_extractor import DatumLike, OpenposeExtractor
from pose_extractor.find_signaling_centroid import FindSignalingCentroid
from pose_extractor.df_utils import update_xy_pose_df
from libras_classifiers.generate_dataframe_person_2_sign import \
    DataframePerson2Sign


class PoseCentroidTracker:
    """
    Classe para rastrear os esqueletos e mãos, a classe funciona minimizando
    a distância entre os centroid dos esqueletos rastreados e atribuindo os
    esqueletos mais proximos aos seus respectivos ID, e um novo ID aos
    esqueletos mais distantes que um limitante (threshold).
    """

    def __init__(self, all_video_csv_path, db_path=None, folder_2_track=None,
                 centroids_df_path=None, openpose_path='../openpose',
                 persons_id=None, body_parts=None, hand_parts=None,
                 head_parts=None):
        """
        Parameters
        ----------
        persons_id: list[int] or None

        body_parts: list[str] or None

        hand_parts: list[str] or None

        head_parts: list[str] or None

        """
        self.persons_id = persons_id if persons_id is not None else []
        self.body_parts = BODY_PARTS_NAMES if body_parts is None else body_parts
        self.hands_parts = HAND_PARTS_NAMES if hand_parts is None else hand_parts
        self.head_parts = head_parts

        self.curr_dt = None
        self.signaler_find = FindSignalingCentroid(all_video_csv_path)
        self.pose_extractor = OpenposeExtractor(openpose_path=openpose_path)
        self.person_2_sign = DataframePerson2Sign(None)

        self.centroids_df = pd.read_csv(centroids_df_path) \
            if centroids_df_path is not None else None
        self.folder_2_track = folder_2_track
        if self.centroids_df is not None:
            self.centroids_df['centroid'] = \
                self.centroids_df['centroid'].apply(self.parse_npy_vec_str)
        print('end')

    @staticmethod
    def parse_npy_vec_str(str_array_like):
        if not isinstance(str_array_like, str):
            return str_array_like

        res = str_array_like[1:len(str_array_like) - 1].split(' ')
        recovered_np_array = []
        for r in res:
            if r == '' or ' ' in r:
                continue
            f = float(r)
            recovered_np_array.append(f)
        return np.array(recovered_np_array)


    def retrive_persons_centroid_from_sign_df(self, folder_path, db_path,
                                              pbar=None):
        """
        Encontra o centroid das pessoas presentes em um video e legenda presente no
        dataframe da base de dados. Relacionando o centroid das pessoas ao id da
        legenda. Para facilitar a extração das poses de quem fala um sinal. 

        Parameters
        ----------
        folder_path: str
            Nome do folder (atributo do dataframe).

        db_path: str
            Path de onde é encontrado a base de dados de videos.

        pbar: tqdm.tqdm
            Barra de progresso a ser utilizado enquanto é extraido os centroids das
            pessoas presentes no video.

        Raise
        -----
        RumtimeError:
            Caso o video termine inexperadamente.

        Returns
        -------
        pandas.DataFrame
            Dataframe com as seguintes colunas: folder, talker_id, centroid, que
            indicam respectivamente nome do folder (atributo do dataframe da base de
            dados de sinais), id da pessoa relacionado a legenda e o centroid da
            pessoa.
        """
        persons_alone = \
            self.signaler_find.find_where_signalers_talks_alone(folder_path)

        src_file_name = os.path.join(db_path, folder_path)
        video = cv.VideoCapture(src_file_name)

        df_persons_centroid_video = pd.DataFrame(columns=['folder',
                                                          'talker_id',
                                                          'centroid'])

        for person_sub_id, alone_talk in persons_alone.items():
            person_sub_id = int(person_sub_id)

            video.set(cv.CAP_PROP_POS_FRAMES, alone_talk['beg'])
            end_pos = alone_talk['end']
            fps = video.get(cv.CAP_PROP_FPS)
            end_pos = int(alone_talk['beg'] + fps * 5) + 1 \
                if end_pos > int(alone_talk['beg'] + fps * 5) + 1 else end_pos

            persons_centroids = [[], []]
            pose_df_cols = ['person', 'frame']
            pose_df_cols.extend(self.body_parts)
            pose_df_cols.extend(['l' + x for x in self.hands_parts])
            pose_df_cols.extend(['r' + x for x in self.hands_parts])

            pose_df = pd.DataFrame(columns=pose_df_cols)

            if pbar is not None:
                pbar.reset(total=int(end_pos - alone_talk['beg']))

            for _ in range(int(end_pos - alone_talk['beg'])):

                ret, frame = video.read()
                if not ret:
                    raise RuntimeError(f'Video Ended Beforme was expected.'
                                       f'The video is: {folder_path}')
                # plt.imshow(frame)
                # plt.show()

                dt = self.pose_extractor.extract_poses(frame)

                curr_frame = int(video.get(cv.CAP_PROP_POS_FRAMES))

                curr_centroids = list(map(self.make_xy_centroid,
                                          dt.poseKeypoints))

                x_mid_point = sum(map(lambda x: x[0],
                                      curr_centroids))
                x_mid_point = x_mid_point / len(curr_centroids)

                left_id = 0 if curr_centroids[0][0] < x_mid_point else 1
                right_id = 1 if left_id == 0 else 0
                persons_pos_id = [left_id, right_id]
                persons_centroids[0].append(
                    curr_centroids[left_id])
                persons_centroids[1].append(
                    curr_centroids[right_id])

                # construindo o df com a posição relativa apenas a divisão da
                # cena para falante da direita e falante da esquerda.
                # para apos achar o que mais fala no tempo atual e atribuir
                # a ele o ID correto que será o da legenda.

                for person_id in range(2):
                    pose_df = update_xy_pose_df(dt, pose_df, curr_frame,
                                                person_id,
                                                persons_pos_id[person_id],
                                                self.body_parts,
                                                None)
                if pbar is not None:
                    pbar.update(1)
                    pbar.refresh()

            print(persons_centroids)
            persons_body_centroid = [persons_centroids[0][0], persons_centroids[1][0]]
            talking_person_id = \
                self.person_2_sign.process_single_sample(pose_df)

            talking_person_id = talking_person_id['dist']['id']
            talker_centroid = persons_body_centroid[talking_person_id]

            curr_data = pd.DataFrame(data=dict(folder=[folder_path],
                                               talker_id=[person_sub_id],
                                               centroid=[talker_centroid]))
            df_persons_centroid_video = \
                df_persons_centroid_video.append(curr_data, ignore_index=True)

            not_talking_person_id = 0 if talking_person_id == 1 else 1
            print(not_talking_person_id, talking_person_id)

            not_talker_centroid = persons_body_centroid[not_talking_person_id]
            not_talker_sub_id = 1 if person_sub_id == 2 else 2

            print(not_talker_sub_id, person_sub_id, person_sub_id == 2, 
                  int(person_sub_id) == 2)

            curr_data = pd.DataFrame(data=dict(folder=[folder_path],
                                               talker_id=[not_talker_sub_id],
                                               centroid=[not_talker_centroid]))
            df_persons_centroid_video = \
                df_persons_centroid_video.append(curr_data, ignore_index=True)
            break

        video.release()
        return df_persons_centroid_video

    def subdivide_2_persons_scene(self, folder):
        if self.centroids_df is None:
            raise ValueError('self.centroids_df must non None if wanted to '
                             'divide scene')

        sample = self.centroids_df[self.centroids_df['folder'] == folder]

        x_mid = sum(list(map(lambda x: x[0], sample.centroid))) / 2

        left_person_id = sample['talker_id'].iloc[0] \
            if sample.centroid.iloc[0][0] < x_mid \
            else sample['talker_id'].iloc[1]

        right_person_id = 2 if left_person_id == 1 else 1

        return x_mid, left_person_id, right_person_id

    def filter_persons_by_x_mid(self, dt, first_x_mid):

        if dt.poseKeypoints.size > 1:
            curr_body_centroids = list(map(self.make_xy_centroid,
                                           dt.poseKeypoints))

            x_mid_body = sum(map(lambda x: x[0], curr_body_centroids))
            x_mid_body = x_mid_body / len(curr_body_centroids)

            if len(curr_body_centroids) > 1:
                left_person_in_dt = 0 \
                    if curr_body_centroids[0][0] < x_mid_body else 1
                right_person_in_dt = 1 if left_person_in_dt == 0 else 0
            else:
                left_person_in_dt = 0 \
                    if curr_body_centroids[0][0] < first_x_mid else None
                right_person_in_dt = 0 \
                    if curr_body_centroids[0][0] > first_x_mid else None
        else:
            left_person_in_dt, right_person_in_dt = None, None

        if dt.handKeypoints[0].size > 1:
            curr_r_hands_centroids = list(map(self.make_xy_centroid,
                                              dt.poseKeypoints))

            x_mid_hands = sum(map(lambda x: x[0], curr_body_centroids))
            x_mid_hands = x_mid_hands / len(curr_body_centroids)

            if len(curr_r_hands_centroids) > 1:
                left_person_r_hand_dt = 0 \
                    if curr_r_hands_centroids[0][0] < x_mid_hands else 1
                right_person_r_hand_dt = 1 if left_person_r_hand_dt == 0 else 0
            else:
                left_person_r_hand_dt = 0 \
                    if curr_r_hands_centroids[0][0] < first_x_mid else None
                right_person_r_hand_dt = 0 \
                    if curr_r_hands_centroids[0][0] < first_x_mid else None
        else:
            left_person_r_hand_dt, right_person_r_hand_dt = None, None

        return ([left_person_in_dt, left_person_r_hand_dt],
                [right_person_in_dt, right_person_r_hand_dt])

    def __remove_unused_joints(self, dt: DatumLike):
        """
        param: dt

        returns:
        """
        pass

    def update(self, datum: DatumLike):
        self.curr_dt = copy(datum)
        # self.clean_current_datum()

    @staticmethod
    def make_xy_centroid(pose):
        """
        Função para calcular um centroid de um conjunto de juntas no formato xy.

        Parameters
        ----------
        pose: numpy.array or numpy.ndarray
            Contem um vetor de juntas contend.

        Returns
        -------
        np.array
            Centroid das juntas xy informadas.
        """

        length = pose.shape[0]
        x_part = np.sum(pose[:, 0])
        y_part = np.sum(pose[:, 1])

        centroid = np.array([x_part / length, y_part / length])

        return centroid
