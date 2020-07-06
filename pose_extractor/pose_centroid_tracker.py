import numpy as np
import pandas as pd
import cv2 as cv
import os
from copy import copy
from shutil import copyfile
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

    def __init__(self, all_video_csv_path, body_dist_threshold=20,
                 hand_dist_threshold=20, hand_body_dist_threshold=50,
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
        self.body_dist_threshold = body_dist_threshold
        self.hand_dist_threshold = hand_dist_threshold
        self.hand_body_dist_threshold = hand_body_dist_threshold

        # essa variavel contem informação sobre os corpos, maos e cabeças das
        # pessoas. A informção esta no seguinte formato:
        # {
        #   id: int,                    id principal, indicando a posição na
        #                               legenda ou primeiro id registro para
        #                               pessoa.
        #
        #   pos_id: int,                Posição da pose da pessoa no datum atual
        #                               (self.curr_dt).
        #
        #   centroid: np.array([x, y]), centroid da pessoa.
        #
        #   right_hand_pos_id: int.     Posição da pose da mão direita da pessoa
        #                               no datum atual.
        #
        #   left_hand_pos_id: int       Posição da pose da mão esquerda da
        #                               pessoa no datum atual.
        #
        #   left_hand_centroid: int,    Centroid atual da mão esquerda.
        #
        #   right_hand_centroid: int    Centroid atual da mão direita.
        # }
        self.all_persons_all_parts_centroid = {}

        self.last_persons_list = None
        self.last_hands_list = None
        self.curr_dt = None
        self.signaler_find = FindSignalingCentroid(all_video_csv_path)
        openpose_path = '../openpose/'
        self.pose_extractor = OpenposeExtractor(openpose_path=openpose_path)
        self.person_2_sign = DataframePerson2Sign(None)

    def retrive_persons_centroid_from_sign_df(self, folder_path, db_path, pbar=None):
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

        # alguns folder_names estão com espaço no inicio de cada diretorios,
        # para copia-los é necessário remover os espaços do inicio do nome
        # dos diretorios.
        correct_f_name = folder_path.split('/')
        correct_f_name = list(map(lambda x: x if x[0] != ' ' else x[1:],
                                  correct_f_name))
        correct_f_name = os.path.join(*correct_f_name)

        src_file_name = os.path.join(db_path, folder_name)
        dst_file_name = './v0.mp4'
        copyfile(src_file_name, dst_file_name)
        video = cv.VideoCapture(dst_file_name)
        persons_body_centroid = None
        x_mid_point = None
        df_persons_centroid_video = pd.DataFrame(columns=['folder',
                                                          'talker_id',
                                                          'centroid'])

        for person_sub_id, alone_talk in persons_alone.items():
            video.set(cv.CAP_PROP_POS_FRAMES, alone_talk['beg'])
            end_pos = alone_talk['end']
            fps = video.get(cv.CAP_PROP_FPS)
            end_pos = int(alone_talk['beg'] + fps * 5) + 1 if end_pos > int(alone_talk['beg'] + fps * 5) + 1 else end_pos

            persons_centroids = [[], []]
            pose_df_cols = ['person', 'frame']
            pose_df_cols.extend(self.body_parts)
            pose_df_cols.extend(self.hands_parts)
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

                # TODO:
                # [ok] - extrai o X do centroid de cada pessoa e pega o valor
                #        do meio como elas estão sentadas vc tem como dividir a
                #        cena em 2 e ter menos trabalho para rastreamento.
                #
                # [ok] - Separa as pessoas pelo ponto do meio analisando o
                #        centroid. Todas as duas pessoas presentes na cena,
                #        são ordenadas da esquerda para direita.
                #        Logo se o centroid anterior ao meio deve ser a
                #        posição 0 e o a direita a posição 1.
                #
                # [  ] - Checar se os anteriores estão certos.

                if persons_body_centroid is None:
                    persons_body_centroid = list(map(self.make_xy_centroid,
                                                     dt.poseKeypoints))
                    x_mid_point = sum(map(lambda x: x[0],
                                      persons_body_centroid))
                    x_mid_point = x_mid_point / len(persons_body_centroid)

                curr_frame = int(video.get(cv.CAP_PROP_POS_FRAMES))

                curr_centroids = list(map(self.make_xy_centroid,
                                          dt.poseKeypoints))

                left_id = 0 if curr_centroids[0][0] < x_mid_point else 1
                right_id = 1 if left_id == 0 else 0
                persons_pos_id = [left_id, right_id]
                persons_centroids[left_id].append(
                    curr_centroids[left_id])
                persons_centroids[right_id].append(
                    curr_centroids[right_id])

                # construindo o df com a posição relativa apenas a divisão da
                # cena para falante da direita e falante da esquerda.
                # para apos achar o que mais fala no tempo atual e atribuir
                # a ele o ID correto que será o da legenda.

                for person_id in persons_pos_id:
                    pose_df = update_xy_pose_df(dt, pose_df, curr_frame,
                                                person_id,
                                                persons_pos_id[person_id],
                                                self.body_parts,
                                                None)
                if pbar is not None:
                    pbar.update(1)
                    pbar.refresh()
            # TODO:
            # [ok] - Achar quem tem mais enegergia/distancia gasta na DF e
            #        atribuir o ID da pessoa na legenda ao centroid dessa
            #        pessoa.

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
            not_talker_centroid = persons_body_centroid[not_talking_person_id]
            not_talker_sub_id = 1 if person_sub_id == 2 else 2
            curr_data = pd.DataFrame(data=dict(folder=[folder_path],
                                               talker_id=[not_talker_sub_id],
                                               centroid=[not_talker_centroid]))
            df_persons_centroid_video = \
                df_persons_centroid_video.append(curr_data, ignore_index=True)
            break

        os.remove(dst_file_name)
        return df_persons_centroid_video

    def __make_persons_list(self, dt):
        """
        É construido uma lista dicionarios com ID e centroid a partir do datum
        passado. Nessa lista é construido sem levar em consideração os ID
        corretos de cada pessoa. O intuito desse método é apenas organizar as
        poses.

        Parameters
        ----------
        dt: Openpose Datum
            Datum contendo as poses a serem utilizadas

        Returns
        -------
        List[dict]
            Contém em cada dicionario as inforções do ID relacionado a posição
            da pessoa no datum e o seu centroid.
        """

        persons_list = []
        for it, body_pose in enumerate(dt.poseKeypoints):
            person = {
                'id': it,
                'centroid': self.make_xy_centroid(body_pose)}
            persons_list.append(person)
        return persons_list

    def __make_r_hands_list(self, dt: DatumLike):
        """
        É construido uma lista dicionarios com ID e centroid a partir do datum
        passado. Nessa lista é construido sem levar em consideração os ID
        corretos de cada pessoa. O intuito desse método é apenas organizar as
        poses.

        Parameters
        ----------
        dt: Openpose Datum
            Datum contendo as poses a serem utilizadas

        Returns
        -------
        List[dict]
            Contém em cada dicionario as inforções do ID relacionado a posição
            da pessoa no datum e o seu centroid.
        """

        hands_list = []
        for it, hand_pose in enumerate(dt.handsKeypoints[0]):
            hands = {
                'id': it,
                'centroid': self.make_xy_centroid(hand_pose)}
            hands_list.append(hands)
        return hands_list

    def __make_hands_lists(self, dt):
        """
        Parameters
        ----------
        dt: Openpose Datum
            Datum contendo as poses a serem utilizadas

        Returns
        -------
        """

        hands_list = []
        for it, hands in enumerate(dt.handKeypoints):
            person = {
                'id': it,
                'centroid': self.make_xy_centroid(hands)}
            hands_list.append(person)
        return hands_list

    def __remove_unused_joints(self, dt: DatumLike):
        """
        param: dt

        returns:
        """
        pass

    def update(self, datum: DatumLike, curr_frame):
        self.curr_dt = copy(datum)
        self.clean_current_datum()

        # TODO:
        # [  ] - checar o meio atual, separar os IDs nos datums.
        # [  ] - Atribuir o ID relacionado a cada lado.


    def get_from_id_persons(self, person_id):
        pass

    def hand_owners_by_centroid(self):
        pass

    def update_persons_list(self, person_lists):
        """
        Atualiza a lista de pessoas baseado no centroid anterior e atual.

        Nessa função é associado o centroid mais proximo da lista velha com os
        novos calculados.

        param: person_lists: list

        return: persons: list
            lista de pessoas com os seus respectivos id corrigidos e seus
            centroides atuais.
        """

        persons = []
        for old_person in self.last_persons_list:
            failed_2_track = False
            correct_pos_id, centroid, dist, _ = \
                self.find_closest_centroid(person_lists,
                                           old_person['centroid'])

            if dist > self.body_dist_threshold:
                centroid = old_person['centroid']
                correct_pos_id = old_person['id']
                failed_2_track = True

            persons.append({
                'id': correct_pos_id,
                'centroid': centroid,
                'failed': failed_2_track
            })
        return persons

    @staticmethod
    def find_closest_centroid(curr_persons_list, old_centroid):
        """
        Função para achar o centroid mais próximo das pessoas dentro da
        curr_person_list em relação ao old_centroid

        Parameters
        ----------
        curr_persons_list: list


        old_centroid: np.array([x, y])

        Returns
        -------
        closest_centroid: np.array([x, y]), correct_pos_id: int,
        curr_person_dist: float
            O closest_centroid indica o centroid mais proximo das pessoas
            presentes na lista atual em relação ao antigo. O correct_pos_id é a
            posição da pessoa correspondente ao centroid antigo na nova lista.
            E curr_person_dist é a distância entre o centroid antigo e novo.

        """
        correct_pos_id = -1
        curr_person_dist = 9999999
        closest_centroid = None

        for c_person in curr_persons_list:
            if c_person['centroid'].size == 0:
                continue

            centroids_dist = \
                np.linalg.norm(old_centroid - c_person['centroid'])

            if centroids_dist < curr_person_dist:
                curr_person_dist = centroids_dist
                correct_pos_id = c_person['id']
                closest_centroid = c_person['centroid']

        return closest_centroid, correct_pos_id, curr_person_dist

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


class PoseAngleDataframe:
    """
    Classe para axiliar na construção dos dataframes das poses em angulos.

    Essa classe pode ser utilizada com contexto do python e.g:

    ```
    with PoseAngleDataframe(['Neck', 'Nose', ...], 'my-pose-df.csv') as pad:
        # process
    ```

    """

    def __init__(self, joint_names, output_file):
        self.joint_names = joint_names
        self.df = pd.DataFrame()
        self.output_file = output_file

    def update_dataframe(self, new_poses):
        pass

    def close(self):
        self.__exit__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        self.df.to_csv(self.output_file)
