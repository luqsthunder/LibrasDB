import sys
sys.path.append('.')
sys.path.append('pose_extractor')

from pose_extractor.df_utils import update_xy_pose_df_single_person
from pose_extractor.openpose_extractor import OpenposeExtractor, DatumLike
from pose_extractor.extract_signs_from_video import process_single_sample
from pose_extractor.pose_centroid_tracker import PoseCentroidTracker
import pose_extractor.all_parts as all_parts

import numpy as np
import os
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorama import init


class ExtractMultipleVideos:

    def __init__(self, db_path: str, all_videos: pd.DataFrame or str, vid_sync: str or pd.DataFrame,
                 all_persons_subtitle: str or pd.DataFrame, openpose_path: str,
                 path_to_save_dfs: str, needed_signs_list: list = None, gpu_count: int = 1):
        """

        Parameters
        ----------
        db_path
        all_videos
        vid_sync
        all_persons_subtitle
        openpose_path
        needed_signs_list

        """

        def remove_unnamed_0_col(df):
            if 'Unnamed: 0' in df.keys():
                return df.drop(columns=['Unnamed: 0'])
            return df

        self.gpu_count = gpu_count
        self.db_path = db_path
        self.all_videos = all_videos if isinstance(all_videos, pd.DataFrame) else pd.read_csv(all_videos)
        self.all_videos = remove_unnamed_0_col(self.all_videos)

        self.extractor = OpenposeExtractor(openpose_path=openpose_path)

        self.vid_sync = vid_sync if isinstance(vid_sync, pd.DataFrame) else pd.read_csv(vid_sync)
        self.vid_sync = remove_unnamed_0_col(self.vid_sync)

        self.all_persons_subtitle = all_persons_subtitle if isinstance(all_persons_subtitle, pd.DataFrame) \
            else pd.read_csv(all_persons_subtitle)
        self.all_persons_subtitle = remove_unnamed_0_col(self.all_persons_subtitle)

        self.pose_centroid_tracker = PoseCentroidTracker(all_videos, all_videos, openpose_path=self.extractor,
                                                         centroids_df_path='centroids.csv')

        self.needed_sings = needed_signs_list
        self.path_to_save_dfs = path_to_save_dfs

    def process(self):
        """
        Extrai os sinais de todos os videos.

        Returns
        -------

        """

        all_v_parts = self.vid_sync.v_part.unique().tolist()

        pbar_single_folder = tqdm(desc='single_folder', position=1, leave=False)
        pbar_for_video_extraction = tqdm(desc='all_folders', total=len(all_v_parts), position=0)

        for v_part in all_v_parts:

            pbar_for_video_extraction.update(1)

            if v_part == -1:
                continue

            try:
                sings_in_folder = self._process_single_folder(v_part, pbar=pbar_single_folder)
            except Exception as e:
                err_str = f'{e}'.encode("ascii", errors="ignore").decode()
                print(err_str)
                print(err_str, file=open('log.txt', mode='a'))
                continue
            # for sign_df in sings_in_folder:
            #     folder_path = os.path.join(self.path_to_save_dfs, sign_df['sign_name'])
            #     os.makedirs(folder_path, exist_ok=True)
            #     file_name = f'{v_part}---{sign_df["sign_name"]}---{sign_df["beg"]}---{sign_df["end"]}---' \
            #                 f'{sign_df["view"]}---{sign_df["talker_id"]}.csv'
            #     csv_file_path = os.path.join(folder_path, file_name)
            #
            #     sign_df['df'].to_csv(csv_file_path)

    def _process_single_folder(self, v_part: int, pbar: tqdm = None):
        """

        É iterado sobre a pasta extraindo todos os sinais e retorna uma lista com os CSVs de cada sinal extraido.

        Parameters
        ----------
        v_part

        Returns
        -------

        Lista contendo todos os sinais dentro do folder.

        """

        # aqui é encontrado o nome da pasta a partir do v_part dela no DF que contem, v_part, folder_name como chaves.
        curr_folder = self.all_persons_subtitle[self.all_persons_subtitle.v_part == v_part].folder_name
        if curr_folder.shape[0] == 0:
            return []

        curr_folder = curr_folder.values[0]
        signs_in_folder = self.all_videos[self.all_videos.folder == curr_folder]

        # encontrado os sinais que possuem dentro desse folder.

        if self.needed_sings is not None:
            signs_in_folder = signs_in_folder[signs_in_folder.sign.isin(self.needed_sings)]

        # id da pessoa a esquerda na legenda.
        lf_id_subtitle = self.all_persons_subtitle[self.all_persons_subtitle.v_part == v_part].left_person.values[0]

        if signs_in_folder.shape[0] == 0:
            return []

        if pbar is not None:
            pbar.reset(total=signs_in_folder.shape[0])
            pbar.set_description(f'single folder {v_part}')

        all_signs_pose_df = []
        count = 0
        for sign_row in signs_in_folder.iterrows():
            sign_row = sign_row[1]

            count += 1

            folder_path = os.path.join(self.path_to_save_dfs, sign_row.sign.replace("?", "").replace(" ", ""))
            front_file_name = f'{v_part}---{sign_row.sign.replace("?", "").replace(" ", "")}---{sign_row.beg}' \
                              f'---{sign_row.end}---front---{sign_row.talker_id}.csv'

            side_file_name = f'{v_part}---{sign_row.sign.replace("?", "").replace(" ", "")}---{sign_row.beg}' \
                             f'---{sign_row.end}---side---{sign_row.talker_id}.csv'

            side_file_name = side_file_name.replace('\\', '')
            side_file_name = side_file_name.replace('/', '')

            front_file_name = front_file_name.replace('\\', '')
            front_file_name = front_file_name.replace('/', '')

            front_file_path = os.path.join(folder_path, front_file_name)
            side_file_path = os.path.join(folder_path, side_file_name)
            # achar quem sinaliza, se ta no video 1 ou no 2 e extrair.
            is_left = sign_row.talker_id == lf_id_subtitle

            if pbar is not None:
                pbar.update(1)

            os.makedirs(folder_path, exist_ok=True)

            if not os.path.exists(front_file_path):
                curr_sign_pose_df = self.__extract_sings_from_single_person_video(
                    beg_msec=sign_row.beg,
                    end_msec=sign_row.end,
                    v_part=v_part,
                    is_left=is_left,
                    enable_debug=False,
                )

                if curr_sign_pose_df is not None:
                    curr_sign_pose_df.to_csv(front_file_path)
                    # all_signs_pose_df.append(dict(df=curr_sign_pose_df, sign_name=sign_row.sign, beg=sign_row.beg,
                    #                               end=sign_row.end, talker_id=sign_row.talker_id, view='side'))
            if not os.path.exists(side_file_path):
                curr_sign_pose_df = self.__extract_sign_from_video_with_2_person(
                    beg_msec=sign_row.beg,
                    end_msec=sign_row.end,
                    v_part=v_part,
                    is_left=is_left
                )

                if curr_sign_pose_df is not None:
                    curr_sign_pose_df.to_csv(side_file_path)
                    # all_signs_pose_df.append(dict(df=curr_sign_pose_df, sign_name=sign_row.sign, beg=sign_row.beg,
                    #                               end=sign_row.end, talker_id=sign_row.talker_id, view='front'))

        return all_signs_pose_df

    def __extract_sign_from_video_with_2_person(self, beg_msec: int, end_msec: int, v_part: int, is_left: bool):
        """

        Parameters
        ----------
        beg_msec: int
            Posição que começa o video em milisgundos.

        end_msec: int
            Posição que termina o video em milisegundos.

        v_part: int
          ID unico para cada folder que esta escrito no fim do nome dos folder.

        is_left: bool
            boleano indicando se a pessoa a ser extraida é a esquerda ou nao

        Returns
        -------
           pd.DataFrame

        """

        row_info = self.all_persons_subtitle[self.all_persons_subtitle.v_part == v_part]
        curr_id_in_subtitle = row_info.left_person.values[0] if is_left else row_info.right_person.values[0]

        vid_path, vid_name = self.__read_vid_path_from_vpart(v_part, 1)
        vid = cv.VideoCapture(vid_path)
        res = process_single_sample(extractor=self.extractor, curr_video=vid, beg=beg_msec, end=end_msec,
                                    person_needed_id=curr_id_in_subtitle,
                                    # pbar=tqdm(position=2),
                                    num_gpus=self.gpu_count,
                                    pose_tracker=self.pose_centroid_tracker)
        vid.release()
        return res

    def __extract_sings_from_single_person_video(self, beg_msec: int, end_msec: int, v_part: int, is_left: bool,
                                                 pbar: tqdm = None, enable_debug: bool = False, cv_debug: bool = True):

        """

        Parameters
        ----------
        beg_msec
        end_msec
        v_part
        is_left
        enable_debug

        Returns
        -------

        """

        def update_df_with_single_pose(dt_, curr_df, curr_person_, curr_msec):
            if enable_debug:
                if curr_person_ is not None:
                    cv.circle(dt.cvOutputData, center=tuple(map(int, curr_person_[1][:2])), radius=5, thickness=-1,
                              color=(255, 128, 0))

                if cv_debug:
                    cv.imshow('debug', dt.cvOutputData)
                    cv.waitKey(1)
                else:
                    plt.imshow(dt.cvOutputData[:, :, :: -1])
                    plt.show()

            curr_person_, person_id_ = ExtractMultipleVideos.get_person_sorted_left_2_right(dt_, is_left, is_body=True)
            curr_person_hand_, person_id_hand_ = ExtractMultipleVideos.get_person_sorted_left_2_right(dt_, is_left,
                                                                                                      is_body=False)

            v_df = update_xy_pose_df_single_person(dt_, curr_df,
                                                   int(curr_msec),
                                                   person_id_,
                                                   person_id_hand_,
                                                   all_parts.BODY_PARTS_NAMES,
                                                   all_parts.HAND_PARTS_NAMES)

            if enable_debug and cv_debug:
                cv.destroyAllWindows()

            return v_df

        # Encontrando qual é a pasta/projeto/item que deve ser lido.
        row_vid_sync = self.vid_sync[self.vid_sync.v_part == v_part]
        if row_vid_sync.left_id.values[0] == -1 or row_vid_sync.right_id.values[0] == -1:
            return None

        curr_vid_num = row_vid_sync.left_id.values[0] if is_left else row_vid_sync.right_id.values[0]
        curr_vid_num += 1

        vid_path, vid_name = self.__read_vid_path_from_vpart(v_part, curr_vid_num)
        vid = cv.VideoCapture(vid_path)

        # opencv/ffmeg tem um erro ao setar o video por milisegundo, para burlar esse erro é necessario ler um frame do
        # video.
        vid.set(cv.CAP_PROP_POS_MSEC, beg_msec)
        ret, frame = vid.read()
        if not ret:
            return None

        dt: DatumLike = self.extractor.extract_poses(frame)

        if enable_debug:
            if cv_debug:
                cv.imshow('debug', dt.cvOutputData)
                cv.waitKey(1)
            else:
                plt.imshow(dt.cvOutputData[:, :, :: -1])
                plt.show()

        df_cols = ['frame'] + all_parts.BODY_PARTS_NAMES + \
                  ['left-' + x for x in all_parts.HAND_PARTS_NAMES] + \
                  ['right-' + x for x in all_parts.HAND_PARTS_NAMES]

        video_df = pd.DataFrame(columns=df_cols)

        curr_msec_pos = vid.get(cv.CAP_PROP_POS_MSEC)
        curr_person, person_id = ExtractMultipleVideos.get_person_sorted_left_2_right(dt, is_left, is_body=True)
        curr_person_hand, person_id_hand = ExtractMultipleVideos.get_person_sorted_left_2_right(dt, is_left,
                                                                                                is_body=False)

        video_df = update_xy_pose_df_single_person(dt, video_df,
                                                   int(curr_msec_pos),
                                                   person_id,
                                                   person_id_hand,
                                                   all_parts.BODY_PARTS_NAMES,
                                                   all_parts.HAND_PARTS_NAMES)

        if pbar is not None:
            pbar.reset(total=end_msec - beg_msec)
            pbar.update(beg_msec - curr_msec_pos)

        last_msec_pos = curr_msec_pos
        while curr_msec_pos <= end_msec:

            if self.gpu_count > 0:
                frames = []
                msecs = []
                for _ in range(self.gpu_count):
                    ret, frame = vid.read()
                    curr_msec_pos = vid.get(cv.CAP_PROP_POS_MSEC)

                    if curr_msec_pos >= end_msec:
                        break

                    if not ret and len(frames) == 0:
                        return None

                    msecs.append(curr_msec_pos)
                    frames.append(frame)
                poses = self.extractor.extract_multiple_gpus(frames)
                for p, msec in zip(poses, msecs):
                    video_df = update_df_with_single_pose(p, video_df, curr_person, msec)

            else:
                ret, frame = vid.read()
                curr_msec_pos = vid.get(cv.CAP_PROP_POS_MSEC)
                if not ret:
                    return None
                dt: DatumLike = self.extractor.extract_poses(frame)
                video_df = update_df_with_single_pose(dt, video_df, curr_person, curr_msec_pos)

            curr_msec_pos = vid.get(cv.CAP_PROP_POS_MSEC)

            if pbar is not None:
                pbar.update(int(curr_msec_pos - last_msec_pos))

            last_msec_pos = curr_msec_pos

        vid.release()
        return video_df

    def __read_vid_path_from_vpart(self, v_part: int, vid_number: int):
        """

        Parameters
        ----------
        vid_number

        v_part: int
          ID unico para cada folder que esta escrito no fim do nome dos folder.

        is_left: bool
            boleano indicando se a pessoa a ser extraida é a esquerda ou nao

        Returns
        -------

        """
        row_info = self.all_persons_subtitle[self.all_persons_subtitle.v_part == v_part]

        folder_complete_path = os.path.join(self.db_path, row_info.folder_name.values[0])

        # ordeno o nome dos videos dentro das pastas, note que cada pasta os videos tem nomes como algo_cam1.mp4,
        # algo_cam2.mp4, e respectivamente o cam1 é o video com 2 pessoas vista lateral, os cam3 e cam2 são
        # rescpetivamente os video com vista de uma pessoa inclinado, e cam4 o vista superior com duas pessoas.
        vid_path, vid1_name = ExtractMultipleVideos.read_vid_path_from_vpart(folder_complete_path, vid_number)

        return vid_path, vid1_name

    def __submit_work_to_multi_gpu(self):
        pass

    @staticmethod
    def read_vid_path_from_vpart(folder_complete_path, vid_number):
        vid1_name = list(filter(lambda x: '.mp4' in x, os.listdir(folder_complete_path)))
        vid1_name = sorted(vid1_name, key=lambda x: int(x.split('.mp4')[0][-1]))

        if vid1_name[vid_number - 1][-5] != str(vid_number):
            return None, None
        vid1_name = vid1_name[vid_number - 1]
        vid_path = os.path.join(folder_complete_path, vid1_name)

        return vid_path, vid1_name

    def unittest_for___extract_sign_from_video_with_2_person(self):
        try:
            df_p1_side = self.__extract_sign_from_video_with_2_person(5000, 5000 + (1000 * 60), 1098, is_left=True)
            df_p1_side.to_csv('v1098_side_view_left_person.csv')

            df_p2_side = self.__extract_sign_from_video_with_2_person(5000, 5000 + (1000 * 60), 1098, is_left=False)
            df_p2_side.to_csv('v1098_side_view_right_person.csv')

        except BaseException:
            assert False

    def unittest_for___extract_signs_from_single_person_video(self):

        try:
            res1 = self.__extract_sings_from_single_person_video(5000, 5000 + (1000 * 60), 1098, is_left=True,
                                                                 pbar=tqdm())
            res2 = self.__extract_sings_from_single_person_video(5000, 5000 + (1000 * 60), 1098, is_left=False,
                                                                 pbar=tqdm())
            res1.to_csv('v1098_barely_front_view_left_person.csv')
            res2.to_csv('v1098_barely_front_view_right_person.csv')
        except BaseException:
            assert False

    @staticmethod
    def get_person_sorted_left_2_right(dt: DatumLike, is_left: bool, is_body: bool):
        """

        Parameters
        ----------
        dt
        is_left
        is_body

        Returns
        -------

        """

        def remove_zero_poses(poses):
            new_poses = []
            for p in poses:
                if p[0] != 0 and p[1] != 0 and p[2] != 0:
                    new_poses.append(p)

            return new_poses

        def clean_poses(pose_list):
            valid_poses = []
            for it in range(len(pose_list)):
                new_pose = remove_zero_poses(curr_pose[it])
                if len(new_pose) > 0:
                    valid_poses.append(new_pose)

            return valid_poses

        curr_left, curr_right, left_id, right_id, p1, p2 = None, None, None, None, None, None

        curr_pose = dt.poseKeypoints if is_body else dt.handKeypoints[0]
        curr_pose = curr_pose.tolist()
        valid_poses = clean_poses(curr_pose)

        if len(valid_poses) == 0 and not is_body:
            curr_pose = dt.handKeypoints[1].tolist()
            valid_poses = clean_poses(curr_pose)

        curr_pose = valid_poses

        if len(curr_pose) == 0:
            return None, None

        if len(curr_pose) > 1:
            p1 = ExtractMultipleVideos.bbox_from_pose(curr_pose[0])
            p2 = ExtractMultipleVideos.bbox_from_pose(curr_pose[1])
        else:
            p1 = ExtractMultipleVideos.bbox_from_pose(curr_pose[0])

        if is_left:
            if len(curr_pose) > 1:
                curr_left = p1 if p1[1][0] < p2[1][0] else p2
                left_id = 0 if p1[1][0] < p2[1][0] else 1
            elif len(curr_pose) == 1:
                curr_left = p1
                left_id = 0
            else:
                raise RuntimeError("more person than expected")
        else:
            if len(curr_pose) > 1:
                curr_right = p1 if p1[1][0] > p2[1][0] else p2
                right_id = 0 if p1[1][0] > p2[1][0] else 1
            elif len(curr_pose) == 1:
                curr_right = p1
                right_id = 0
            else:
                raise RuntimeError("more person than expected")

        result = None, None
        if is_left:
            result = curr_left, left_id
        else:
            result = curr_right, right_id
        return result

    @staticmethod
    def bbox_from_pose(pose):
        """

        Parameters
        ----------
        pose

        Returns
        -------

        """

        np.seterr(divide='raise', invalid='ignore')
        centroid = np.array([0.0, 0.0])
        part_count = 0
        y_max, x_max = 0, 0
        y_min, x_min = 999999, 99999

        for body_part in pose:
            if body_part[0] != 0 and body_part[0] is not None:
                if x_min > body_part[0]:
                    x_min = body_part[0]

                if y_min > body_part[1]:
                    y_min = body_part[1]

                if x_max < body_part[0]:
                    x_max = body_part[0]

                if y_max < body_part[1]:
                    y_max = body_part[1]

                centroid += body_part[:2]
                part_count += 1

        centroid = centroid / part_count if centroid[0] != 0 and centroid[1] != 0 and part_count != 0 else None

        return centroid, (x_min, y_min, x_max, y_max)


if __name__ == '__main__':
    #init()
    sign_list = [
        # 'NÃO', 'TER', 'BOM', 'E(esperar)','COMO', 'E(acabar)', 'VER', 'HOMEM', 'PORQUE', 'ESTUDAR'
        'VER'
        # 'HOMEM'
    ]

    new_sign_list = [
        'TRABALHAR',
        # 'SABER',
        # 'CERTO',
        # 'OUVIR',
        # 'MÃE',
    ]
    extractMultipleVideos = ExtractMultipleVideos(db_path='D:/libras corpus',
                                                  all_videos='all_videos3.csv',
                                                  openpose_path='C:/Users/lucas/Documents/Projects/Libras/PoseExtractors/openpose',
                                                  vid_sync='vid_sync.csv',
                                                  # path_to_save_dfs='../sign_db_front_view',
                                                  path_to_save_dfs='D:/sign_db_front_view',
                                                  # needed_signs_list=new_sign_list,
                                                  all_persons_subtitle='all_persons_from_subtitle.csv')

    try:
        extractMultipleVideos.process()
    except:
        print('LOOOL')
    # extractMultipleVideos.unittest_for___extract_sign_from_video_with_2_person()
    # extractMultipleVideos.process()
    # 217 sinais da palavra homem extraidos.
