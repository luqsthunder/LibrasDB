from pose_extractor.openpose_extractor import OpenposeExtractor, DatumLike
from pose_extractor.extract_signs_from_video import process_single_sample
from pose_extractor.pose_centroid_tracker import PoseCentroidTracker

import numpy as np
import os
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm


class ExtractMultipleVideos:

    def __init__(self, db_path: str, all_videos: pd.DataFrame or str, vid_sync: str or pd.DataFrame,
                 all_persons_subtitle: str or pd.DataFrame, openpose_path: str):
        """

        Parameters
        ----------
        db_path
        all_videos
        """

        def remove_unnamed_0_col(df):
            if 'Unnamed: 0' in df.keys():
                return df.drop(columns=['Unnamed: 0'])
            return  df

        self.db_path = db_path
        self.all_videos = all_videos if isinstance(all_videos, pd.DataFrame) else pd.read_csv(all_videos)
        self.all_videos = remove_unnamed_0_col(self.all_videos)

        self.extractor = OpenposeExtractor(openpose_path=openpose_path)

        self.vid_sync = vid_sync if isinstance(vid_sync, pd.DataFrame) else pd.read_csv(vid_sync)
        self.vid_sync = remove_unnamed_0_col(self.vid_sync)

        self.all_persons_subtitle = all_persons_subtitle if isinstance(all_persons_subtitle, pd.DataFrame) \
                                                         else pd.read_csv(all_persons_subtitle)
        self.all_persons_subtitle = remove_unnamed_0_col(self.all_persons_subtitle)

        self.pose_tracker = PoseCentroidTracker(all_videos, all_videos, openpose_path=self.extractor,
                                                centroids_df_path='centroids.csv')

    def process(self):
        pass

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
                                    person_needed_id=curr_id_in_subtitle, pbar=tqdm(), pose_tracker=self.pose_tracker)
        vid.release()
        return res

    def __extract_sings_from_sinlge_person_video(self, beg_msec: int, end_msec: int, v_part: int, is_left: bool):

        row_vid_sync = self.vid_sync[self.vid_sync.v_part == v_part]
        curr_vid_num = row_vid_sync.left_id.values[0] if is_left else row_vid_sync.right_id.values[0]
        curr_vid_num += 1

        vid_path, vid_name = self.__read_vid_path_from_vpart(v_part, curr_vid_num)
        vid = cv.VideoCapture(vid_path)
        vid.set(cv.CAP_PROP_POS_MSEC, beg_msec)

        ret, frame = vid.read()

        if not ret:
            return

        dt: DatumLike = self.extractor.extract_poses(frame)
        plt.imshow(dt.cvOutputData)
        plt.show()

        while vid.get(cv.CAP_PROP_POS_MSEC) <= end_msec:
            ret, frame = vid.read()
            if not ret:
                return

            dt: DatumLike = self.extractor.extract_poses(frame)

            if len(dt.poseKeypoints) > 1:
                p1 = self.bbox_from_pose(dt, 0)
                p2 = self.bbox_from_pose(dt, 1)
            else:
                p1 = self.bbox_from_pose(dt, 0)

            if len(dt.poseKeypoints) > 1:
                curr_left = p1 if p1[1][0] < p2[1][0] else p2
            elif len(dt.poseKeypoints) == 1:
                curr_left = p1
            else:
                raise RuntimeError("more person than expected")

            cv.circle(frame, center=curr_left[1][:2], radius=5, thickness=-1, color=(255, 128, 0))

            plt.imshow(frame)
            plt.show()

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
            df_p1_side.to_csv('v0180_side_view_left_person.csv')

            df_p2_side = self.__extract_sign_from_video_with_2_person(5000, 5000 + (1000 * 60), 1098, is_left=False)
            df_p2_side.to_csv('v0180_side_view_right_person.csv')

        except BaseException:
            assert False

    def unittest_for___extract_signs_from_single_person_video(self):
        self.__extract_sings_from_sinlge_person_video(5000, 5000 + (1000), 1098, is_left=False)

    @staticmethod
    def bbox_from_pose(dt: DatumLike, person_pos: int):
        centroid = np.array([0.0, 0.0])
        part_count = 0
        y_max, x_max = 0, 0
        y_min, x_min = 999999, 99999
        for body_part in dt.poseKeypoints[person_pos]:
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
        centroid = centroid / part_count
        return centroid, (x_min, y_min, x_max, y_max)




if __name__ == '__main__':
    extractMultipleVideos = ExtractMultipleVideos(db_path='D:/gdrive',
                                                  all_videos='all_videos.csv',
                                                  openpose_path='C:/Users/usuario/Documents/Libraries/repos/openpose',
                                                  vid_sync='vid_sync.csv',
                                                  all_persons_subtitle='all_persons_from_subtitle.csv')
    extractMultipleVideos.unittest_for___extract_signs_from_single_person_video()
