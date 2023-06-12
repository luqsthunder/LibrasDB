import sys

sys.path.append("/home/usuario/Projects/Libras/LibrasDB")
from libras_classifiers.librasdb_loaders import DBLoader2NPY
import pandas as pd
import numpy as np
import cv2 as cv
import random


class AutoCalibrateFromSkell:
    """
    Classe para encontrar calibração automatica das cameras usadas em todos os videos.

    Aqui é construido a homografia e utilizando do autocalibration do opencv achamos os parametros relacionados a
    calibração.
    """

    def __init__(self, skell_side_view, skell_front_view, front_video, side_video):
        pass

    def _get_homography_from_skell(
        self, side_skell_df: pd.DataFrame, front_skell_df: pd.DataFrame
    ):
        """

        Parameters
        ----------
        side_skell_df
        front_skell_df

        Returns
        -------

        """
        matches = self.filter_joints_by_threshold(side_skell_df, front_skell_df, 0.6)
        matches_front = np.array([x[0] for x in matches]).reshape(-1, 1, 2)
        matches_side = np.array([x[1] for x in matches]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(matches_side, matches_front, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        f0, f1, f0_ok, f1_ok = None, None, None, None
        cv.detail.focalsFromHomography(M, f0, f1, f0_ok, f1_ok)
        print(f0, f1, f0_ok, f1_ok, M)

    @staticmethod
    def _get_homography_from_frames(video_left_frames: list, video_right_frames: list):
        """
        """
        pass

    def filter_joints_by_threshold(self, side_skell_df, front_skell_df, threshold):
        matches = []

        all_frames = side_skell_df.frame.unique().tolist()
        for i in range(0, len(all_frames), 5):
            i += random.randrange(-2, 2)
            row_front = front_skell_df[front_skell_df["frame"] == all_frames[i]]
            row_front = row_front.drop(columns=["frame", "Unnamed: 0"])
            row_front = row_front.applymap(
                lambda x: np.array([0.0, 0.0, 0.0]) if isinstance(x, float) else x
            )
            row_front = row_front.values[0]
            row_front = [x if x[2] >= threshold else None for x in row_front]

            row_side = side_skell_df[side_skell_df["frame"] == all_frames[i]]
            row_side = row_side.drop(columns=["frame", "Unnamed: 0"])
            row_side = row_side.applymap(
                lambda x: np.array([0.0, 0.0, 0.0]) if isinstance(x, float) else x
            )
            row_side = row_side.values[0]
            row_side = [x if x[2] >= threshold else None for x in row_side]

            for front, side in zip(row_front, row_side):
                if front is not None and side is not None:
                    matches.append((front[:2], side[:2]))

        return matches

    def process_single_video(self):
        person = "left"
        front_data_path = "v1098_barely_front_view_" + person + "_person.csv"
        side_data_path = "v0180_side_view_" + person + "_person.csv"

        left_front_data = pd.read_csv(front_data_path)
        left_side_data = pd.read_csv(side_data_path)

        left_front_data = left_front_data.applymap(DBLoader2NPY.parse_npy_vec_str)
        left_side_data = left_side_data.applymap(DBLoader2NPY.parse_npy_vec_str)

        left_side_data.frame = left_front_data.frame
        self._get_homography_from_skell(left_side_data, left_front_data)


AutoCalibrateFromSkell(None, None, None, None).process_single_video()
