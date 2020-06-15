import pandas as pd
import numpy as np
from pose_extractor.pose_centroid_tracker import PoseCentroidTracker


class FindSignalingCentroid:

    def __init__(self, all_videos_csv_path):
        self._df = pd.read_csv(all_videos_csv_path, index_col=0)

    @staticmethod
    def try_import_openpose(path_to_openpose):
        pass

    def process_all(self):
        pass

    def find_each_signaler_frame_talks_alone(self, folder):
        persons = self._df[self._df['folder_name'] == folder]['talker_id']\
            .unique()

        end_video_frame_value = 0
        beg_video_frame_value = 999999
        for p in persons:
            end_talks = self._df.loc[(self._df['folder_name'] == folder) &
                                     (self._df['talker_id'] == p)].end.max()
            if end_video_frame_value < end_talks:
                end_video_frame_value = end_talks

            beg_talks = self._df.loc[(self._df['folder_name'] == folder) &
                                     (self._df['talker_id'] == p)].beg.min()
            if beg_video_frame_value > beg_talks:
                beg_video_frame_value = beg_talks

        end_video_frame_value = end_video_frame_value \
            if end_video_frame_value % 2 != 0 else end_video_frame_value + 1
        talking_frames = [np.zeros((end_video_frame_value + 1, ))
                          for _ in persons]

        for it, p in enumerate(persons):
            end_talks = self._df.loc[(self._df['folder_name'] == folder) &
                                     (self._df['talker_id'] == p)].end
            beg_talks = self._df.loc[(self._df['folder_name'] == folder) &
                                     (self._df['talker_id'] == p)].beg

            for beg, end in zip(beg_talks, end_talks):
                talking_frames[it][beg:end].fill(1)

        sum_talkers = talking_frames[0]
        for tk in talking_frames[1:]:
            sum_talkers = np.add(sum_talkers, tk)

        where_persons_talks_alone = np.where(sum_talkers == 1)
        where_persons_talks_alone = where_persons_talks_alone[0]
        persons_alone = {}
        for it, p in enumerate(persons):
            last = 0
            beg = 0
            res = None
            for frame_pos in where_persons_talks_alone:
                if res is None:
                    res = self._df.loc[(self._df['folder_name'] == folder) &
                                       (self._df['talker_id'] == p) &
                                       (self._df['beg'] == frame_pos)]
                    if res.shape[0] > 0:
                        beg = frame_pos

                elif res.shape[0] > 0 and frame_pos - last > 1:
                    persons_alone.update({str(p): {'beg': beg, 'end': last}})
                    break
                elif res.shape[0] == 0:
                    res = None

                last = frame_pos

        return persons_alone



    @staticmethod
    def __make_centroid_from_sing(self):
        pass
