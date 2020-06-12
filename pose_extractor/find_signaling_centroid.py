import pandas as pd
import numpy as np
import itertools as itools

class FindSignalingCentroid:

    def __init__(self, all_videos_csv_path):
        self._df = pd.read_csv(all_videos_csv_path, index_col=0)

    def process_all(self):
        pass

    def find_each_signaler_frame_talks_alone(self, folder_name):
        persons = self._df[self._df['folder_name'] == folder_name]['talker_id']\
            .unique()

        end_video_frame_value = 0
        beg_video_frame_value = 999999
        for p in persons:
            end_talks = self._df.loc[(self._df['folder_name'] == folder_name) &
                                     (self._df['talker_id'] == p)].end.max()
            if end_video_frame_value < end_talks:
                end_video_frame_value = end_talks

            beg_talks = self._df.loc[(self._df['folder_name'] == folder_name) &
                                     (self._df['talker_id'] == p)].beg.min()
            if beg_video_frame_value > beg_talks:
                beg_video_frame_value = beg_talks

        talking_frames = [np.zeros((end_video_frame_value + 1, ))
                          for _ in persons]
        for it, p in enumerate(persons):
            end_talks = self._df.loc[(self._df['folder_name'] == folder_name) &
                                     (self._df['talker_id'] == p)].end
            beg_talks = self._df.loc[(self._df['folder_name'] == folder_name) &
                                     (self._df['talker_id'] == p)].beg

            for beg, end in zip(beg_talks, end_talks):
                talking_frames[it][beg].fill(1)

        sum_talkers = talking_frames[0]
        for tk in talking_frames[1:]:
            sum_talkers = np.add(sum_talkers, tk)

        where_persons_talks_alone = np.where(sum_talkers == 1)
        if len(where_persons_talks_alone) % 2 != 0:
            where_persons_talks_alone.append(-1)

        for it, p in enumerate(persons):
            is_p_talking = False
            for frame_pos in where_persons_talks_alone:
                # pergunta se p ta no frame_pos se n da continue ate parar de
                # iterar de 1 em 1 no frame_pos, pq isso indica que ja achou
                # outra pessoa falando.

        print(end_video_frame_value, beg_video_frame_value)



    @staticmethod
    def __make_centroid_from_sing(self):
        pass
