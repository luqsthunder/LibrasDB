import os
import face_recognition
import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm


class SyncSubtitles2Videos:

    def __init__(self, all_videos, db_path):
        self,db_path = db_path
        self.all_videos = all_videos if isinstance(all_videos, pd.DataFrame) else pd.read_csv(all_videos)
        self.folders = all_videos.folder_name.unique().tolist()

    def find_all_left_signaler(self):
        """
        Nessa função é para encontrar em cada folder da base de dados quem é cada pessoa da legenda.

        Levando em consideração que para cada folder da base de dados há duas pessoas por video, encontramos na
        legenda respectiva ao folder, utilizando  o primeiro video do folder (video que contém as duas pessoas)
        quem qual pessoa corresponde ao marcador/id na legenda.

        Returns
        -------
        pd.Dataframe
            Um pd.dataframe com as informações indicando qual pessoa é a esquerda e a direita no folder de acordo com
            a legenda.
        """

        all_persons_df = pd.DataFrame()
        for k, f in tqdm(enumerate(self.folders[8:10])):
            f_path = os.path.join(self.db_path, f)
            vid = cv.VideoCapture(f_path)

            p1_interruptible_speach = False
            p2_interruptible_speach = False

            curr_signer_num = 1
            signer_holes = self.__find_where_subtitle_has_holes(f, 1)
            if signer_holes is None:
                curr_signer_num = 2
                signer_holes = self.__find_where_subtitle_has_holes(f, 2)
            elif len(signer_holes) == 0:
                p1_interruptible_speach = True
                curr_signer_num = 2
                signer_holes = self.__find_where_subtitle_has_holes(f, 2)

            if signer_holes is None:
                continue

            if len(signer_holes) == 0:
                p2_interruptible_speach = True
                if not (bool(p1_interruptible_speach) != bool(p1_interruptible_speach)):
                    continue

                curr_signer_num = 1 if p1_interruptible_speach and not p2_interruptible_speach else 2
                curr_beg_pos = self.all_videos[self.all_videos.folder_name == f]
                curr_beg_pos = curr_beg_pos[curr_beg_pos.talker_id == curr_signer_num].beg.iloc[0]
                signer_holes = [dict(beg=curr_beg_pos, end=curr_beg_pos + 5000)]

            res = self.__find_left_signaling_in_single_video(vid, signer_holes)
            left_id, right_id = self.__from_left_signaling_result(res, curr_signer_num)
            print(res, left_id, right_id)
            # all_persons_df = all_persons_df.append(pd.DataFrame(dict(
            #     folder_name=[f], person_1=[left_id], person_2=[right_id]
            # )))
            vid.release()

    @staticmethod
    def __from_left_signaling_result(result, talker_id):
        left_id = 0
        right_id = 0
        if result['left'] > result['right']:
            left_id = talker_id
        elif result['left'] < result['right']:
            right_id = talker_id
        else:
            print('ERROR in comparing left signalers')

        if left_id == 0:
            right_id = 1 if left_id == 2 else 2
        elif right_id == 0:
            left_id = 1 if right_id == 2 else 2

        return left_id, right_id

    @staticmethod
    def __find_middle_xpoint_in_faces_bbox(frame):
        faces = face_recognition.face_locations(frame, number_of_times_to_upsample=1, model='cnn')
        middle_xpoint = (np.abs(faces[0][3] - faces[1][3]) / 2) + min([faces[0][3], faces[1][3]])
        return middle_xpoint

    @staticmethod
    def _make_talkers_motion_check(vid, end_frame_time, x_middle):
        last_frame = None

        ret, frame = vid.read()
        if not ret:
            return []

        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        curr_video_pos_msec = vid.get(cv.CAP_PROP_POS_MSEC)

        if x_middle is None:
            x_middle = SyncSubtitles2Videos.__find_middle_xpoint_in_faces_bbox(frame)

        left_most_white = []
        while curr_video_pos_msec <= end_frame_time:
            (thresh, frame) = cv.threshold(frame, 127, 255, cv.THRESH_BINARY)
            curr_frame = frame - last_frame if last_frame is not None else frame
            last_frame = frame
            (thresh, curr_frame) = cv.threshold(curr_frame, 127, 255, cv.THRESH_BINARY)

            left = curr_frame[:, int(x_middle):]
            right = curr_frame[:, :int(x_middle)]

            left_most_white.append(np.count_nonzero(left > 1) < np.count_nonzero(right > 1))

            ret, frame = vid.read()
            if not ret:
                print(curr_video_pos_msec)
                break
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            curr_video_pos_msec = vid.get(cv.CAP_PROP_POS_MSEC)

        return left_most_white, x_middle

    def __find_left_signaling_in_single_video(self, vid, holes):
        """

        Parameters
        ----------

        holes: List

        """
        all_left_whites_count = []
        all_left_in_video = {'Left': 0, 'Right': 0}
        x_middle = None

        for c_hole_pos in range(len(holes)):
            fps = int(vid.get(cv.CAP_PROP_FPS))
            beg_frame_time = holes[c_hole_pos]['beg']
            res = vid.set(cv.CAP_PROP_POS_MSEC, beg_frame_time)
            if not res:
                raise RuntimeError('Can not set position in this video')

            end_frame_time = holes[c_hole_pos]['end']

            left_most_white, x_middle = self._make_talkers_motion_check(vid, end_frame_time, x_middle)

            unique, counts = np.unique(np.array(left_most_white), return_counts=True)
            white_count = dict(zip(unique, counts))
            all_left_in_video['Left'] += white_count[True] if True in white_count else 0

            all_left_in_video['Right'] += white_count[False] if False in white_count else 0

            all_left_whites_count.append(white_count)

        return all_left_in_video

    def __find_where_subtitle_has_holes(self, folder_name, talker_id, pbar=None):
        """
        Acha onde um sinalizador de uma legenda na pasta do projeto não fala nada.

        parameters
        ----------
        folder_name: str
            nome da pasta na base de dados.

        talker_id: int
            id do sinalizador na legenda

        returns: List
            holes, lista com os locais onde o sinalizador não fala.
        """
        talker_1_signs = self.all_video_df[self.all_video_df.folder_name == folder_name]
        talker_1_signs = talker_1_signs[talker_1_signs.talker_id == talker_id]
        talker_1_signs = talker_1_signs[talker_1_signs.hand == 'D']

        if talker_1_signs.shape[0] == 0:
            return None

        talker_1_times = [(x[1].beg, x[1].end)
                          for x in talker_1_signs.iterrows() if x[1].talker_id == talker_id]
        talker_1_times = sorted(talker_1_times, key=lambda x: x[0])

        last_beg = talker_1_times[0][0]
        times_talking = []
        last_end = talker_1_times[0][1]
        holes = []

        if pbar is not None:
            pbar.reset(total=len(talker_1_times))

        for time_talk in talker_1_times[1:]:
            if last_end >= time_talk[0]:
                last_end = time_talk[1] if last_end < time_talk[1] else last_end
            else:
                times_talking.append(dict(beg=last_beg, end=last_end))
                last_end = time_talk[1]
                last_beg = time_talk[0]
            if pbar is not None:
                pbar.update(1)
                pbar.refresh()

        # for x in times_talking:
        #     print(f"beg: {mili_to_minutes(x['beg'])}, end: {mili_to_minutes(x['end'])}")

        for time_it in range(1, len(times_talking)):
            try:
                hole = times_talking[time_it]['beg'] - times_talking[time_it - 1]['end']
                holes.append(dict(beg=times_talking[time_it - 1]['end'],
                                  end=times_talking[time_it]['beg'],
                                  hole=hole))
            except IndexError:
                continue

        holes = list(filter(lambda x: x['hole'] > 1000, holes))

        return holes

    def _unittest_find_where_subtitle_has_holes(self):
        pass

    def _unittest_find_left_signaling_in_single_video(self):
        pass
