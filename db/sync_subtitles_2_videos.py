import os
import face_recognition
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier as KNN


class SyncSubtitles2Videos:

    def __init__(self, all_videos, db_path):
        self.db_path = db_path
        self.all_videos = all_videos if isinstance(all_videos, pd.DataFrame) else pd.read_csv(all_videos)
        self.folders = self.all_videos.folder.unique().tolist()
        self.db_folders_path = [os.path.join(self.db_path, x) for x in self.folders]
        self.db_folders_path = sorted(self.db_folders_path, key=lambda x: int(x.split(' v')[-1]), reverse=True)
        self.font = cv.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def __find_video_in_folder_by_num(folder_path: str, num: int):
        videos_path = list(filter(lambda x: '.mp4' in x, os.listdir(folder_path)))
        for video in videos_path:
            try:
                num_in_video = int(video.split('.mp4')[0][-1])
                if num_in_video == num:
                    return video
            except BaseException:
                continue

        return None

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
        for k, f in tqdm(enumerate(self.folders), total=len(self.folders)):
            f_path = os.path.join(self.db_path, f)
            v_part = f.split(' v')[-1]

            video_path = self.__find_video_in_folder_by_num(f_path, 1)
            video_path = os.path.join(f_path, video_path)

            vid = cv.VideoCapture(video_path)


            signaling_1_quiet_moments = self.__find_where_signaling_is_quiet(f, 1)
            signaling_2_quiet_moments = self.__find_where_signaling_is_quiet(f, 2)

            if signaling_1_quiet_moments is None and signaling_2_quiet_moments is None:
                continue

            if signaling_1_quiet_moments is not None and signaling_2_quiet_moments is not None:
                if len(signaling_1_quiet_moments) == 0 and len(signaling_2_quiet_moments) == 0:
                    continue

            curr_signaler_quiet = 1
            if signaling_1_quiet_moments is None:
                curr_signaler_quiet = 1
                curr_beg_pos = self.all_videos[self.all_videos.folder == f].beg.min()
                signaling_1_quiet_moments = [dict(beg=curr_beg_pos, end=curr_beg_pos + 5000)]

            elif signaling_2_quiet_moments is None:
                curr_signaler_quiet = 2
                curr_beg_pos = self.all_videos[self.all_videos.folder == f].beg.min()
                signaling_2_quiet_moments = [dict(beg=curr_beg_pos, end=curr_beg_pos + 5000)]


            quiet_to_send = signaling_1_quiet_moments if curr_signaler_quiet == 1 else signaling_2_quiet_moments
            res = self.__find_left_signaling_in_single_video(vid, quiet_to_send, debug_frame=False)
            if res['Left'] == res['Right']:
                continue

            left_id, right_id = self.__from_left_signaling_result(res, curr_signaler_quiet)
            all_persons_df = all_persons_df.append(pd.DataFrame(dict(
                folder_name=[f], left_person=[left_id], right_person=[right_id],
                v_part=[v_part]
            )))
            vid.release()
        all_persons_df.to_csv('all_persons_from_subtitle.csv')
        return all_persons_df

    def process_all_folders_2_sync_with_person_2_video(self):
        if not os.path.exists('vid_sync.csv'):
            pd.DataFrame(dict(v_part=[-1], left_id=[-1], right_id=[-1])).to_csv('vid_sync.csv', mode='w')
        for db_folder in tqdm(self.db_folders_path):
            v_part = db_folder.split(' v')[-1]
            fig_name_path = f'../fig-folder-libras/{v_part}.pdf'
            if os.path.exists(fig_name_path):
                continue

            all_videos_in_folder = list(filter(lambda x: '.mp4' in x, os.listdir(db_folder)))
            vid_numbers = []
            for v in all_videos_in_folder:
                try:
                    vid_num = int(v.split('.mp4')[0][-1])
                    vid_numbers.append(vid_num)
                except BaseException:
                    continue

            if not (1 in vid_numbers and 2 in vid_numbers and 3 in vid_numbers):
                continue

            def conv_2_int_handling_except(str_num):
                try:
                    a = int(str_num)
                    return a
                except ValueError:
                    return 999999

            all_videos_in_folder = sorted(all_videos_in_folder,
                                          key=lambda x: conv_2_int_handling_except(x.split('.mp4')[0][-1]))
            if len(all_videos_in_folder) < 3:
                for it in range(len(all_videos_in_folder)):
                    all_videos_in_folder[it].release()

                continue

            all_videos_in_folder = [cv.VideoCapture(os.path.join(db_folder, x)) for x in all_videos_in_folder]

            bar_occurrences = self.db_path.replace('\\', '/').count('/') - 1
            folder_name = os.path.join(*(db_folder.replace('\\', '/').split('/')[bar_occurrences:]))
            first_sign_beg = self.all_videos[self.all_videos['folder'] == folder_name].beg
            if first_sign_beg.shape[0] == 0:
                continue

            first_sign_beg = int(first_sign_beg.min())
            first_sign_beg = first_sign_beg / (1000 / 30)
            for it in range(len(all_videos_in_folder)):
                all_videos_in_folder[it].set(cv.CAP_PROP_POS_FRAMES, first_sign_beg)

            encodes, p_ids = self._find_faces_ids_and_embedings_sorted_left_2_right(all_videos_in_folder,
                                                                                    show_video1_frame=True,
                                                                                    v_part=v_part)

            if len(p_ids) == 0:
                p_ids = [-2, -2]

            pd.DataFrame(dict(v_part=[v_part], left_id=[p_ids[0]], right_id=[p_ids[1]]))\
              .to_csv('vid_sync.csv', mode='a', index=False, header=False)

            for it in range(len(all_videos_in_folder)):
                all_videos_in_folder[it].release()

    @staticmethod
    def __from_left_signaling_result(result, talker_id):
        left_id = 0
        right_id = 0
        if result['Left'] < result['Right']:
            left_id = talker_id
        elif result['Right'] < result['Left']:
            right_id = talker_id
        else:
            print('ERROR in comparing left signalers')

        if left_id == 0:
            left_id = 1 if right_id == 2 else 2
        elif right_id == 0:
            right_id = 1 if left_id == 2 else 2

        return left_id, right_id

    @staticmethod
    def __find_middle_xpoint_in_faces_bbox(frame):
        faces = face_recognition.face_locations(frame, number_of_times_to_upsample=1, model='cnn')
        if len(faces) < 2:
            return None

        middle_xpoint = (np.abs(faces[0][3] - faces[1][3]) / 2) + min([faces[0][3], faces[1][3]])
        return middle_xpoint

    @staticmethod
    def _make_talkers_motion_check(vid, end_frame_time, x_middle, debug_frames=False):
        last_frame = None

        ret, frame = vid.read()
        if not ret:
            return [], x_middle

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        curr_video_pos_msec = vid.get(cv.CAP_PROP_POS_MSEC)

        if x_middle is None:
            x_middle = SyncSubtitles2Videos.__find_middle_xpoint_in_faces_bbox(frame)
            while x_middle is None:
                ret, frame = vid.read()
                if not ret:
                    return [], x_middle
                x_middle = SyncSubtitles2Videos.__find_middle_xpoint_in_faces_bbox(frame)

        most_white = []
        while curr_video_pos_msec <= end_frame_time:
            (thresh, bin_frame) = cv.threshold(gray_frame, 127, 255, cv.THRESH_BINARY)
            curr_frame = bin_frame - last_frame if last_frame is not None else bin_frame
            last_frame = bin_frame
            (thresh, curr_frame) = cv.threshold(curr_frame, 127, 255, cv.THRESH_BINARY)

            left = curr_frame[:, :int(x_middle)]
            right = curr_frame[:, int(x_middle):]

            if debug_frames:
                # key = 0
                # while key != 27:
                final_frame = cv.hconcat([left, right, gray_frame])
                cv.imshow('debug_make_talker_motion_check', final_frame)
                key = cv.waitKey(30)

            most_white.append([np.count_nonzero(left > 1), np.count_nonzero(right > 1)])

            ret, bin_frame = vid.read()
            if not ret:
                print(curr_video_pos_msec)
                break
            gray_frame = cv.cvtColor(bin_frame, cv.COLOR_BGR2GRAY)
            curr_video_pos_msec = vid.get(cv.CAP_PROP_POS_MSEC)

        cv.destroyAllWindows()

        most_talkers = [
            np.sum([x[0] for x in most_white]),
            np.sum([x[1] for x in most_white])
        ]
        return most_talkers, x_middle

    def _face_rec_create_encodings(self, vc, amount_faces: int = 20, show=False, ret_amount_faces=False,
                                   pbar: tqdm = None):
        """

        Parameters
        ----------
        vc : cv.VideoCapture
        amount_faces : int
        show : bool
        ret_amount_faces : bool
        pbar : tqdm

        Returns
        -------

        """

        def x_middle(face):
            return np.abs((face[1] - face[3]) / 2) + face[3]

        encondings = []

        if pbar is not None:
            pbar.reset(total=amount_faces)

        count = 0
        while count < amount_faces:
            ret, frame = vc.read()

            if not ret:
                return [], 0

            face_location = face_recognition.face_locations(frame[:, :, ::-1], number_of_times_to_upsample=1,
                                                            model='cnn')

            # print(face_location, [x_middle(x) for x in face_location])
            if len(face_location) > 1:
                continue

            curr_face_encoding = face_recognition.face_encodings(frame[:, :, ::-1], face_location)
            encondings.append(curr_face_encoding)

            if show:
                self._draw_bbox_in_frame(frame, face_location[0])
                cv.imshow('win', frame)
                key = cv.waitKey(-1)
                if key == 27:  # exit on ESC
                    break

            if show:
                cv.destroyAllWindows()

            if pbar is not None:
                pbar.update(1)
                pbar.refresh()

            count += 1

        return encondings if not ret_amount_faces else encondings, len(face_location)

    @staticmethod
    def __gen_train_set_from_encodings(p1_encode_vecs, p2_encode_vecs):
        embeddings = []
        y_train = []
        lbl = 0
        for person_encodings in [p1_encode_vecs, p2_encode_vecs]:
            if person_encodings is None:
                continue

            for encode_vec in person_encodings:
                if len(encode_vec) > 0:
                    embeddings.append(encode_vec[0])
                    y_train.append(lbl)
            lbl = lbl + 1

        x_train = np.array(embeddings)
        x_train = x_train.reshape(x_train.shape[0], 128)

        return x_train, y_train

    def _draw_bbox_in_frame(self, frame, face, id=None, identity_score=None):
        x1 = face[3]
        y1 = face[0]

        x2 = face[1]
        y2 = face[2]
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), (50, 205, 50), 2)

        if id is not None and identity_score is not None:
            text = str(f'person_{id} {identity_score}')
            x_text = int(np.min([x1, x2]) - (cv.getTextSize(text, self.font, 0.5, 2)[0][0] // 2))
            cv.putText(frame, text,
                       (x_text, y1 - 5), self.font, 0.5, (50, 205, 50), 2)

        return frame

    def _find_faces_ids_and_embedings_sorted_left_2_right(self, captures, show_video1_frame=False, v_part=None):
        ret, frame = captures[0].read()
        if not ret:
            return [], []

        p1_encodings, amount1_faces = self._face_rec_create_encodings(vc=captures[1], ret_amount_faces=True)
        p2_encodings, amount2_faces = self._face_rec_create_encodings(vc=captures[2], ret_amount_faces=True)
        xtrain, ytrain = self.__gen_train_set_from_encodings(p1_encodings, p2_encodings)

        clf = KNN(n_neighbors=5)
        clf.fit(xtrain, ytrain)

        def check_faces_in_main_video_frame(frame_):
            faces = face_recognition.face_locations(frame_[:, :, ::-1], number_of_times_to_upsample=1, model='cnn')
            if len(faces) < 2:
                while len(faces) < 2:
                    ret, frame_ = captures[0].read()
                    if not ret:
                        return [], [], [], [], [], [], [], []

                    faces = face_recognition.face_locations(frame_[:, :, ::-1], number_of_times_to_upsample=1,
                                                            model='cnn')

            f1_mid = np.abs((faces[0][1] - faces[0][3]) / 2) + faces[0][3]
            f2_mid = np.abs((faces[1][1] - faces[1][3]) / 2) + faces[1][3]
            x_middle = np.abs((f1_mid - f2_mid) / 2) + np.min([f1_mid, f2_mid])

            left_id = 0 if f1_mid <= x_middle else 1
            right_id = 0 if left_id == 1 else 1

            left_encodings = face_recognition.face_encodings(frame_[:, :, ::-1], [faces[left_id]])
            right_encodings = face_recognition.face_encodings(frame_[:, :, ::-1], [faces[right_id]])

            left_encodings = left_encodings[0]; right_encodings = right_encodings[0]

            return faces, f1_mid, f2_mid, x_middle, left_encodings, right_encodings, p1_encodings, p2_encodings

        # if show_video1_frame:
        #     f = self._draw_bbox_in_frame(frame, faces[right_id], clf.predict(right_encodings.reshape((1, -1))),
        #                                  clf.predict_proba(right_encodings.reshape((1, -1))))
        #     f = self._draw_bbox_in_frame(f, faces[left_id], clf.predict(left_encodings.reshape((1, -1))),
        #                                  clf.predict_proba(left_encodings.reshape((1, -1))))
        #     key = 0
        #     while key != 27:
        #         if key == 122:
        #             ret, frame = captures[0].read()
        #
        #             faces = face_recognition.face_locations(frame[:, :, ::-1], number_of_times_to_upsample=1,
        #                                                     model='cnn')
        #             if len(faces) < 2:
        #                 continue
        #
        #             f1_mid = np.abs((faces[0][1] - faces[0][3]) / 2) + faces[0][3]
        #             f2_mid = np.abs((faces[1][1] - faces[1][3]) / 2) + faces[1][3]
        #             x_middle = np.abs((f1_mid - f2_mid) / 2) + np.min([f1_mid, f2_mid])
        #
        #             left_id = 0 if f1_mid <= x_middle else 1
        #             right_id = 0 if left_id == 1 else 1
        #
        #             left_encodings = face_recognition.face_encodings(frame[:, :, ::-1], [faces[left_id]])
        #             right_encodings = face_recognition.face_encodings(frame[:, :, ::-1], [faces[right_id]])
        #             left_encodings = left_encodings[0]; right_encodings = right_encodings[0]
        #
        #             f = self._draw_bbox_in_frame(frame, faces[right_id], clf.predict(right_encodings.reshape((1, -1))),
        #                                          clf.predict_proba(right_encodings.reshape((1, -1))))
        #             f = self._draw_bbox_in_frame(f, faces[left_id], clf.predict(left_encodings.reshape((1, -1))),
        #                                          clf.predict_proba(left_encodings.reshape((1, -1))))
        #         cv.imshow('w', f)
        #         key = cv.waitKey(30)
        #
        #     cv.destroyAllWindows()

        faces, f1_mid, f2_mid, x_middle, left_encodings, right_encodings, p1_encodings, p2_encodings \
            = check_faces_in_main_video_frame(frame)

        if len(faces) == 0:
            return [], []

        left_id = np.argmax(clf.predict_proba(left_encodings.reshape((1, -1))))
        right_id = np.argmax(clf.predict_proba(right_encodings.reshape((1, -1))))

        if left_id == right_id:
            try_more = 30 * 5
            pbar_2 = tqdm(desc=f'get bad decision in video with vpart -> {v_part}', total=try_more)
            while try_more > 0 and left_id == right_id:
                ret, frame = captures[0].read()
                faces, f1_mid, f2_mid, x_middle, left_encodings, right_encodings, p1_encodings, p2_encodings \
                    = check_faces_in_main_video_frame(frame)

                if len(faces) == 0:
                    return [], []

                left_id = np.argmax(clf.predict_proba(left_encodings.reshape((1, -1))))
                right_id = np.argmax(clf.predict_proba(right_encodings.reshape((1, -1))))

                try_more -= 1
                pbar_2.update(1)
                #pbar_2.refresh()


        if show_video1_frame:
            cv.circle(frame, center=(int(f1_mid), faces[0][2]), radius=5, color=(255, 0, 0), thickness=-1)
            p1_str = 'left' if f1_mid <= x_middle else 'right'
            vid1_id = clf.predict_proba(left_encodings.reshape((1, -1))) if p1_str == 'left' \
                else clf.predict_proba(right_encodings.reshape((1, -1)))

            text = str(f'person_{vid1_id} {p1_str}')
            x_text = int(f1_mid - (cv.getTextSize(text, self.font, 0.5, 2)[0][0] // 2))
            cv.putText(frame, text, (x_text, faces[0][2] - 5), self.font, 0.5, (50, 205, 50), 2)

            cv.circle(frame, center=(int(f2_mid), faces[1][2]), radius=5, color=(0, 255, 0), thickness=-1)
            p2_str = 'left' if f2_mid <= x_middle else 'right'
            vid2_id = clf.predict_proba(left_encodings.reshape((1, -1))) if p2_str == 'left' \
                else clf.predict_proba(right_encodings.reshape((1, -1)))

            text = str(f'person_{vid2_id} {p2_str}')
            x_text = int(f2_mid - (cv.getTextSize(text, self.font, 0.5, 2)[0][0] // 2))
            cv.putText(frame, text, (x_text, faces[0][2] - 5), self.font, 0.5, (50, 205, 50), 2)

            # print(x_middle, frame.shape, f1_mid, f2_mid)

            vid1_id = np.argmax(vid1_id) + 1
            vid2_id = np.argmax(vid2_id) + 1

            cv.circle(frame, center=(int(x_middle), faces[0][2]), radius=5, color=(0, 0, 255), thickness=-1)
            ret, left_frame = captures[vid1_id].read() if p1_str == 'left' else captures[vid2_id].read()
            ret, right_frame = captures[vid1_id].read() if p2_str == 'left' else captures[vid2_id].read()
            try:
                frame = cv.hconcat([frame, left_frame, right_frame])
            except cv.error:
                print(f'error comcat {v_part}')
            plt.figure(0, dpi=720 // 9, figsize=(21, 9))
            plt.title(f'{v_part}')
            plt.imshow(frame[:, :, ::-1])
            plt.savefig(f'../fig-folder-libras/{v_part}.pdf')
            plt.show()

        if right_id == left_id:
            return [], [-1, -1]

        return ([p1_encodings if left_id == 0 else p2_encodings, p1_encodings if right_id == 0 else p2_encodings],
                [left_id + 1, right_id + 1])

    def __find_left_signaling_in_single_video(self, vid, holes, debug_frame=False):
        """

        Parameters
        ----------

        holes: List

        """
        all_left_in_video = {'Left': 0, 'Right': 0}
        x_middle = None

        for c_hole_pos in range(len(holes)):
            fps = int(vid.get(cv.CAP_PROP_FPS))
            beg_frame_time = holes[c_hole_pos]['beg']
            res = vid.set(cv.CAP_PROP_POS_MSEC, beg_frame_time)
            if not res:
                raise RuntimeError('Can not set position in this video')

            end_frame_time = holes[c_hole_pos]['end']

            most_white, x_middle = self._make_talkers_motion_check(vid, end_frame_time, x_middle, debug_frame)
            if len(most_white) == 0:
                continue

            all_left_in_video['Left'] += most_white[0]

            all_left_in_video['Right'] += most_white[1]

        return all_left_in_video

    def __find_where_signaling_is_quiet(self, folder_name, talker_id, pbar=None):
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
        talker_1_signs = self.all_videos[self.all_videos.folder == folder_name]
        talker_1_signs = talker_1_signs[talker_1_signs.talker_id == talker_id]
        # talker_1_signs = talker_1_signs[talker_1_signs.hand == 2]

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


if __name__ == '__main__':
    sync_vid_db = SyncSubtitles2Videos('all_videos.csv',
                                       '/media/usuario/Others/gdrive/')
    #sync_vid_db.process_all_folders_2_sync_with_person_2_video()
    persons_sub_df = sync_vid_db.find_all_left_signaler()

