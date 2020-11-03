import os
import cv2 as cv
import pandas as pd

from tqdm import tqdm

from libras_classifiers.librasdb_loaders import DBLoader2NPY
from pose_extractor.extract_mutitple_videos import ExtractMultipleVideos
from pose_extractor.pose_centroid_tracker import PoseCentroidTracker
from visualization.db_player import View2VideoDB


class OcvVideoPlayer:

    def __init__(self, libras_corpus_path, sign_db_path, all_videos, vid_sync, all_persons_from_subtitle):

        def read_csv_or_ret_df(csv_or_df):
            return csv_or_df if isinstance(csv_or_df, pd.DataFrame) else pd.read_csv(csv_or_df)

        self.libras_corpus_path = libras_corpus_path
        self.sign_db_path = sign_db_path
        self.all_videos = read_csv_or_ret_df(all_videos)
        self.vid_sync = read_csv_or_ret_df(vid_sync)
        self.all_persons_from_subtitle = read_csv_or_ret_df(all_persons_from_subtitle)
        self._all_samples_name, self.cls_dirs = DBLoader2NPY.read_all_db_folders(db_path=self.sign_db_path,
                                                                                 only_that_classes=['PORQUE', 'HOMEM', 'NÃO','COMO', 'TER'],#only_that_classes=None,
                                                                                 angle_or_xy='xy-hands',
                                                                                 custom_internal_dir='')
        self.curr_sample_idx = 0
        self.cap = None
        self.curr_sample_joints = None
        self.last_v_part = None
        self.curr_frame_time_ms = 1000 // 30
        self.last_talker_id = None
        self.update_cap_n_sample_joints_df()

        self.selected_signs = []

        self.joints_angle_2_use = [
            'Neck-RShoulder-RElbow',
            'RShoulder-RElbow-RWrist',
            'Neck-LShoulder-LElbow',
            'LShoulder-LElbow-LWrist',
            'RShoulder-Neck-LShoulder',
            'left-Wrist-left-ThumbProximal-left-ThumbDistal',
            'right-Wrist-right-ThumbProximal-right-ThumbDistal',
            'left-Wrist-left-IndexFingerProximal-left-IndexFingerDistal',
            'right-Wrist-right-IndexFingerProximal-right-IndexFingerDistal',
            'left-Wrist-left-MiddleFingerProximal-left-MiddleFingerDistal',
            'right-Wrist-right-MiddleFingerProximal-right-MiddleFingerDistal',
            'left-Wrist-left-RingFingerProximal-left-RingFingerDistal',
            'right-Wrist-right-RingFingerProximal-right-RingFingerDistal',
            'left-Wrist-left-LittleFingerProximal-left-LittleFingerDistal',
            'right-Wrist-right-LittleFingerProximal-right-LittleFingerDistal'
        ]

        self.color_array = ['#e53242', '#ffb133', '#3454da', '#ddc3d0', '#005a87', '#df6722', '#00ffff',
                            '#b7b7b7', '#ddba95', '#ffb133', '#4b5c09', '#00ff9f', '#e2f4c7', '#a2798f',
                            '#8caba8', '#ffb3ba', '#ffdfba', '#ffffba', '#baffc9', '#ff8000', '#8f139f']

        self.color_map = {}
        for x in self.joints_angle_2_use:
            if x not in self.color_map:
                for col in self.color_array:
                    if col not in self.color_map.values():
                        self.color_map.update({x: col})

    def update_cap_n_sample_joints_df(self):
        curr_opts = self._get_opts_from_curr_vid()

        self.curr_sample_joints = pd.read_csv(self._all_samples_name[self.curr_sample_idx][0])
        self.curr_sample_joints = self.curr_sample_joints.applymap(PoseCentroidTracker.parse_npy_vec_str)

        if self.last_v_part is None and self.last_talker_id is None:
            self.last_v_part = curr_opts['v_part']
            self.last_talker_id = curr_opts['talker_id']
        elif self.last_v_part == curr_opts['v_part'] and self.last_talker_id == curr_opts['talker_id']:
            self.cap.set(cv.CAP_PROP_POS_MSEC, curr_opts['beg'])
            return

        vid_path, vid_name = ExtractMultipleVideos.read_vid_path_from_vpart(curr_opts['curr_folder'],
                                                                            curr_opts['video_cam'])

        if self.cap is not None:
            self.cap.release()

        self.cap = cv.VideoCapture(vid_path)
        self.cap.set(cv.CAP_PROP_POS_MSEC, curr_opts['beg'])

    def _get_opts_from_curr_vid(self):
        curr_opts_list = self._all_samples_name[self.curr_sample_idx][0].replace('\\', '/').split('/')[-1].split('---')

        v_part = int(curr_opts_list[0])
        talker_id = int(curr_opts_list[-1][0])
        folder = self.all_persons_from_subtitle[self.all_persons_from_subtitle.v_part == v_part].folder_name.values[0]
        curr_folder_path = os.path.join(self.libras_corpus_path, folder)
        beg = int(curr_opts_list[2])
        end = int(curr_opts_list[3])
        sign=curr_opts_list[1]

        persons_id_sub = self.all_persons_from_subtitle[self.all_persons_from_subtitle['v_part'] == v_part]
        persons_id_syc = self.vid_sync[self.vid_sync['v_part'] == v_part]
        is_left = persons_id_sub.left_person.values[0] == talker_id

        vid_id = persons_id_syc.left_id if is_left else persons_id_syc.right_id

        return dict(
            v_part=v_part,
            talker_id=talker_id,
            video_cam=vid_id.values[0] + 1,
            curr_folder=curr_folder_path,
            beg=beg,
            end=end,
            sign=sign,
            name='---'.join(curr_opts_list).split('.csv')[0]

        )

    def save_all(self):
        for it in tqdm(range(len(self._all_samples_name))):
            self.save_single_vid(it)

    def save_single_vid(self, pos):
        self.curr_sample_idx = pos
        curr_opts = self._get_opts_from_curr_vid()
        self.update_cap_n_sample_joints_df()

        ret, frame = self.cap.read()
        if not ret:
            return False

        curr_msec = self.cap.get(cv.CAP_PROP_POS_MSEC)
        self.__write_joints_2_im(frame, curr_msec)

        fourcc = int(cv.VideoWriter_fourcc(*'H264'))
        video_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(self.cap.get(cv.CAP_PROP_FPS))

        if os.path.exists('../vid-folder/' + curr_opts['name'] + '.mp4'):
            return True

        vid_save = cv.VideoWriter('../vid-folder/' + curr_opts['name'] + '.mp4', fourcc, video_fps, (video_width, video_height))
        vid_save.write(frame)

        while curr_msec <= curr_opts['end']:

            ret, frame = self.cap.read()
            curr_msec = self.cap.get(cv.CAP_PROP_POS_MSEC)
            if not ret:
                vid_save.release()
                return False
            self.__write_joints_2_im(frame, curr_msec)

            vid_save.write(frame)

        return True

    def run(self):
        curr_opts = self._get_opts_from_curr_vid()
        while True:
            ret, frame = self.cap.read()
            curr_msec_pos = self.cap.get(cv.CAP_PROP_POS_MSEC)

            if curr_msec_pos >= curr_opts['end']:
                self.cap.set(cv.CAP_PROP_POS_MSEC, curr_opts['beg'])

            self.__write_joints_2_im(frame, curr_msec_pos)
            cv.imshow(f'Player {curr_opts["sign"]}', frame)
            key = cv.waitKey(self.curr_frame_time_ms)
            if key == 27:
                break
            elif key & 0xFF == ord('p'):
                self.curr_sample_idx += 1
                curr_opts = self._get_opts_from_curr_vid()
                self.update_cap_n_sample_joints_df()

        cv.destroyAllWindows()

    def __write_joints_2_im(self, im, frame_pos, radius=2, color=(255, 0, 255)):
        frame_search_res = self.curr_sample_joints.frame == int(frame_pos)
        if not any(frame_search_res.values):
            return

        joints_used = []
        for k in self.joints_angle_2_use:
            keys = k.split('-')
            if 'left' in keys or 'right' in keys:
                keys = [keys[0] + '-' + keys[1],
                        keys[2] + '-' + keys[3],
                        keys[4] + '-' + keys[5]]

            joints_used.extend(keys)

        joints_at_frame = self.curr_sample_joints[frame_search_res]
        for joint_name in joints_used:
            joint = joints_at_frame[joint_name].values[0]
            try:
                cv.circle(im, tuple(map(int, joint[:2])), radius, color, 1)
            except BaseException:
                continue

        joints_at_frame_df = self.curr_sample_joints[frame_search_res]
        for key in self.joints_angle_2_use:
            keys = key.split('-')
            if 'left' in keys or 'right' in keys:
                keys = [keys[0] + '-' + keys[1],
                        keys[2] + '-' + keys[3],
                        keys[4] + '-' + keys[5]]

            color = View2VideoDB.hex_2_rgb(self.color_map[key])
            try:
                joint_0 = joints_at_frame_df[keys[0]].values[0][:2]
                joint_1 = joints_at_frame_df[keys[1]].values[0][:2]
                joint_2 = joints_at_frame_df[keys[2]].values[0][:2]

                cv.line(im, tuple(map(int, joint_1)), tuple(map(int, joint_0)), color=color)
                cv.line(im, tuple(map(int, joint_1)), tuple(map(int, joint_2)), color=color)
            except BaseException:
                continue
                

if __name__ == '__main__':
    ocv_video_player = OcvVideoPlayer(libras_corpus_path='d:/gdrive', sign_db_path='../sign_db_front_view',
                                      all_videos='all_videos.csv', vid_sync='vid_sync.csv',
                                      all_persons_from_subtitle='all_persons_from_subtitle.csv')
    ocv_video_player.save_all()