import cv2
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pose_extractor.pose_centroid_tracker import PoseCentroidTracker
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QMutex
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QPushButton, \
    QHBoxLayout, QVBoxLayout


class OCVVideoThread(QThread):
    changePixmap = pyqtSignal(QImage)
    pause = False
    cap = None

    def read_cur_video(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        return ret, frame

    def prev_vid(self):
        pass

    def next_vid(self):
        pass

    def prev_sample(self):
        pass

    def next_sample(self):
        pass

    def run(self):
        ret = True
        frame = None
        while True:
            if not self.pause:
                ret, frame = self.read_cur_video()
            if ret and frame is not None:
                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine,
                                           QImage.Format_RGB888)
                p = convertToQtFormat.scaled(1920, 1080, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class App(QWidget):
    label = None
    title = 'Debugger DB PLayer'
    next_btn = None
    prev_btn = None
    pause_btn = None
    video_name_label = None
    th = None
    th_cls = None
    th_cls_kwargs = None
    next_sample_btn = None
    prev_sample_btn = None

    def __init__(self, th_cls=OCVVideoThread, **th_cls_kwargs):
        super().__init__()
        self.th_cls = th_cls
        self.th_cls_kwargs=th_cls_kwargs
        self.init_ui()

    @pyqtSlot(QImage)
    def set_image(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def init_ui(self):
        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.next_btn = QPushButton('next video')
        self.prev_btn = QPushButton('prev video')
        self.next_sample_btn = QPushButton('next sample')
        self.prev_sample_btn = QPushButton('prev sample')
        self.pause_btn = QPushButton('pause')

        self.pause_btn.clicked.connect(self.click_pause_btn)
        self.prev_btn.clicked.connect(self.click_prev_btn)
        self.next_btn.clicked.connect(self.click_next_btn)
        self.prev_sample_btn.clicked.connect(self.click_prev_sample_btn)
        self.next_sample_btn.clicked.connect(self.click_next_sample_btn)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.prev_btn)
        hbox.addWidget(self.prev_sample_btn)
        hbox.addWidget(self.pause_btn)
        hbox.addWidget(self.next_sample_btn)
        hbox.addWidget(self.next_btn)

        vbox = QVBoxLayout()
        vbox.addStretch(1)

        # self.resize(1366, 768)
        # create a label
        self.label = QLabel(self)
        self.label.resize(1920, 1080)

        self.th = self.th_cls(**self.th_cls_kwargs)

        vbox.addWidget(self.label)

        vbox.addLayout(hbox)

        self.th.changePixmap.connect(self.set_image)
        self.th.start()

        self.setLayout(vbox)
        self.show()

    def click_pause_btn(self):
        self.th.pause = not self.th.pause

    def click_next_sample_btn(self):
        self.th.next_sample()

    def click_prev_sample_btn(self):
        self.th.prev_sample()

    def click_next_btn(self):
        self.th.next_vid()

    def click_prev_btn(self):
        self.th.next_vid()


class ViewDBCutVideos(OCVVideoThread):

    def __init__(self, all_videos_df_path, db_path, angle_db):
        super().__init__()
        self.all_videos = pd.read_csv(all_videos_df_path)
        self.all_videos_path = sorted(list(self.all_videos.folder_name.unique()))
        self.curr_video_idx = 0
        self.cap = None
        self.db_path = db_path
        self.curr_frame_time_sec = None
        self.curr_frame_time_ms = None
        self.all_samples_path_from_video = None
        self.curr_sample_idx = 0
        self.curr_sample_opts = None
        self.angle_db = angle_db
        self.mutex = QMutex()
        self.__update_videos_n_paths()
        self.class_list = self.make_list_class_samples()
        self.curr_sample_joints = None
        self.curr_sample_angles = None

    def make_list_class_samples(self):
        cls = list(map(lambda x: (x[0], x[1].split('/')[-1]), enumerate(self.all_samples_path_from_video)))
        return cls

    def read_cur_video(self):
        ret, frame = None, None

        if self.cap is None:
            video_path = os.path.join(self.db_path, self.all_videos_path[self.curr_video_idx])
            self.mutex.lock()
            self.cap = cv2.VideoCapture(video_path)
            self.curr_frame_time_sec = (1000 / self.cap.get(cv2.CAP_PROP_FPS)) / 1000
            self.curr_frame_time_ms = 1000 / self.cap.get(cv2.CAP_PROP_FPS)
            self.mutex.unlock()

            self.__reset_cap_position()

        else:
            time.sleep(self.curr_frame_time_sec)

            self.mutex.lock()
            ret, frame = self.cap.read()
            curr_frame_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            if curr_frame_pos >= (self.curr_sample_opts['end'] // self.curr_frame_time_ms):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.curr_sample_opts['beg'] // self.curr_frame_time_ms)
            self.mutex.unlock()

            if not ret:
                raise RuntimeError(f'bad video {self.all_videos_path[self.curr_video_idx]}')

            self.__write_joints_2_im(frame)

            frame = cv2.resize(frame, (1920, 1080))
            self.__write_info_2_im(frame, font_scale=1)

            frame_angle = self.__make_matplotlib_2_npy(curr_frame_pos)
            frame = self.__hconcat_resize_min([frame, frame_angle])

        return ret, frame

    @staticmethod
    def __hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
        h_max = max(im.shape[0] for im in im_list)
        im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_max / im.shape[0]), h_max), interpolation=interpolation)
                          for im in im_list]
        return cv2.hconcat(im_list_resize)

    def prev_sample(self):
        self.curr_sample_idx = self.curr_sample_idx - 1 if self.curr_sample_idx != 0 \
                                                        else len(self.all_samples_path_from_video)
        self.__reset_cap_position()

    def next_sample(self):
        self.curr_sample_idx = self.curr_sample_idx + 1 \
            if self.curr_sample_idx < len(self.all_samples_path_from_video) - 1 else 0
        self.__reset_cap_position()

    def prev_vid(self):
        self.curr_video_idx = self.curr_video_idx - 1 if self.curr_video_idx != 0 else len(self.all_videos_path) - 1
        self.__update_videos_n_paths()

    def next_vid(self):
        self.curr_video_idx = self.curr_video_idx + 1 if self.curr_video_idx < len(self.all_videos_path) - 1 else 0
        self.__update_videos_n_paths()

    def __reset_cap_position(self):
        video_path = os.path.join(self.db_path, self.all_videos_path[self.curr_video_idx])

        sample_name_non_split = \
            self.all_samples_path_from_video[self.curr_sample_idx].replace('\\', '/').split('/')[-1]
        # pos 4: video name; pos 5: sinal;  pos 7: beg; pos 9: end + '.csv'
        sample_name = \
            self.all_samples_path_from_video[self.curr_sample_idx].replace('\\', '/').split('/')[-1].split('-')
        sample_csv_path = os.path.join(self.db_path,
                                       *(self.all_videos_path[self.curr_video_idx].replace('\\', '/').split('/')[:-1]),
                                       self.all_samples_path_from_video[self.curr_sample_idx])
        self.curr_sample_joints = pd.read_csv(sample_csv_path)
        self.curr_sample_joints = self.curr_sample_joints.applymap(PoseCentroidTracker.parse_npy_vec_str)
        self.curr_sample_opts = {'beg': float(sample_name[7]),
                                 'end': float(sample_name[9].split('.')[0]),
                                 'video_name': sample_name[5], 'sign': sample_name[5]}
        cur_angle_df_path = os.path.join(self.angle_db, self.curr_sample_opts['sign'], 'hands-angle',
                                         sample_name_non_split)
        self.curr_sample_angles = pd.read_csv(cur_angle_df_path)
        if self.mutex.tryLock():
            succ = self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.curr_sample_opts['beg'] // self.curr_frame_time_ms)
            self.mutex.unlock()
            if not succ:
                raise RuntimeError(f'Could not set frame pos {video_path}')

    def __update_videos_n_paths(self):
        path_end_in_folder = self.all_videos_path[self.curr_video_idx].replace('\\', '/').split('/')[:-1]
        path_end_in_folder = os.path.join(*path_end_in_folder)
        video_path = os.path.join(self.db_path, path_end_in_folder)
        self.all_samples_path_from_video = list(filter(lambda x: 'sample' in x, os.listdir(video_path)))
        if self.cap is not None:
            self.cap.release()

    @staticmethod
    def __convert_msec_2_hour_text(ms):
        ms = int(ms)
        seconds = (ms / 1000) % 60
        milisec = (seconds % 1) * 1000
        seconds = int(seconds)
        minutes = (ms / (1000 * 60)) % 60
        minutes = int(minutes)
        hours = (ms / (1000 * 60 * 60)) % 24
        return '%d: %d: %d: %d' % (hours, minutes, seconds, milisec)

    def __write_info_2_im(self, im, font_scale=0.3):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (10, 200)
        font_color = (255, 255, 255)
        line_type = 2

        for it, opt in enumerate(self.curr_sample_opts.items()):
            text = str(opt[1]) if not isinstance(opt[1], float) else self.__convert_msec_2_hour_text(opt[1])
            cv2.putText(im, text,
                        (bottom_left_corner_of_text[0], bottom_left_corner_of_text[1] + it * 40),
                        font,
                        font_scale,
                        font_color,
                        line_type)

    def __write_joints_2_im(self, im, radius=2, color=(255, 0, 255)):
        frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_search_res = self.curr_sample_joints.frame == frame_pos
        if not any(frame_search_res.values):
            return

        joints_at_frame = self.curr_sample_joints[frame_search_res].values[0][2:]
        for joint in joints_at_frame:
            cv2.circle(im, tuple(map(int, joint[:2])), radius, color, 1)

    def __make_matplotlib_2_npy(self, frame_pos, range_base=5):
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        first_frame = self.curr_sample_angles.frame.iloc[0]
        beg = frame_pos - range_base if frame_pos - range_base > first_frame else first_frame
        frame_pos = int(frame_pos)
        beg = int(beg)
        for key in self.curr_sample_angles.keys():
            angle_data = self.curr_sample_angles[key].iloc[beg:frame_pos]
            ax.plot([x + beg for x in range(len(angle_data))], angle_data)

        canvas.draw()  # draw the canvas, cache the renderer

        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image

    def __render_all_matplotlib_frames(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = App(th_cls=ViewDBCutVideos, all_videos_df_path='all_videos.csv', db_path='D:/gdrive/LibrasCorpus',
            angle_db='../libras-db-folders')
    w.show()

    sys.exit(app.exec_())
