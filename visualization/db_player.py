import cv2
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from copy import copy
from libras_classifiers.librasdb_loaders import DBLoader2NPY
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from pose_extractor.extract_mutitple_videos import ExtractMultipleVideos
from pose_extractor.pose_centroid_tracker import PoseCentroidTracker
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QMutex, QModelIndex
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QApplication,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QSlider,
    QListWidget,
    QListWidgetItem,
)


class OCVVideoThread(QThread):
    changePixmap = pyqtSignal(QImage)
    pause = False
    cap = None
    time_slider = None
    curr_vid_interval: QSlider = None
    curr_samples_list: QListWidget = None
    vid_list: QListWidget = None

    def change_to_sample_idx(self, idx):
        pass

    def set_sample_position(self, pos):
        pass

    def change_curr_vid_pos(self, pos):
        pass

    def change_fps_video(self, val):
        pass

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
                convertToQtFormat = QImage(
                    rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888
                )
                p = convertToQtFormat.scaled(2560, 1080)
                self.changePixmap.emit(p)


class App(QWidget):
    label = None
    title = "Debugger DB PLayer"
    next_btn = None
    prev_btn = None
    pause_btn = None
    video_name_label = None
    th = None
    th_cls = None
    th_cls_kwargs = None
    next_sample_btn = None
    prev_sample_btn = None
    vid_time_slider = None
    vid_time_label = None
    fps_slider = None
    fps_label = None
    vid_list = None
    sample_list = None
    save_fig_btn = None

    def __init__(self, th_cls=OCVVideoThread, **th_cls_kwargs):
        super().__init__()
        self.th_cls = th_cls
        self.th_cls_kwargs = th_cls_kwargs
        self.init_ui()

    @pyqtSlot(QImage)
    def set_image(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.save_fig_btn = QPushButton("Save Curr Figure")
        self.next_btn = QPushButton("next video")
        self.prev_btn = QPushButton("prev video")
        self.next_sample_btn = QPushButton("next sample")
        self.prev_sample_btn = QPushButton("prev sample")
        self.pause_btn = QPushButton("pause")

        self.pause_btn.clicked.connect(self.click_pause_btn)
        self.prev_btn.clicked.connect(self.click_prev_btn)
        self.next_btn.clicked.connect(self.click_next_btn)
        self.prev_sample_btn.clicked.connect(self.click_prev_sample_btn)
        self.next_sample_btn.clicked.connect(self.click_next_sample_btn)
        self.save_fig_btn.clicked.connect(self.save_fig_fn)

        self.fps_label = QLabel("0", self)
        self.fps_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.fps_label.setMinimumWidth(80)
        self.fps_label.setText(str(30))

        self.fps_slider = QSlider(Qt.Horizontal, self)
        self.fps_slider.setValue(30)
        self.fps_slider.setRange(1, 60)
        self.fps_slider.valueChanged.connect(self.changed_fps_video)

        self.vid_time_slider = QSlider(Qt.Horizontal, self)
        self.vid_time_slider.valueChanged.connect(self.changed_time_video)
        self.vid_time_label = QLabel("0", self)

        hbox_botton = QHBoxLayout()
        hbox_botton.addStretch(1)

        hbox_botton.addWidget(self.vid_time_slider, Qt.AlignVCenter)
        hbox_botton.addWidget(self.vid_time_label, Qt.AlignVCenter)
        hbox_botton.addWidget(self.fps_slider, Qt.AlignVCenter)
        hbox_botton.addWidget(self.fps_label, Qt.AlignVCenter)
        hbox_botton.addWidget(self.save_fig_btn, Qt.AlignVCenter)
        hbox_botton.addWidget(self.prev_btn, Qt.AlignVCenter)
        hbox_botton.addWidget(self.prev_sample_btn, Qt.AlignVCenter)
        hbox_botton.addWidget(self.pause_btn, Qt.AlignVCenter)
        hbox_botton.addWidget(self.next_sample_btn, Qt.AlignVCenter)
        hbox_botton.addWidget(self.next_btn, Qt.AlignVCenter)

        vbox = QVBoxLayout()
        vbox.addStretch(1)

        # create a label
        self.label = QLabel(self)
        self.sample_list = QListWidget()
        self.sample_list.itemSelectionChanged.connect(self.click_on_item)

        self.th = self.th_cls(
            **self.th_cls_kwargs,
            th_list=self.sample_list,
            th_slider=self.vid_time_slider,
        )

        hbox_top = QHBoxLayout()

        hbox_top.addWidget(self.label, Qt.AlignTop)
        hbox_top.addWidget(self.sample_list, Qt.AlignTop)

        vbox.addLayout(hbox_top, Qt.AlignTop)
        vbox.addLayout(hbox_botton)

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

    def changed_fps_video(self, val):
        self.fps_label.setText(str(val))
        self.th.change_fps_video(val)

    def changed_time_video(self, val):
        self.vid_time_label.setText(str(val))

    def save_fig_fn(self):
        self.th.make_matplotlib_2_npy(None)

    def click_on_item(self):
        self.th.change_to_sample_idx(self.sample_list.selectedIndexes()[0].row())


class ViewDBCutVideos(OCVVideoThread):
    def __init__(
        self, all_videos_df_path, db_path, angle_db, th_list, th_slider, vid_sync
    ):
        super().__init__()
        self.curr_samples_list = th_list
        self.time_slider = th_slider
        self.all_videos = pd.read_csv(all_videos_df_path)
        self.vid_sync = (
            vid_sync if isinstance(vid_sync, pd.DataFrame) else pd.read_csv(vid_sync)
        )
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
        self.memory_angle_plot = {}
        self.color_map = {}
        self.save_fig = False
        self.frame_angle = None
        self.could_no_reset_cap = False
        self.last_frame = None

        self.joints_angle_2_use = [
            "Neck-RShoulderight-RElbow",
            "RShoulderight-RElbow-RWrist",
            "Neck-LShoulderight-LElbow",
            "LShoulderight-LElbow-LWrist",
            "RShoulderight-Neck-LShoulder",
            "left-Wrist-left-ThumbProximal-left-ThumbDistal",
            "right-Wrist-right-ThumbProximal-right-ThumbDistal",
            "left-Wrist-left-IndexFingerProximal-left-IndexFingerDistal",
            "right-Wrist-right-IndexFingerProximal-right-IndexFingerDistal",
            "left-Wrist-left-MiddleFingerProximal-left-MiddleFingerDistal",
            "right-Wrist-right-MiddleFingerProximal-right-MiddleFingerDistal",
            "left-Wrist-left-RingFingerProximal-left-RingFingerDistal",
            "right-Wrist-right-RingFingerProximal-right-RingFingerDistal",
            "left-Wrist-left-LittleFingerProximal-left-LittleFingerDistal",
            "right-Wrist-right-LittleFingerProximal-right-LittleFingerDistal",
        ]

        self.color_array = [
            "#e53242",
            "#ffb133",
            "#3454da",
            "#ddc3d0",
            "#005a87",
            "#df6722",
            "#00ffff",
            "#b7b7b7",
            "#ddba95",
            "#ffb133",
            "#4b5c09",
            "#00ff9f",
            "#e2f4c7",
            "#a2798f",
            "#8caba8",
            "#ffb3ba",
            "#ffdfba",
            "#ffffba",
            "#baffc9",
            "#ff8000",
            "#8f139f",
        ]
        self.mutex = QMutex()
        self.joint_db_mutex = QMutex()
        self.angle_db_mutex = QMutex()

        self.__update_videos_n_paths()
        self.class_list = self.make_list_class_samples()
        self.curr_sample_joints = None
        self.curr_sample_angles = None

    def change_curr_vid_pos(self, pos):
        pass

    def set_sample_position(self, pos):
        pass

    def make_list_class_samples(self):
        cls = list(
            map(
                lambda x: (x[0], x[1].split("/")[-1]),
                enumerate(self.all_samples_path_from_video),
            )
        )
        return cls

    def change_fps_video(self, val):
        if self.mutex.tryLock():
            self.curr_frame_time_sec = 1000 / val
            self.mutex.unlock()

    def read_cur_video(self):
        ret, frame = None, None

        if self.cap is None:
            video_path = os.path.join(
                self.db_path, self.all_videos_path[self.curr_video_idx]
            )
            if self.mutex.tryLock():
                self.cap = cv2.VideoCapture(video_path)
                self.curr_frame_time_sec = (
                    (1000 / 30) / 1000
                    if self.curr_frame_time_sec is None
                    else self.curr_frame_time_sec
                )

                self.curr_frame_time_ms = 1000 / self.cap.get(cv2.CAP_PROP_FPS)
                self.mutex.unlock()

            self.__reset_cap_position()

        else:
            curr_frame_pos = None

            if self.could_no_reset_cap:
                self.__reset_cap()

            if self.mutex.tryLock():
                time.sleep(self.curr_frame_time_sec)
                ret, frame = self.cap.read()
                curr_frame_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                if curr_frame_pos >= (
                    self.curr_sample_opts["end"] // self.curr_frame_time_ms
                ):
                    self.cap.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        self.curr_sample_opts["beg"] // self.curr_frame_time_ms,
                    )
                self.mutex.unlock()

            if ret is None and frame is None:
                return True, self.last_frame

            if not ret:
                raise RuntimeError(
                    f"bad video {self.all_videos_path[self.curr_video_idx]}"
                )

            self.__write_joints_2_im(frame, curr_frame_pos)

            frame = cv2.resize(frame, (1920, 1080))
            self.__write_info_2_im(frame, font_scale=1)

            if self.frame_angle is not None:
                frame = self.hconcat_resize_min([frame, self.frame_angle])

        self.last_frame = frame
        return ret, frame

    @staticmethod
    def hex_2_rgb(h):
        h = h[1:]
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
        h_max = max(im.shape[0] for im in im_list)
        im_list_resize = [
            cv2.resize(
                im,
                (int(im.shape[1] * h_max / im.shape[0]), h_max),
                interpolation=interpolation,
            )
            for im in im_list
        ]
        return cv2.hconcat(im_list_resize)

    def change_to_sample_idx(self, idx):
        self.curr_sample_idx = idx
        self.__reset_cap_position()

    def prev_sample(self):
        self.curr_sample_idx = (
            self.curr_sample_idx - 1
            if self.curr_sample_idx != 0
            else len(self.all_samples_path_from_video)
        )
        self.__reset_cap_position()

    def next_sample(self):
        self.curr_sample_idx = (
            self.curr_sample_idx + 1
            if self.curr_sample_idx < len(self.all_samples_path_from_video) - 1
            else 0
        )
        self.__reset_cap_position()

    def prev_vid(self):
        self.curr_video_idx = (
            self.curr_video_idx - 1
            if self.curr_video_idx != 0
            else len(self.all_videos_path) - 1
        )
        self.__update_videos_n_paths()

    def next_vid(self):
        self.curr_video_idx = (
            self.curr_video_idx + 1
            if self.curr_video_idx < len(self.all_videos_path) - 1
            else 0
        )
        self.__update_videos_n_paths()

    def __reset_cap_position(self):
        video_path = os.path.join(
            self.db_path, self.all_videos_path[self.curr_video_idx]
        )

        sample_name_non_split = (
            self.all_samples_path_from_video[self.curr_sample_idx]
            .replace("\\", "/")
            .split("/")[-1]
        )
        # pos 4: video name; pos 5: sinal;  pos 7: beg; pos 9: end + '.csv'
        sample_name = (
            self.all_samples_path_from_video[self.curr_sample_idx]
            .replace("\\", "/")
            .split("/")[-1]
            .split("-")
        )
        sample_csv_path = os.path.join(
            self.db_path,
            *(
                self.all_videos_path[self.curr_video_idx]
                .replace("\\", "/")
                .split("/")[:-1]
            ),
            self.all_samples_path_from_video[self.curr_sample_idx],
        )

        self.curr_sample_joints = pd.read_csv(sample_csv_path)
        self.curr_sample_joints = self.curr_sample_joints.applymap(
            PoseCentroidTracker.parse_npy_vec_str
        )

        self.curr_sample_opts = {
            "beg": float(sample_name[7]),
            "end": float(sample_name[9].split(".")[0]),
            "video_name": sample_name[5],
            "sign": sample_name[5],
        }
        cur_angle_df_path = os.path.join(
            self.angle_db,
            self.curr_sample_opts["sign"],
            "hands-angle",
            sample_name_non_split,
        )
        self.curr_sample_angles = pd.read_csv(cur_angle_df_path)

        self.__reset_cap()

        beg_frame = self.curr_sample_angles.frame.iloc[0]
        end_frame = self.curr_sample_angles.frame.iloc[-1]
        self.time_slider.setRange(0, int(end_frame - beg_frame))
        self.frame_angle = self.make_matplotlib_2_npy(None)

    def __reset_cap(self):
        video_path = os.path.join(
            self.db_path, self.all_videos_path[self.curr_video_idx]
        )
        if self.mutex.tryLock():
            succ = self.cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                self.curr_sample_opts["beg"] // self.curr_frame_time_ms,
            )
            self.mutex.unlock()
            self.could_no_reset_cap = False
            if not succ:
                raise RuntimeError(f"Could not set frame pos {video_path}")
        else:
            self.could_no_reset_cap = True

    def __update_videos_n_paths(self):
        path_end_in_folder = (
            self.all_videos_path[self.curr_video_idx].replace("\\", "/").split("/")[:-1]
        )
        path_end_in_folder = os.path.join(*path_end_in_folder)
        video_path = os.path.join(self.db_path, path_end_in_folder)
        self.all_samples_path_from_video = list(
            filter(lambda x: "sample" in x, os.listdir(video_path))
        )
        for it, sample_name in enumerate(self.all_samples_path_from_video):
            name_to_add = "".join(sample_name.split("-")[-5:]) + " " + str(it)
            self.curr_samples_list.addItem(name_to_add)

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
        return "%d: %d: %d: %d" % (hours, minutes, seconds, milisec)

    def __write_info_2_im(self, im, font_scale=0.3):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (10, 200)
        font_color = (255, 255, 255)
        line_type = 2

        for it, opt in enumerate(self.curr_sample_opts.items()):
            text = (
                str(opt[1])
                if not isinstance(opt[1], float)
                else self.__convert_msec_2_hour_text(opt[1])
            )
            cv2.putText(
                im,
                text,
                (
                    bottom_left_corner_of_text[0],
                    bottom_left_corner_of_text[1] + it * 40,
                ),
                font,
                font_scale,
                font_color,
                line_type,
            )

    def __write_joints_2_im(self, im, frame_pos, radius=2, color=(255, 0, 255)):
        frame_search_res = self.curr_sample_joints.frame == frame_pos
        if not any(frame_search_res.values):
            return

        joints_used = []
        for k in self.joints_angle_2_use:
            keys = k.split("-")
            if "l" in keys or "r" in keys:
                keys = [
                    keys[0] + "-" + keys[1],
                    keys[2] + "-" + keys[3],
                    keys[4] + "-" + keys[5],
                ]

            joints_used.extend(keys)

        joints_at_frame = self.curr_sample_joints[frame_search_res]
        for joint_name in joints_used:
            joint = joints_at_frame[joint_name].values[0]
            try:
                cv2.circle(im, tuple(map(int, joint[:2])), radius, color, 1)
            except BaseException:
                continue

        joints_at_frame_df = self.curr_sample_joints[frame_search_res]
        for key in self.joints_angle_2_use:
            keys = key.split("-")
            if "l" in keys or "r" in keys:
                keys = [
                    keys[0] + "-" + keys[1],
                    keys[2] + "-" + keys[3],
                    keys[4] + "-" + keys[5],
                ]

            color = self.hex_2_rgb(self.color_map[key])
            joint_0 = joints_at_frame_df[keys[0]].values[0][:2]
            joint_1 = joints_at_frame_df[keys[1]].values[0][:2]
            joint_2 = joints_at_frame_df[keys[2]].values[0][:2]
            try:
                cv2.line(
                    im, tuple(map(int, joint_1)), tuple(map(int, joint_0)), color=color
                )
                cv2.line(
                    im, tuple(map(int, joint_1)), tuple(map(int, joint_2)), color=color
                )
            except BaseException:
                continue

    def make_matplotlib_2_npy(self, frame_pos, step_y=0.2, save_fig=False):
        if (
            frame_pos is not None
            and str(frame_pos) in self.memory_angle_plot
            and not self.save_fig
        ):
            return self.memory_angle_plot[str(frame_pos)]

        for x in self.curr_sample_angles.keys()[2:]:
            if x not in self.color_map:
                for col in self.color_array:
                    if col not in self.color_map.values():
                        self.color_map.update({x: col})

        fig = Figure(dpi=1080 // 9, figsize=(16, 9))
        canvas = FigureCanvas(fig)
        ax: plt.axes = fig.gca()
        ax.set_ylim(-4, 4)
        ax.set_xlim(
            self.curr_sample_angles.frame.iloc[0],
            self.curr_sample_angles.frame.iloc[-1],
        )
        ax.yaxis.set_ticks(np.arange(-4, 4, step_y))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.1f"))

        x_points = []
        y_points = []
        c_points = []
        for row in self.curr_sample_angles.iterrows():
            row: pd.Series = row[1]
            curr_y = row[self.joints_angle_2_use].values
            c_points.extend([self.color_map[x] for x in self.joints_angle_2_use])
            x_points.extend([row.frame] * (len(self.joints_angle_2_use)))
            y_points.extend(curr_y)

        ax.scatter(x_points, y_points, c=c_points)
        # ax.vlines(frame_pos, ymin=-4, ymax=4)

        for k in self.joints_angle_2_use:
            color = self.color_map[k]
            y = self.curr_sample_angles[k].values
            x = self.curr_sample_angles.frame.to_list()
            ax.plot(x, y, c=color, label=k)

        ax.legend(bbox_to_anchor=(1.2, 1.1), loc="upper right")

        canvas.draw()  # draw the canvas, cache the renderer

        if save_fig:
            fig.savefig("sample.pdf")

        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image

    def __render_all_matplotlib_frames(self):
        self.memory_angle_plot = {
            str(x[1].frame): self.make_matplotlib_2_npy(x[1].frame)
            for x in self.curr_sample_angles.iterrows()
        }


class View2VideoDB(OCVVideoThread):
    def __init__(
        self,
        all_videos_df_path,
        libras_corpus_db_path,
        vid_sync,
        front_view_db_path,
        all_persons_from_subtitle,
        th_list,
        th_slider,
    ):
        super().__init__()
        self.curr_samples_list = th_list
        self.time_slider = th_slider
        self.all_videos = (
            all_videos_df_path
            if isinstance(all_videos_df_path, pd.DataFrame)
            else pd.read_csv(all_videos_df_path)
        )

        self.front_view_db_path = front_view_db_path
        self.vid_sync = (
            vid_sync if isinstance(vid_sync, pd.DataFrame) else pd.read_csv(vid_sync)
        )
        self.all_v_parts = sorted(self.vid_sync.v_part.unique().tolist(), reverse=True)
        self.all_persons_from_subtitle = (
            all_persons_from_subtitle
            if isinstance(vid_sync, pd.DataFrame)
            else pd.read_csv(all_persons_from_subtitle)
        )

        self._all_samples_name, self.cls_dirs = DBLoader2NPY.read_all_db_folders(
            db_path=front_view_db_path,
            only_that_classes=None,
            angle_or_xy="xy-hands",
            custom_internal_dir="",
        )
        self.curr_folder_idx = 0
        self.curr_sample_idx = 0
        self.curr_video_idx = 0
        self.cap = None
        self.db_path = libras_corpus_db_path
        self.curr_frame_time_sec = None
        self.curr_frame_time_ms = None
        self.all_samples_path_from_video = None
        self.curr_sample_idx = 0
        self.memory_angle_plot = {}
        self.color_map = {}
        self.save_fig = False
        self.frame_angle = None
        self.could_no_reset_cap = False
        self.last_frame = None

        self.joints_angle_2_use = [
            "Neck-RShoulder-RElbow",
            "RShoulder-RElbow-RWrist",
            "Neck-LShoulder-LElbow",
            "LShoulder-LElbow-LWrist",
            "RShoulder-Neck-LShoulder",
            "left-Wrist-left-ThumbProximal-left-ThumbDistal",
            "right-Wrist-right-ThumbProximal-right-ThumbDistal",
            "left-Wrist-left-IndexFingerProximal-left-IndexFingerDistal",
            "right-Wrist-right-IndexFingerProximal-right-IndexFingerDistal",
            "left-Wrist-left-MiddleFingerProximal-left-MiddleFingerDistal",
            "right-Wrist-right-MiddleFingerProximal-right-MiddleFingerDistal",
            "left-Wrist-left-RingFingerProximal-left-RingFingerDistal",
            "right-Wrist-right-RingFingerProximal-right-RingFingerDistal",
            "left-Wrist-left-LittleFingerProximal-left-LittleFingerDistal",
            "right-Wrist-right-LittleFingerProximal-right-LittleFingerDistal",
        ]

        self.color_array = [
            "#e53242",
            "#ffb133",
            "#3454da",
            "#ddc3d0",
            "#005a87",
            "#df6722",
            "#00ffff",
            "#b7b7b7",
            "#ddba95",
            "#ffb133",
            "#4b5c09",
            "#00ff9f",
            "#e2f4c7",
            "#a2798f",
            "#8caba8",
            "#ffb3ba",
            "#ffdfba",
            "#ffffba",
            "#baffc9",
            "#ff8000",
            "#8f139f",
        ]

        for x in self.joints_angle_2_use:
            if x not in self.color_map:
                for col in self.color_array:
                    if col not in self.color_map.values():
                        self.color_map.update({x: col})

        self.video_mutex = QMutex()
        self.joint_db_mutex = QMutex()

        self.curr_sample_joints = None
        self.curr_sample_angles = None

    def read_cur_video(self):
        ret, frame = None, None

        if self.cap is None:
            video_path = self._set_video_name_from_curr_sample()

            if self.video_mutex.tryLock():
                self.cap = cv2.VideoCapture(video_path)
                self.curr_frame_time_sec = (
                    (1000 / 30) / 1000
                    if self.curr_frame_time_sec is None
                    else self.curr_frame_time_sec
                )

                self.curr_frame_time_ms = 1000 / self.cap.get(cv2.CAP_PROP_FPS)
                self.video_mutex.unlock()

            self.__reset_cap()

        else:
            curr_frame_pos = None

            if self.could_no_reset_cap:
                self.__reset_cap()

            if self.video_mutex.tryLock():
                curr_opts = self._get_opts_from_curr_vid()
                time.sleep(self.curr_frame_time_sec)
                ret, frame = self.cap.read()
                curr_frame_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                curr_msec_pos = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                if curr_frame_pos >= (curr_opts["end"] // self.curr_frame_time_ms):
                    self.cap.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        curr_opts["beg"] // self.curr_frame_time_ms,
                    )
                self.video_mutex.unlock()

            if ret is None and frame is None:
                return True, self.last_frame

            if not ret:
                raise RuntimeError(
                    f"bad video {self.all_videos_path[self.curr_video_idx]}"
                )

            self.__write_joints_2_im(frame, curr_msec_pos)

            frame = cv2.resize(frame, (1920, 1080))
            self.__write_info_2_im(frame, font_scale=1)

            if self.frame_angle is not None:
                frame = self.hconcat_resize_min([frame, self.frame_angle])

        self.last_frame = frame
        return ret, frame

    def _set_video_name_from_curr_sample(self):
        curr_opts = self._get_opts_from_curr_vid()
        vid_path, vid_name = ExtractMultipleVideos.read_vid_path_from_vpart(
            curr_opts["curr_folder"], curr_opts["talker_id"]
        )
        return vid_path

    @staticmethod
    def hex_2_rgb(h):
        h = h[1:]
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
        h_max = max(im.shape[0] for im in im_list)
        im_list_resize = [
            cv2.resize(
                im,
                (int(im.shape[1] * h_max / im.shape[0]), h_max),
                interpolation=interpolation,
            )
            for im in im_list
        ]
        return cv2.hconcat(im_list_resize)

    def change_to_sample_idx(self, idx):
        self.curr_sample_idx = idx
        self.__reset_cap_position()

    def prev_sample(self):
        self.curr_sample_idx = (
            self.curr_sample_idx - 1
            if self.curr_sample_idx != 0
            else len(self.all_samples_path_from_video)
        )
        self.__reset_cap_position()

    def next_sample(self):
        self.curr_sample_idx = (
            self.curr_sample_idx + 1
            if self.curr_sample_idx < len(self._all_samples_name) - 1
            else 0
        )
        self.__reset_cap()

    def prev_vid(self):
        self.curr_video_idx = (
            self.curr_video_idx - 1
            if self.curr_video_idx != 0
            else len(self.all_videos_path) - 1
        )
        self.__update_videos_n_paths()

    def next_vid(self):
        self.curr_video_idx = (
            self.curr_video_idx + 1
            if self.curr_video_idx < len(self.all_videos_path) - 1
            else 0
        )
        self.__update_videos_n_paths()

    def _get_opts_from_curr_vid(self):
        curr_opts_list = (
            self._all_samples_name[self.curr_sample_idx][0]
            .replace("\\", "/")
            .split("/")[-1]
            .split("---")
        )

        v_part = int(curr_opts_list[0])
        talker_id = int(curr_opts_list[-1][0])
        folder = self.all_persons_from_subtitle[
            self.all_persons_from_subtitle.v_part == v_part
        ].folder_name.values[0]
        curr_folder_path = os.path.join(self.db_path, folder)
        beg = int(curr_opts_list[2])
        end = int(curr_opts_list[3])
        sign = curr_opts_list[1]

        return dict(
            v_part=v_part,
            talker_id=talker_id,
            curr_folder=curr_folder_path,
            beg=beg,
            end=end,
            sign=sign,
        )

    def __reset_cap(self):
        self.curr_sample_joints = pd.read_csv(
            self._all_samples_name[self.curr_sample_idx][0]
        )
        self.curr_sample_joints = self.curr_sample_joints.applymap(
            PoseCentroidTracker.parse_npy_vec_str
        )
        cur_opts = self._get_opts_from_curr_vid()
        if self.video_mutex.tryLock():
            succ = self.cap.set(
                cv2.CAP_PROP_POS_FRAMES, cur_opts["beg"] // self.curr_frame_time_ms
            )
            self.video_mutex.unlock()
            self.could_no_reset_cap = False
            if not succ:
                raise RuntimeError(f"Could not set frame pos")
        else:
            self.could_no_reset_cap = True

    def __update_videos_n_paths(self):

        curr_opts_list = (
            self._all_samples_name[self.curr_sample_idx][0]
            .replace("\\", "/")
            .split("/")[-1]
            .split("---")
        )
        self.all_samples_path_from_video = list(
            filter(lambda x: "sample" in x, os.listdir(video_path))
        )
        for it, sample_name in enumerate(self.all_samples_path_from_video):
            name_to_add = "".join(sample_name.split("-")[-5:]) + " " + str(it)
            self.curr_samples_list.addItem(name_to_add)

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
        return "%d: %d: %d: %d" % (hours, minutes, seconds, milisec)

    def __write_info_2_im(self, im, font_scale=0.3):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (10, 200)
        font_color = (255, 255, 255)
        line_type = 2

        curr_opts = self._get_opts_from_curr_vid()
        for it, opt in enumerate(curr_opts.items()):
            text = (
                str(opt[1])
                if not isinstance(opt[1], float)
                else self.__convert_msec_2_hour_text(opt[1])
            )
            cv2.putText(
                im,
                text,
                (
                    bottom_left_corner_of_text[0],
                    bottom_left_corner_of_text[1] + it * 40,
                ),
                font,
                font_scale,
                font_color,
                line_type,
            )

    def __write_joints_2_im(self, im, frame_pos, radius=2, color=(255, 0, 255)):
        frame_search_res = self.curr_sample_joints.frame == int(frame_pos)
        if not any(frame_search_res.values):
            return

        joints_used = []
        for k in self.joints_angle_2_use:
            keys = k.split("-")
            if "left" in keys or "right" in keys:
                keys = [
                    keys[0] + "-" + keys[1],
                    keys[2] + "-" + keys[3],
                    keys[4] + "-" + keys[5],
                ]

            joints_used.extend(keys)

        joints_at_frame = self.curr_sample_joints[frame_search_res]
        for joint_name in joints_used:
            joint = joints_at_frame[joint_name].values[0]
            try:
                cv2.circle(im, tuple(map(int, joint[:2])), radius, color, 1)
            except BaseException:
                continue

        joints_at_frame_df = self.curr_sample_joints[frame_search_res]
        for key in self.joints_angle_2_use:
            keys = key.split("-")
            if "left" in keys or "right" in keys:
                keys = [
                    keys[0] + "-" + keys[1],
                    keys[2] + "-" + keys[3],
                    keys[4] + "-" + keys[5],
                ]

            color = self.hex_2_rgb(self.color_map[key])
            joint_0 = joints_at_frame_df[keys[0]].values[0][:2]
            joint_1 = joints_at_frame_df[keys[1]].values[0][:2]
            joint_2 = joints_at_frame_df[keys[2]].values[0][:2]

            joint_0_acc = joints_at_frame_df[keys[0]].values[0][2]
            joint_1_acc = joints_at_frame_df[keys[1]].values[0][2]
            joint_2_acc = joints_at_frame_df[keys[2]].values[0][2]

            # if joint_0_acc < 0.4 or joint_1_acc < 0.4 or joint_2_acc < 0.4:
            #     continue

            try:
                cv2.line(
                    im, tuple(map(int, joint_1)), tuple(map(int, joint_0)), color=color
                )
                cv2.line(
                    im, tuple(map(int, joint_1)), tuple(map(int, joint_2)), color=color
                )
            except BaseException:
                continue


if __name__ == "__main__":
    app = QApplication(sys.argv)

    w = App(
        th_cls=View2VideoDB,
        all_videos_df_path="all_videos.csv",
        libras_corpus_db_path="D:/gdrive/",
        front_view_db_path="../sign_db_front_view",
        vid_sync="vid_sync.csv",
        all_persons_from_subtitle="all_persons_from_subtitle.csv",
    )
    w.show()

    sys.exit(app.exec_())
