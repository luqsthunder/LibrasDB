import cv2
import sys
import pandas as pd
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
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
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class App(QWidget):
    label = None
    title = 'Debugger DB PLayer'
    next_btn = None
    prev_btn = None
    pause_btn = None
    video_name_label = None
    th = None

    def __init__(self):
        super().__init__()
        self.init_ui()

    @pyqtSlot(QImage)
    def set_image(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def init_ui(self):
        self.setWindowTitle(self.title)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.next_btn = QPushButton('next')
        self.prev_btn = QPushButton('prev')
        self.pause_btn = QPushButton('pause')

        self.pause_btn.clicked.connect(self.click_pause_btn)
        # self.prev_btn.clicked.connect(self.click_prev_btn)
        # self.next_btn.clicked.connect(self.click_next_btn)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.prev_btn)
        hbox.addWidget(self.pause_btn)
        hbox.addWidget(self.next_btn)

        vbox = QVBoxLayout()
        vbox.addStretch(1)

        # self.resize(1366, 768)
        # create a label
        self.label = QLabel(self)
        self.label.resize(1366, 768)

        self.th = OCVVideoThread(self)

        vbox.addWidget(self.label)

        vbox.addLayout(hbox)

        self.th.changePixmap.connect(self.set_image)
        self.th.start()

        self.setLayout(vbox)
        self.show()

    def click_pause_btn(self):
        self.th.pause = not self.th.pause


class ViewDBCutVideos(OCVVideoThread):

    def __init__(self, all_videos_df_path):
        vid_df = pd.read_csv(all_videos_df_path)

    def read_cur_video(self):
        pass

    def prev_vid(self):
        pass

    def next_vid(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = App()
    w.show()

    sys.exit(app.exec_())
