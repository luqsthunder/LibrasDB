import os
import tkinter
import threading
import PIL.Image, PIL.ImageTk
import pandas as pd
import numpy as np
import cv2 as cv

from visualization.db_player import ViewDBCutVideos


class CheckBoxes(tkinter.Frame):

    def __init__(self, window_parent):
        super(CheckBoxes, self).__init__()
        self.window_parent = window_parent
        self.check_box_list = []
        self.check_box_var_list = []

        self.__init_ui()

    def __init_ui(self):
        self.check_box_var_list = [tkinter.IntVar() for _ in range(8)]
        self.check_box_list = [
        tkinter.Checkbutton(self, text='Legenda 1: Pessoa: 1', variable=self.check_box_var_list[0],
                            command=self.fn_ck).grid(row=0, column=0),
            tkinter.Checkbutton(self, text='Legenda 2: Pessoa: 1', variable=self.check_box_var_list[1],
                                command=self.fn_ck).grid(row=0, column=1),
            tkinter.Checkbutton(self, text='Legenda 1: Pessoa: 2',  variable=self.check_box_var_list[2],
                                command=self.fn_ck).grid(row=1, column=0),
            tkinter.Checkbutton(self, text='Legenda 2: Pessoa: 2',  variable=self.check_box_var_list[3],
                                command=self.fn_ck).grid(row=1, column=1),

            tkinter.Checkbutton(self, text='Pessoa 1: video: 1', variable=self.check_box_var_list[4],
                                command=self.fn_ck).grid(row=2, column=0),
            tkinter.Checkbutton(self, text='Pessoa 2: video: 1', variable=self.check_box_var_list[5],
                                command=self.fn_ck).grid(row=2, column=1),
            tkinter.Checkbutton(self, text='Pessoa 1: video: 2', variable=self.check_box_var_list[6],
                                command=self.fn_ck).grid(row=3, column=0),
            tkinter.Checkbutton(self, text='Pessoa 2: video: 2', variable=self.check_box_var_list[7],
                                command=self.fn_ck).grid(row=3, column=1)]

    def uncheck_tuples(self, tp):
        if self.check_box_var_list[tp[0]].get() == 1:
            self.check_box_var_list[tp[1]].set(0)
        elif self.check_box_var_list[tp[1]].get() == 1:
            self.check_box_var_list[tp[0]].set(0)

    def fn_ck(self):
        for it in range(0, 8, 2):
            self.uncheck_tuples((it, it + 1))

    def generate_row_df_from_res(self):
        return pd.DataFrame()


class NextPrev(tkinter.Frame):

    def __init__(self, checkboxes):
        super(NextPrev, self).__init__()

        self.btn_prev = tkinter.Button(self)
        self.btn_nect = tkinter.Button(self)
        self.checkboxes = checkboxes

    def read_curr_csv(self):
        pass

    def prev_fn(self):
        print(self.checkboxes.generate_row_df_from_res())

    def next_fn(self):
        print(self.checkboxes.generate_row_df_from_res())


class MakeDB3Videos:

    def __init__(self, vid_db_path, vid_df, all_video_df, window, window_title):
        self.window = window
        self.window.title = window_title
        self.vid_db_path = vid_db_path
        self.vid_df = pd.read_csv(vid_df) if os.path.exists(vid_df) else pd.DataFrame()
        self.all_video_df = all_video_df if isinstance(all_video_df, pd.DataFrame) else pd.read_csv(all_video_df)

        self.canvas = tkinter.Canvas(self.window, width=600, height=800)
        self.canvas.grid(row=0, column=0)
        self.check_boxes = CheckBoxes(self.window)
        self.check_boxes.grid(row=0, column=1)

        self.next_prev = NextPrev(self.check_boxes)
        self.next_prev.grid(row=1, column=0)

        self.cur_vid_pos = 0
        self.cur_beg = None
        self.cur_end = None
        self.cv_caps = [None, None, None, None]
        self.photo = None

        self.next_folder_btn = tkinter.Button(window, text='Next', width=50, command=self.next_folder_btn_fn)

        self.delay = 15
        self.update_caps()
        self.run()
        self.window.mainloop()

    def update_caps(self):
        """
        Atualiza os videos captures para o primeiro sinal do video atual.
        """
        curr_folder_name_in_df = self.all_video_df.folder_name.iloc[self.cur_vid_pos]

        curr_folder_name = self.all_video_df.folder_name.iloc[self.cur_vid_pos].replace('\\', '/').split('/')[:-1]
        curr_folder_name = os.path.join(*curr_folder_name)
        curr_folder_path = os.path.join(self.vid_db_path, curr_folder_name)
        videos_path_at_curr_folder = list(filter(lambda x: '.mp4' in x, os.listdir(curr_folder_path)))
        videos_path_at_curr_folder = list(map(lambda x: os.path.join(self.vid_db_path, x), videos_path_at_curr_folder))

        for cv_cap in self.cv_caps:
            if cv_cap is not None:
                cv_cap.release()

        self.cv_caps = [cv.VideoCapture(x) for x in videos_path_at_curr_folder]

        self.cur_beg = self.all_video_df[self.all_video_df.folder_name == curr_folder_name_in_df].iloc[0].beg // 30
        self.cur_end = self.all_video_df[self.all_video_df.folder_name == curr_folder_name_in_df].iloc[0].end // 30

        for cv_cap in self.cv_caps:
            cv_cap.set(cv.CAP_PROP_POS_FRAMES, (self.cur_beg // 30))

    def next_folder_btn_fn(self):
        pass

    def reset_caps_2_beg_frame(self):
        caps = list(filter(lambda x: x is not None, self.cv_caps))
        for it in range(len(caps)):
            if caps[it].get(cv.CAP_PROP_POS_FRAMES) >= self.cur_end:
                caps[it].set(cv.CAP_PROP_POS_FRAMES, self.cur_beg)

    def run(self):
        self.reset_caps_2_beg_frame()

        frames = list(filter(lambda x: x is not None, [x.read()[1] for x in self.cv_caps]))
        if len(frames) > 0:
            frames = ViewDBCutVideos.hconcat_resize_min(frames)
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frames))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.run)


if __name__ == '__main__':
    mk = MakeDB3Videos('D:/gdrive/LibrasCorpus/', './3video_db.csv',
                       pd.read_csv('all_videos.csv'), tkinter.Tk(), 'Make Video DB')