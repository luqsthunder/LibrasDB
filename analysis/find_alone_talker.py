import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import cv2 as cv
import face_recognition

# %%
all_videos = pd.read_csv('all_videos.csv')

print(all_videos.keys())
folders = all_videos.folder_name.unique().tolist()


# %%

def mili_to_minutes(mili):
    millis=mili
    millis = int(millis)
    seconds = (millis/1000) % 60
    milisec = (seconds % 1) * 1000
    seconds = int(seconds)
    minutes=(millis/(1000*60)) % 60
    minutes = int(minutes)
    return "min=> %03d: seg=> %03d: msec => %04d" % (minutes, seconds, milisec)


# %%
def find_where_signaling_is_quiet(all_video_df, folder_name, talker_id, pbar=None):
    """
    Acha onde um sinalizador de uma legenda na pasta do projeto não fala nada.

    parameters
    ----------
    all_video_df: pd.DataFrame
        base de dados.

    folder_name: str
        nome da pasta na base de dados.

    talker_id: int
        id do sinalizador na legenda

    returns: List
        holes, lista com os locais onde o sinalizador não fala.
    """
    talker_1_signs = all_video_df[all_video_df.folder_name == folder_name]
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


# %%
signer_1_holes = find_where_signaling_is_quiet(all_videos, folders[0], 1)
signer_2_holes = find_where_signaling_is_quiet(all_videos, folders[0], 2)


# %%
s1_minutes_holes = [dict(beg=mili_to_minutes(x['beg']), end=mili_to_minutes(x['end']), hole=mili_to_minutes(x['hole']))
                    for x in signer_1_holes]
s2_minutes_holes = [dict(beg=mili_to_minutes(x['beg']), end=mili_to_minutes(x['end']), hole=mili_to_minutes(x['hole']))
                    for x in signer_2_holes]
# %%
db_path = 'D:/gdrive/LibrasCorpus/Santa Catarina/Inventario Libras'
db_folders_path = [os.path.join(db_path, x) for x in os.listdir(db_path)]

# %%
all_videos_in_folder = [cv.VideoCapture(os.path.join(db_folders_path[0], x))
                        for x in os.listdir(db_folders_path[0]) if '.mp4' in x]
for it in range(len(all_videos_in_folder)):
    all_videos_in_folder[it].set(cv.CAP_PROP_POS_FRAMES, 30 * 60 * 1)


# %%
all_left_whites_count = []
all_lefts = {'Left': 0, 'Right': 0}
faces = None
for curr_hole_pos in tqdm(range(len(signer_1_holes))):
    # curr_hole_pos =
    fps = int(all_videos_in_folder[0].get(cv.CAP_PROP_FPS))
    beg_frame_time = signer_1_holes[curr_hole_pos]['beg']
    res = all_videos_in_folder[0].set(cv.CAP_PROP_POS_MSEC, beg_frame_time)
    end_frame_time = signer_1_holes[curr_hole_pos]['end']

    last_frame = None
    key = None

    ret, frame = all_videos_in_folder[0].read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    curr_video_pos_msec = all_videos_in_folder[0].get(cv.CAP_PROP_POS_MSEC)
    last_msec = curr_video_pos_msec

    if faces is None:
        faces = face_recognition.face_locations(frame)
        break

    left_most_white = []
    while curr_video_pos_msec <= end_frame_time:

        (thresh, frame) = cv.threshold(frame, 127, 255, cv.THRESH_BINARY)
        curr_frame = frame - last_frame if last_frame is not None else frame
        last_frame = frame
        (thresh, curr_frame) = cv.threshold(curr_frame, 127, 255, cv.THRESH_BINARY)

        left = curr_frame[:, :curr_frame.shape[1] // 2]
        right = curr_frame[:, curr_frame.shape[1] // 2:]

        left_most_white.append(np.count_nonzero(left > 1) < np.count_nonzero(right > 1))

        # cv.imshow('window', cv.vconcat([curr_frame, frame]))
        # key = cv.waitKey(fps)
        # if key == 27:  # exit on ESC
        #     break

        ret, frame = all_videos_in_folder[0].read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        curr_video_pos_msec = all_videos_in_folder[0].get(cv.CAP_PROP_POS_MSEC)
        last_msec = curr_video_pos_msec

    unique, counts = np.unique(np.array(left_most_white), return_counts=True)
    count_whites = dict(zip(unique, counts))
    all_lefts['Left'] += count_whites[True] if True in count_whites else 0

    all_lefts['Right'] += count_whites[False] if False in count_whites else 0

    all_left_whites_count.append(count_whites)
    #
    # if key == 27:
    #     break
    #
    # key = cv.waitKey(-1)
    #
    # if key == 27:
    #     break

#cv.destroyAllWindows()


# %%
def find_all_left_signaler(all_vid_df):
    folders = all_vid_df.folder_name.unique().tolist()
    db_path = '/media/usuario/Others/gdrive'

    for k, f in tqdm(enumerate(folders[0:16])):
        f_path = os.path.join(db_path, f)
        vid = cv.VideoCapture(f_path)

        curr_signer_num = 1
        signer_holes = find_where_signaling_is_quiet(all_vid_df, f, 1)
        if signer_holes is None:
            curr_signer_num = 2
            signer_holes = find_where_signaling_is_quiet(all_vid_df, f, 2)
        elif len(signer_holes) == 0:
            curr_signer_num = 2
            signer_holes = find_where_signaling_is_quiet(all_vid_df, f, 2)

        if len(signer_holes) == 0:
            # aki eu testo o caso de ser vazio.
            continue

        res = find_left_signaler_in_one_video(vid, signer_holes)
        print(res, curr_signer_num)
        vid.release()


def find_left_signaler_in_one_video(vid, holes):
    all_left_whites_count = []
    all_left_in_video = {'Left': 0, 'Right': 0}
    x_middle = None

    for c_hole_pos in range(len(holes)):
        fps = int(vid.get(cv.CAP_PROP_FPS))
        beg_frame_time = holes[c_hole_pos]['beg']
        res = vid.set(cv.CAP_PROP_POS_MSEC, beg_frame_time)
        if not res:
            raise RuntimeError('video cant set position')

        end_frame_time = holes[c_hole_pos]['end']

        last_frame = None

        ret, frame = vid.read()
        if not ret:
            print(mili_to_minutes(beg_frame_time))
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        curr_video_pos_msec = vid.get(cv.CAP_PROP_POS_MSEC)

        if x_middle is None:
            x_middle = find_middle_xpoint_in_faces_bbox(frame)

        left_most_white = []
        while curr_video_pos_msec <= end_frame_time:
            (thresh, frame) = cv.threshold(frame, 127, 255, cv.THRESH_BINARY)
            curr_frame = frame - last_frame if last_frame is not None else frame
            last_frame = frame
            (thresh, curr_frame) = cv.threshold(curr_frame, 127, 255, cv.THRESH_BINARY)

            left = curr_frame[:, int(x_middle):]
            right = curr_frame[:, :int(x_middle)]

            left_most_white.append(np.count_nonzero(left > 1) < np.count_nonzero(right > 1))

            # cv.imshow('window',  left)
            # key = cv.waitKey(-1)
            # if key == 27:  # exit on ESC
            #     break

            ret, frame = vid.read()
            if not ret:
                print(curr_video_pos_msec)
                break
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            curr_video_pos_msec = vid.get(cv.CAP_PROP_POS_MSEC)

        unique, counts = np.unique(np.array(left_most_white), return_counts=True)
        white_count = dict(zip(unique, counts))
        all_left_in_video['Left'] += white_count[True] if True in white_count else 0

        all_left_in_video['Right'] += white_count[False] if False in white_count else 0

        all_left_whites_count.append(white_count)

    return all_left_in_video


def find_middle_xpoint_in_faces_bbox(frame):
    faces = face_recognition.face_locations(frame, number_of_times_to_upsample=1, model='cnn')
    middle_xpoint = (np.abs(faces[0][3] - faces[1][3]) / 2) + min([faces[0][3], faces[1][3]])
    return middle_xpoint


find_all_left_signaler(all_videos)

# %%
tk_1_signs = all_videos[all_videos.folder_name == folders[11]]
print(tk_1_signs.shape)
tk_1_signs = tk_1_signs[tk_1_signs.talker_id == 2]
tk_1_signs = tk_1_signs[tk_1_signs.hand == 'D']
print(tk_1_signs.shape)
