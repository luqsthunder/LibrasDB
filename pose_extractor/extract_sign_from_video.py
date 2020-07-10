from pose_extractor.openpose_extractor import OpenposeExtractor
from pose_extractor.pose_centroid_tracker import PoseCentroidTracker
from pose_extractor.df_utils import update_xy_pose_df_single_person
from tqdm import tqdm
import os
import cv2 as cv
import pandas as pd


def process_single_sample(extractor, curr_video, beg, end, person_needed,
                          left_person):
    video.set(cv.CAP_PROP_POS_FRAMES, beg)
    in_need_id = 0 if person_needed == left_person else 1
    df_cols = ['frame'] + pose_tracker.body_parts + \
              ['l' + x for x in pose_tracker.hands_parts] + \
              ['r' + x for x in pose_tracker.hands_parts]

    video_df = pd.DataFrame(columns=df_cols)
    for _ in tqdm(range(end - beg), desc='processing video'):
        ret, frame = curr_video.read()
        if not ret:
            return None
        dt = extractor.extract_poses(frame)
        curr_frame = int(curr_video.get(cv.CAP_PROP_POS_FRAMES))
        persons_id = pose_tracker.filter_persons_by_x_mid(dt)
        video_df = update_xy_pose_df_single_person(dt, video_df,
                                                   curr_frame,
                                                   persons_id[in_need_id][0],
                                                   persons_id[in_need_id][1],
                                                   pose_tracker.body_parts,
                                                   pose_tracker.hands_parts)
    return video_df

db_path = ''
if db_path == '':
    raise RuntimeError('esqueceu de setar o db_path')

pose_extractor = OpenposeExtractor('../openpose')
centroids_df = pd.read_csv('centroids.csv')

all_videos = pd.read('all_videos.csv')
signs_names = all_videos.sign.unique()
count_signs = []
for sign in tqdm(signs_names):
    amount_sign = all_videos[all_videos.sign == sign].sign.count()

    count_signs.append(dict(name=sign,
                            count=amount_sign))

sorted_signs = sorted(count_signs, reverse=True, key=lambda x: x['count'])
needed_signs = [x.keys() for x in sorted_signs[:4]]

pose_tracker = PoseCentroidTracker('all_videos.csv', 'db_path',
                                   centroids_df_path='centroids.csv')

bad_video_df = pd.read_csv('bad_video.csv')

centroid_folder_names = sorted(list(centroids_df.folder.unique()))
for f_name in tqdm(centroid_folder_names, desc='folders'):

    if f_name in bad_video_df.folder.unique():
        continue

    curr_x_mid, curr_left_person_sub_id, curr_right_person_sub_id = \
        pose_tracker.subdivide_2_persons_scene(f_name)

    needed_sings_in_video = all_videos[all_videos['folder_name'] == f_name &
                                       all_videos['sign'] == needed_signs]
    video_path = os.path.join(db_path, f_name)
    sign_path = f_name.split('/')[:-1]
    sign_path = os.path.join(db_path, sign_path)
    video = cv.VideoCapture()
    for it, sign in tqdm(enumerate(needed_sings_in_video),
                         total=len(needed_sings_in_video), desc='signs'):

        only_folder_name = f_name.split('/')
        only_folder_name = only_folder_name[len(only_folder_name) - 2]
        sign_name = sign.sign
        sample_name = f'sample-{only_folder_name}-{sign_name}-{it}.csv'
        sample_path = os.path.join(sign_path, sample_name)
        if os.path.exists(sample_path):
            continue

        print('processing')
        # df = process_single_sample(pose_extractor, video, sign.beg, sign.end,
        #                            sign.talker_id, curr_left_person_sub_id)
        #
        # if df is not None:
        #     df.to_csv(sample_path)

    video.release()

