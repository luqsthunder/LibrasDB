from pose_extractor.openpose_extractor import OpenposeExtractor
from pose_extractor.pose_centroid_tracker import PoseCentroidTracker
from pose_extractor.df_utils import update_xy_pose_df_single_person
from tqdm import tqdm
import os
import unicodedata as ud
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd

def process_single_sample(extractor, curr_video, beg, end, person_needed,
                          left_person, first_x_mid):
    curr_video.set(cv.CAP_PROP_POS_FRAMES, beg)
    curr_frame = int(curr_video.get(cv.CAP_PROP_POS_FRAMES))
    in_need_id = 0 if person_needed == left_person else 1
    df_cols = ['frame'] + pose_tracker.body_parts + \
              ['l-' + x for x in pose_tracker.hands_parts] + \
              ['r-' + x for x in pose_tracker.hands_parts]

    video_df = pd.DataFrame(columns=df_cols)
    debug_centroids = []
    for _ in tqdm(range(end - beg)):
        ret, frame = curr_video.read()
        if not ret:
            return None
        dt = extractor.extract_poses(frame)

        curr_frame = int(curr_video.get(cv.CAP_PROP_POS_FRAMES))
        persons_id = pose_tracker.filter_persons_by_x_mid(dt, first_x_mid)
        # pose_to_centroid = dt.poseKeypoints[persons_id[in_need_id][0]]
        # curr_centroid = \
        #     pose_tracker.make_xy_centroid(pose_to_centroid)
        # debug_centroids.append(curr_centroid)
        video_df = update_xy_pose_df_single_person(dt, video_df,
                                                   curr_frame,
                                                   persons_id[in_need_id][0],
                                                   persons_id[in_need_id][1],
                                                   pose_tracker.body_parts,
                                                   pose_tracker.hands_parts)

    # mean_x_cent = 0
    # for x in debug_centroids:
    #     mean_x_cent += x[0] if x is not None else 0
    # mean_x_cent = mean_x_cent / len(debug_centroids)
    # print(mean_x_cent)
    return video_df


db_path = '/media/lucas/Others/LibrasCorpus/Santa Catarina/'
if db_path == '':
    raise RuntimeError('esqueceu de setar o db_path')

# pose_extractor = OpenposeExtractor('../openpose')
centroids_df = pd.read_csv('centroids.csv')
pose_tracker = PoseCentroidTracker('all_videos.csv', 'db_path',
                                   openpose_path='../',
                                   centroids_df_path='centroids.csv')

all_videos = pd.read_csv('all_videos.csv')[['sign', 'beg', 'end', 'folder_name',
                                           'talker_id', 'hand']]
signs_names = all_videos.sign.unique()
count_signs = []
# for sign in tqdm(signs_names):
#     amount_sign = all_videos[all_videos.sign == sign].sign.count()
#
#     count_signs.append(dict(name=sign,
#                             count=amount_sign))
#
# sorted_signs = sorted(count_signs, reverse=True, key=lambda x: x['count'])
needed_signs = ['IX(eu)', 'E(então)', 'IX(você)', 'E(positivo)']

no_unicode_need_signs = ['IX(eu)', 'E(entao)', 'IX(voce)', 'E(positivo)']
no_unicode_need_signs = {needed_signs[it]: x
                         for it, x in enumerate(no_unicode_need_signs)}
# [x['name'] for x in sorted_signs[:4]]
# print(needed_signs)

bad_video_df = pd.read_csv('bad_video.csv')

folders_name_map = {x.split('v')[-1]: x
                    for x in os.listdir(os.path.join(db_path, 'Inventario Libras'))}

centroid_folder_names = sorted(list(centroids_df.folder.unique()))
processed_signs_count = {x: 0 for x in needed_signs}
for f_name in tqdm(centroid_folder_names, desc='folders'):
    print(f'\n beg video {f_name}: ')

    v_part = f_name.split(' v')[-1].split('/')[0]
    new_folder_name = folders_name_map[v_part]
    if f_name in bad_video_df.folder.unique():
        continue

    curr_x_mid, curr_left_person_sub_id, curr_right_person_sub_id = \
        pose_tracker.subdivide_2_persons_scene(f_name)

    needed_sings_in_video = all_videos[(all_videos['folder_name'] == f_name) &
                                       (all_videos['sign'].isin(needed_signs))]

    list_video_path = os.path.join(db_path, 'Inventario Libras',
                                   new_folder_name)
    list_video_path = [os.path.join(list_video_path, x)
                       for x in os.listdir(list_video_path)]
    video_path = list(filter(lambda x: '1.mp4' in x, list_video_path))
    if len(video_path) == 0:
        continue
    video_path = video_path[0]

    sign_path = os.path.join(db_path, 'Inventario Libras', new_folder_name)
    video = cv.VideoCapture(video_path)

    if not os.path.exists(video_path):
        print(f'video not exists {video_path}')
        continue

    for it, sign in tqdm(enumerate(needed_sings_in_video.iterrows()),
                         total=needed_sings_in_video.shape[0], desc='signs'):
        sign = sign[1]
        if processed_signs_count[sign.sign] >= 40:
            continue

        sign_name = no_unicode_need_signs[sign.sign]
        sample_name = f'sample-xy-500ms-{new_folder_name}-{sign_name}-beg-{sign.beg}-end-{sign.end}.csv'
        sample_path = os.path.join(sign_path, sample_name)
        if os.path.exists(sample_path):
            processed_signs_count[sign.sign] += 1
            continue

        print(f'\n beg sign {sign.sign}')
        #beg = sign.beg - 500 if sign.beg - 500 > 0 else 0
        df = process_single_sample(pose_tracker.pose_extractor,
                                   video, sign.beg-500, sign.end+500,
                                   sign.talker_id, curr_left_person_sub_id,
                                   curr_x_mid)
        #print(df)
        if df is not None:
            df.to_csv(sample_path)
            processed_signs_count[sign.sign] += 1

    video.release()

