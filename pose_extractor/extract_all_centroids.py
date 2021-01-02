import os
import pandas as pd
from tqdm import tqdm
from pose_extractor.pose_centroid_tracker import PoseCentroidTracker


all_videos = pd.read_csv('all_videos.csv')

tracker = PoseCentroidTracker('all_videos.csv', openpose_path='../../Libraries/repos/openpose')

if os.path.exists('centroids.csv'):
    centroids_df = pd.read_csv('centroids.csv')
else:
    centroids_df = pd.DataFrame(columns=['folder', 'talker_id', 'centroid'])
    centroids_df.to_csv('centroids.csv', index=False)

if os.path.exists('bad_video.csv'):
    bad_video_df = pd.read_csv('bad_video.csv')
else:
    bad_video_df = pd.DataFrame(columns=['folder'])

centroid_df_list = []
bad_videos = 0
bad_videos_list = []
pbar2 = tqdm()
folder_names = list(all_videos.folder_name.unique())

all_video_pbar = tqdm(total=len(folder_names), desc='all videos centroid')
for it, f_name in enumerate(folder_names):

    if f_name in centroids_df.folder.unique():
        continue

    if f_name in bad_video_df.folder.unique():
        continue
    try:
        # print(f'beging to process: {it} at video {f_name}')
        res = tracker.retrive_persons_centroid_from_sign_df(f_name,
                                                            'D:/gdrive/LibrasCorpus/',
                                                            pbar2)
        # print(f_name, '\n', res[['talker_id', 'centroid']])
        if not res.empty:
            centroid_df_list.append(res)
            res.to_csv('centroids.csv', mode='a', header=None, index=False)
            centroids_df = centroids_df.append(res)
        else:
            print('empty', f_name)
            print(f'beging to process: {it} at video {f_name}')
            break
    except (RuntimeError, IndexError) as e:
        print(e)
        print(f'beging to process: {it} at video {f_name}')
        bad_videos += 1
        bad_video_df = bad_video_df.append(pd.DataFrame(data=dict(folder=[f_name])))
        bad_video_df.to_csv('bad_video.csv')
        print(f'bad {it} -> {f_name}')

    all_video_pbar.update(1)
    all_video_pbar.refresh()