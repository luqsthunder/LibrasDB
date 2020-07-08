from pose_extractor.openpose_extractor import OpenposeExtractor
import pandas as pd
import tqdm

extractor = OpenposeExtractor('../openpose')
centroids_df = pd.read_csv('centroids.csv')

all_videos = pd.read('all_videos.csv')
signs_names = all_videos.sign.unique()
count_signs = []
for sign in tqdm(signs_names):
    count_signs.append(dict(name=sign,
                            count=all_videos[all_videos.sign == sign].sign.count()))

sorted_signs = sorted(count_signs, reverse=True, key=lambda x: x['count'])
print(sorted_signs)