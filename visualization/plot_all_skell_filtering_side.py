# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.manifold import TSNE
from libras_classifiers.librasdb_loaders import DBLoader2NPY
from libras_classifiers.make_angle_df_from_xy import make_angle_df_from_xy
from pose_extractor.all_parts import *


def label_point(x, y, val, ax):
    a = pd.DataFrame({'x': x, 'y': y, 'val': val})
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))


plt.rcParams['figure.figsize'] = [16.0, 9.0]
plt.rcParams['figure.dpi'] = 1080 // 9
tsne = TSNE(n_components=2, random_state=0)

body_angle_pairs_to_7 = [[1, 2, 3], [2, 3, 4], [1, 5, 6], [5, 6, 7], [2, 1, 5]]
body_angle_pairs_to_7 = [[INV_BODY_PARTS[x[0]], INV_BODY_PARTS[x[1]], INV_BODY_PARTS[x[2]]]
                         for x in body_angle_pairs_to_7]
hand_angle_pair = [[0, 2, 4], [0, 6, 8], [0, 10, 12], [0, 14, 16], [0, 18, 20]]
hand_angle_pair = [[INV_HAND_PARTS[x[0]], INV_HAND_PARTS[x[1]], INV_HAND_PARTS[x[2]]]
                   for x in hand_angle_pair]

db_angle = DBLoader2NPY(db_path='../Libras-db-folders', batch_size=1, angle_pose=True, no_hands=False)
db_xy = DBLoader2NPY(db_path='../Libras-db-folders', batch_size=1, angle_pose=False, no_hands=False)

db_angle_x, db_angle_y = db_angle.batch_load_samples(list(range(0, db_angle.db_length())), as_npy=False)
db_xy_x, db_xy_y = db_xy.batch_load_samples(list(range(0, db_xy.db_length())), as_npy=False)
angles_keys = ['Neck-RShoulder-RElbow',
               'RShoulder-RElbow-RWrist',
               'Neck-LShoulder-LElbow',
               'LShoulder-LElbow-LWrist',
               'RShoulder-Neck-LShoulder']
# %%
checking_each_df_is_equals = []
for angle_sample, xy_sample in tqdm(zip(db_angle_x, db_xy_x), total=len(db_xy_x)):
    if xy_sample['Neck'].iloc[0][0] > 200.0:
        curr_angle_df = make_angle_df_from_xy(xy_sample, no_hands=True, body_angles=body_angle_pairs_to_7)
        curr_angle_df = curr_angle_df.fillna(0.0)
        res = np.isclose(curr_angle_df[angles_keys].values, angle_sample[angles_keys].values).all()
        if not res:
            print(f' {curr_angle_df[angles_keys].values}, \n {angle_sample[angles_keys].values}')
        checking_each_df_is_equals.append(res)

print(all(checking_each_df_is_equals))


# %%
amount_frames_at_longest = db_angle.longest_sample
color_array = ['#e53242', '#ffb133', '#3454da', '#ddc3d0', '#005a87', '#df6722', '#00ffff',
               '#b7b7b7', '#ddba95', '#ffb133', '#4b5c09', '#00ff9f', '#e2f4c7', '#a2798f',
               '#8caba8', '#ffb3ba', '#ffdfba', '#ffffba', '#baffc9', '#ff8000', '#8f139f']

color_map = {}
for x in angles_keys:
    if x not in color_map:
        for col in color_array:
            if col not in color_map.values():
                color_map.update({x: col})


count = 0
for angle_sample, xy_sample, sample_name in tqdm(zip(db_angle_x, db_xy_x, db_angle.samples_path), total=len(db_xy_x)):
    if xy_sample['Neck'].iloc[0][0] > 200.0:

        fig_name = sample_name[0].replace('\\', '/').split('/')[-1][:-4]

        #plt.Figure(dpi=1080, figsize=(16, 9), tight_layout=True)
        plt.ylim(-4, 4)
        plt.xticks(np.arange(-4, 4, 0.2))
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

        x_points = []
        y_points = []
        c_points = []
        for row in angle_sample.iterrows():
            row: pd.Series = row[1]
            curr_y = row[angles_keys].values
            c_points.extend([color_map[x] for x in angles_keys])
            x_points.extend([row.frame] * (len(angles_keys)))
            y_points.extend(curr_y)

        chart = plt.scatter(x_points, y_points, c=c_points)

        # ax.vlines(frame_pos, ymin=-4, ymax=4)

        for k in angles_keys:
            color = color_map[k]
            y = angle_sample[k].values
            x = angle_sample.frame.to_list()
            plt.plot(x, y, c=color, label=k)

        plt.legend(bbox_to_anchor=(1.2, 1.1), loc='upper right')

        plt.savefig(f'figs/{fig_name}.pdf')
        plt.show()

# %%
all_x = []
all_y = []
all_samples_idx = []
all_idx = []
enumerator = list(range(0, len(db_xy_x)))
iterator_samples = zip(db_angle_x, db_xy_x, db_xy_y,  enumerator, db_angle.samples_path)
for angle_sample, xy_sample, sample_y, it, sample_name in tqdm(iterator_samples, total=len(db_xy_x)):
    if xy_sample['Neck'].iloc[0][0] > 200.0:
        sample_id_aux = []
        for it_frame, frame in enumerate(angle_sample.iterrows()):
            all_x.append(frame[1][angles_keys].values)
            all_y.append(np.argmax(sample_y))
            all_idx.append(it)
            sample_id_aux.append(it)
        all_samples_idx.append(sample_id_aux)

res = tsne.fit_transform(all_x)

df = pd.DataFrame(dict(x=[x[0] for x in res], y=[x[1] for x in res], sample_idx=[x for x in all_idx],
                       cls=all_y))

np.random.seed(5)
samples_ids = np.random.randint(0, 120, size=10).tolist() + np.random.randint(120, 240, size=10).tolist()
a = df['sample_idx'].isin(samples_ids)
df2 = df[a]

single_sample_idx = df2.sample_idx.unique().tolist()
single_sample_list = []
for sample_id in single_sample_idx:
    sample = df2[df2['sample_idx'] == sample_id]
    single_sample_list.append(sample.iloc[0])

plt.tight_layout()
chart = sns.scatterplot(data=df2, x='x', y='y', style='cls')
chart.legend_.remove()
# label_point(x=[x.x for x in single_sample_list], y=[x.y for x in single_sample_list],
#              val=[x.sample_idx for x in single_sample_list], ax=chart)
plt.show()

# %%
db_path = './all_videos.csv'
sign_db = pd.read_csv(db_path)

signs_name = sign_db.sign.unique()
count_signs = []
for sign in tqdm(signs_name):
    count_signs.append(dict(name=sign,
                            count=sign_db[sign_db.sign == sign].sign.count()))

sorted_signs = sorted(count_signs, reverse=True, key=lambda x: x['count'])
sorted_sings_count_mode = []
for x in sorted_signs:
    if x['count'] < 30:
        continue
    sorted_sings_count_mode.extend([x['name']] * int(x['count']))