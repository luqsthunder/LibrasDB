# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# %%
all_videos = pd.read_csv('all_videos.csv')
all_signs = all_videos.sign.unique().tolist()

# %%
m_sec_threshold = 5 # 1 segundo de distancia entre os sinais ainda vou considerar proximo
all_videos_folders = all_videos.folder.unique().tolist()
close_signs_mat = np.zeros((len(all_signs), len(all_signs)))

for folder in tqdm(all_videos_folders):
    talkers = all_videos[all_videos['folder'] == folder].talker_id.unique().tolist()
    for talker_id in talkers:
        all_samples_from_curr_folder = all_videos[all_videos['folder'] == folder]
        all_samples_from_curr_folder = \
            all_samples_from_curr_folder[all_samples_from_curr_folder['talker_id'] == talker_id]
        for it in range(1, all_samples_from_curr_folder.shape[0], 2):
            sign_range = all_samples_from_curr_folder.iloc[it - 1].end - all_samples_from_curr_folder.iloc[it].beg

            if sign_range <= m_sec_threshold:
                first_sign = all_samples_from_curr_folder.iloc[it - 1].sign
                second_sign = all_samples_from_curr_folder.iloc[it].sign

                first_idx = all_signs.index(first_sign)
                second_idx = all_signs.index(second_sign)
                close_signs_mat[first_idx, second_idx] += 1


# %%

threshold_count = 40
indexes = np.where(close_signs_mat >= threshold_count)
indexes_count = indexes[0].shape[0] * 2
unique_indexes = set(indexes[0].tolist() + indexes[1].tolist())
indexes_map = {str(x): it for it, x in enumerate(unique_indexes)}
rev_indexes_map = { str(v):int(k) for k, v in indexes_map.items()}
threshold_mat_close_sings = np.zeros((len(unique_indexes), len(unique_indexes)))

for idx1, idx2 in zip(indexes[0], indexes[1]):
    threshold_mat_close_sings[indexes_map[str(idx1)], indexes_map[str(idx2)]] = close_signs_mat[idx1, idx2]


# %%
df_mat = pd.DataFrame()

for it_row, row in enumerate(threshold_mat_close_sings):
    for it_col, col in enumerate(row):
        df_mat = df_mat.append(pd.DataFrame(dict(
            sinal_1=[all_signs[rev_indexes_map[str(it_row)]]],
            sinal_2=[all_signs[rev_indexes_map[str(it_col)]]],
            contagem=[col]
        )), ignore_index=True)
df_mat = df_mat.pivot('sinal_1', 'sinal_2', 'contagem')
chart: plt.axes = sns.heatmap(df_mat, annot=True, linewidths=.5, cbar=False, fmt=' ')
plt.tight_layout()
plt.show()

