import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import ImageFont, ImageDraw, Image
import matplotlib.ticker as ticker
import cv2 as cv


db_path = 'all_videos.csv'
sign_db = pd.read_csv(db_path)
print(sign_db.head(n=10))

# %%
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

more_than_30_df = pd.DataFrame(data=dict(sign=sorted_sings_count_mode))

# %%
last_count = 0
end = 0
for i in tqdm(range(20)):
    end = last_count + int(len(sorted_signs) * 0.05)
    end = end if end < len(sorted_signs) else len(sorted_signs) - 1
    df_10percent = pd.DataFrame(data=sorted_signs[last_count: end])
    last_count = end
    plt.figure(0, figsize=(21, 9), dpi=int(1080 / 9))
    chart = sns.barplot(x='name', y='count', data=df_10percent,
                        palette="Blues_d")
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize=5)
    chart.set(ylim=(0, sorted_signs[0]['count']))
    # plt.savefig('figfolder/5%_{}_signs_count.svg'.format(i))
    plt.show()

# %% Checando a distribuição dos sinais
tick_spacing = 3
fig = plt.figure(figsize=(21, 9), dpi=int(1080 / 9))
ax = fig.add_subplot(111)
ax.set_ylim(0, 100)
g = sns.catplot(x='sign', data=more_than_30_df, kind="count",
                palette="ch:.25",
                order=more_than_30_df.sign.value_counts(ascending=False).index,
                ax=ax)
plt.close(g.fig)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9)
ax.set_xlabel('Sinais')
ax.set_ylabel('Quantidade de Amostras')
plt.tight_layout()
plt.savefig('sign\'s-distribution.pdf')
plt.show()

# plt.figure(0, figsize=(21, 9), dpi=int(1080 / 9))
# chart = sns.violinplot(pd.DataFrame(sorted_signs)['count'])
# chart.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# plt.tight_layout()
# plt.savefig('sign\'s-violinplot.svg')
# plt.show()


# %%

all_count = np.array([x['count'] for x in sorted_signs])
last_quartil = np.array([x['count'] for x in sorted_signs[:int(len(sorted_signs) * 0.25)]])

mid = int(len(sorted_signs) * 0.25)
mid_quartil = np.array([x['count'] for x in sorted_signs[mid: mid * 2]])

print(np.mean(all_count))
print(np.median(all_count))
print(np.var(all_count))

print(np.mean(last_quartil))
print(np.median(last_quartil))
print(np.var(last_quartil))

print(np.mean(mid_quartil))
print(np.median(mid_quartil))
print(np.var(mid_quartil))

# %% plotando os sinais entre o 2 e o 3 quartil
last_count = int(len(sorted_signs) * 0.25)
end = 0
count = 10
for i in tqdm(range(25, 75, 5)):
    end = last_count + int(len(sorted_signs) * 0.05)
    end = end if end < len(sorted_signs) else len(sorted_signs) - 1
    df_10percent = pd.DataFrame(data=sorted_signs[last_count: end])
    last_count = end
    plt.figure(0, figsize=(21, 9), dpi=int(1080 / 9))
    chart = sns.barplot(x='name', y='count', data=df_10percent,
                        palette="Blues_d")
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90, fontsize=5)
    plt.tight_layout()
    plt.savefig('midfigs/5%_{}_signs_count.svg'.format(i))

# %% validando se algum video n�o conseguiu ser carregado completamente
bad_downloaded_videos = []
for v_name in tqdm(sign_db.video_file):
    res = cv.VideoCapture(v_name)
    if res is None:
        bad_downloaded_videos.append(v_name)

    res.release()

# %% video para o professor marcos
error_list = [
    'db/Inventário+Libras/FLN G2 D1 2entrevista v367/v0.mp4',
    'db/Inventário+Libras/FLN G1 D2 conversacaolivre v407/v0.mp4',
    'db/Inventário+Libras/FLN G2 D1 1entrevista v380/v0.mp4'
]
single_video_ref = cv.VideoCapture(sign_db.iloc[0].video_file)
fourcc = int(cv.VideoWriter_fourcc(*'H264'))
video_width = int(single_video_ref.get(cv.CAP_PROP_FRAME_WIDTH))
video_height = int(single_video_ref.get(cv.CAP_PROP_FRAME_HEIGHT))
video_fps = int(single_video_ref.get(cv.CAP_PROP_FPS))
slow_down = 2
final_video = cv.VideoWriter('all_cuted_signs_slow_down.mp4', fourcc,
                             video_fps // slow_down,
                             (video_width, video_height))

more_than_30_samples = [x for x in sorted_signs if x['count'] > 30]
# more_than_30_samples = [x for x in sorted_signs]
np.random.seed(5)
fontpath = "./font.ttf"
font = ImageFont.truetype(fontpath, 32)
black_frame = cv.imread('blackimg.jpeg')
black_frame = cv.resize(black_frame, (video_width, video_height))

for sample in tqdm(more_than_30_samples):
    name = sample['name']
    cur_sign_all_videos = sign_db[sign_db['sign'] == name]

    rnd_size = int(.25 * len(cur_sign_all_videos))
    rnd_size = rnd_size if rnd_size > 1 else 1
    five_percent_random = np.random.randint(len(cur_sign_all_videos),
                                            size=rnd_size)

    # five_percent_random = np.arange(len(cur_sign_all_videos))

    for it in five_percent_random:
        beg_pos_sign = cur_sign_all_videos.iloc[it].beg
        end_pos_sign = cur_sign_all_videos.iloc[it].end

        cur_video = None
        if cur_sign_all_videos.iloc[it].video_file in error_list:
            continue
        try:
            cur_video = cv.VideoCapture(cur_sign_all_videos.iloc[it].video_file)
        except ...:
            print(cur_sign_all_videos.iloc[it].video_file)
            continue

        cur_video.set(cv.CAP_PROP_POS_MSEC, beg_pos_sign)
        while cur_video.get(cv.CAP_PROP_POS_MSEC) <= end_pos_sign:
            ret, frame = cur_video.read()
            frame = cv.resize(frame, (video_width, video_height))
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.text((20, video_height - 80),
                      cur_sign_all_videos.iloc[it].sign,
                      font=font, fill=(0, 255, 255, 0))
            frame = np.array(img_pil)
            for __ in range(slow_down):
                final_video.write(frame)
        # black screen 2 seconds
        for _ in range(video_fps // 2):
            for __ in range(slow_down):
                final_video.write(black_frame)

final_video.release()

