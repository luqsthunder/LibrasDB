from libras_classifiers.librasdb_loaders import DBLoader2NPY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def label_point(x, y, val, ax):
    a = pd.DataFrame({"x": x, "y": y, "val": val})
    for i, point in a.iterrows():
        ax.text(point["x"] + 0.02, point["y"], str(point["val"]))


batch_size = 8
db = DBLoader2NPY(
    "../libras-db-folders", batch_size=batch_size, no_hands=False, angle_pose=True
)
db.fill_samples_absent_frames_with_na()
tsne = TSNE(n_components=2, random_state=0)
all_samples = db.batch_load_samples(samples_idxs=[x for x in range(db.db_length())])

all_x = []
all_y = []
all_samples_idx = []
all_idx = []
for it, sample in enumerate(all_samples[0]):
    sample_id_aux = []
    for it_frame, frame in enumerate(sample):
        if any(frame):
            all_x.append(frame[1:])
            all_y.append(np.argmax(all_samples[1][it]))
            all_idx.append(it)
            sample_id_aux.append(it)
    all_samples_idx.append(sample_id_aux)

res = tsne.fit_transform(all_x)

df = pd.DataFrame(
    dict(
        x=[x[0] for x in res],
        y=[x[1] for x in res],
        sample_idx=[x for x in all_idx],
        cls=all_y,
    )
)
np.random.seed(5)
samples_ids = (
    np.random.randint(0, 120, size=5).tolist()
    + np.random.randint(120, 240, size=5).tolist()
)
a = df["sample_idx"].isin(samples_ids)
df = df[a]

single_sample_idx = df.sample_idx.unique().tolist()
single_sample_list = []
for sample_id in single_sample_idx:
    sample = df[df["sample_idx"] == sample_id]
    single_sample_list.append(sample.iloc[0])

plt.tight_layout()
chart = sns.scatterplot(data=df, x="x", y="y", hue="sample_idx", style="cls")
label_point(
    x=[x.x for x in single_sample_list],
    y=[x.y for x in single_sample_list],
    val=[x.sample_idx for x in single_sample_list],
    ax=chart,
)
plt.show()
