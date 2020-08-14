from libras_classifiers.librasdb_loaders import DBLoader2NPY
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm.auto import tqdm

batch_size = 8
epochs = 10
db = DBLoader2NPY('../libras-db-folders', batch_size=batch_size,
                  no_hands=True, angle_pose=True)
#db.fill_samples_absent_frames_with_na()

max_len_seq = db.find_longest_sample()
amount_joints_used = 5#len(db.joints_used()) - 1

# %%
distributed_1d_conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(activation='relu', kernel_size=4,
                                                                              filters=4),
                                                       )

distributed_1d_pool1 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool1D(pool_size=2))

distributed_1d_conv2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(activation='relu', kernel_size=4,
                                                                              filters=16))
distributed_1d_pool2 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool1D(pool_size=2))

distributed_flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(),
                                                      input_shape=(max_len_seq, amount_joints_used, 2))

lstm_layer = \
    tf.keras.layers.LSTM(units=120, activation='tanh',
                         recurrent_activation='sigmoid',
                         return_sequences=False,
                         )

lstm_layer2 = \
    tf.keras.layers.LSTM(units=16, activation='tanh',
                         recurrent_activation='sigmoid',
                         return_sequences=False)

lstm_layer3 = \
    tf.keras.layers.LSTM(units=60, activation='tanh',
                         recurrent_activation='sigmoid',
                         return_sequences=False)

amount_classes = db.amount_classes()

net = tf.keras.Sequential()
#net.add(distributed_1d_conv1)
#net.add(distributed_1d_pool1)
#net.add(distributed_1d_conv2)
#net.add(distributed_1d_pool2)
net.add(distributed_flatten)
net.add(lstm_layer)
#net.add(lstm_layer2)
#net.add(lstm_layer3)
net.add(tf.keras.layers.Dense(units=amount_classes, activation='softmax'))
#net.build()
net.summary()
try:
    net.compile(optimizer='Adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
except ValueError as e:
    print(e)


# %%
def data_generator(batchsize, db, amount_epochs, mode='train', random=False,
                   use_seed=True, seed=5):
    for epc in range(amount_epochs):
        # return batches
        samples_idx = list(range(0, db.db_length()))
        samples_cache = [None for _ in range(len(samples_idx))]
        if random:
            if use_seed:
                np.random.seed(seed)

            np.random.shuffle(samples_idx)

        for it in range(0, db.db_length(), batchsize):
            end = it + batchsize \
                if it + batchsize < db.db_length() \
                else db.db_length()

            samples_2_get = samples_idx[it:end]
            X, y = db.batch_load_samples(samples_2_get, as_npy=False)

            shape_before = y[0].shape
            y = np.concatenate(y).reshape(len(samples_2_get), shape_before[0], shape_before[1])

            x_new = []

            for s_it, sample in enumerate(X):

                if samples_cache[s_it + it] is None:
                    sample_in_npy = []
                    for row in sample.iterrows():
                        row = row[1]
                        sample_in_npy.append(np.stack(row.values[1:], axis=0))

                    sample_in_npy = np.stack(sample_in_npy, axis=0)
                    samples_cache[s_it + it] = sample_in_npy

                sample_in_npy = samples_cache[s_it + it]
                x_new.append(sample_in_npy)
            x_new = np.stack(x_new, axis=0)

            yield x_new, y, [None]


class SaveBestAcc(tf.keras.callbacks.Callback):
    best_acc = 0
    best_model = None

    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy'] > self.best_acc:
            self.best_acc = logs['accuracy']
            self.model.save('model.h5')
        print(logs)


def plot_tsne_frames():
    tsne = TSNE(n_components=2, random_state=0)
    all_samples = db.batch_load_samples(samples_idxs=[x for x in range(db.db_length())])

    all_x = []
    all_y = []
    for it, sample in enumerate(all_samples[0]):
        for frame in sample:
            all_x.append(frame[1:])
            all_y.append(np.argmax(all_samples[1][it]))

    res = tsne.fit_transform(all_x)

    plt.figure(0, figsize=(21, 9), dpi=720//9)
    df2 = pd.DataFrame(dict(tsne_2d_one=[x[0] for x in res], tsne_2d_two=[x[1] for x in res],
                            cls=all_y))
    sns.scatterplot(
            x="tsne_2d_one", y="tsne_2d_two",
            hue="cls",
            palette=sns.color_palette("hls", 2),
            data=df2,
            legend="full",
            alpha=1
        )
    plt.show()


try:
    plot_tsne_frames()
    # save_best_acc = SaveBestAcc()
    # model_file_name = '{epoch:02d}-.hdf5'
    # net.fit(data_generator(batch_size, db, amount_epochs=epochs),
    #         steps_per_epoch=db.db_length() // batch_size,
    #         epochs=epochs,
    #         callbacks=[save_best_acc])
    # net = tf.keras.models.load_model('model.h5')
    # res = net.predict(data_generator(1, db, amount_epochs=1, mode='test'), steps=80)
    #
    # class_map = {'1': 'nome', '2': 'nome2'}
    # for it, samples_item_path in enumerate(db.samples_path):
    #     if not np.argmax(res[it]) == samples_item_path[1]:
    #         print(samples_item_path, file=open('wrong-positivo-', mode='a'))

except BaseException as e:
    raise e
