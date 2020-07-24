import libras_classifiers.librasdb_loaders
from libras_classifiers.librasdb_loaders import DBLoader2NPY

import numpy as np
import tensorflow as tf
#import tensorflow_addons as tfa
from tqdm.auto import tqdm

batch_size = 8
epochs = 10
db = DBLoader2NPY('../libras-db-folders', batch_size=batch_size,
                  no_hands=False, angle_pose=False)
db.fill_samples_absent_frames_with_na()

max_len_seq = db.find_longest_sample()
amount_joints_used = len(db.joints_used()) - 1

# %%
distributed_1d_conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(activation='relu', kernel_size=8,
                                                                              filters=32),
                                                       input_shape=(max_len_seq, amount_joints_used, 2))

distributed_1d_pool1 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool1D(pool_size=2))

distributed_1d_conv2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(activation='relu', kernel_size=4,
                                                                              filters=16))
distributed_1d_pool2 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool1D(pool_size=2))

distributed_flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())

lstm_layer = \
    tf.keras.layers.LSTM(units=120, activation='tanh',
                         recurrent_activation='sigmoid',
                         return_sequences=True)

lstm_layer2 = \
    tf.keras.layers.LSTM(units=80, activation='tanh',
                         recurrent_activation='sigmoid',
                         return_sequences=False)

lstm_layer3 = \
    tf.keras.layers.LSTM(units=60, activation='tanh',
                         recurrent_activation='sigmoid',
                         return_sequences=False)

amount_classes = db.amount_classes()

net = tf.keras.Sequential()
net.add(distributed_1d_conv1)
net.add(distributed_1d_pool1)
net.add(distributed_1d_conv2)
net.add(distributed_1d_pool2)
net.add(distributed_flatten)
net.add(lstm_layer)
net.add(lstm_layer2)
#net.add(lstm_layer3)
net.add(tf.keras.layers.Dense(units=amount_classes, activation='softmax'))
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

        for it in tqdm(range(0, db.db_length(), batchsize), desc=f'epoch {epc}'):
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
                        # sample_in_npy.append(np.vstack(row.values[1:]).reshape(1, 61, 2))
                        sample_in_npy.append(np.stack(row.values[1:], axis=0))

                    sample_in_npy = np.stack(sample_in_npy, axis=0)
                    samples_cache[s_it + it] = sample_in_npy

                sample_in_npy = samples_cache[s_it + it]
                x_new.append(sample_in_npy)
            x_new = np.stack(x_new, axis=0)

            #print(x_new.shape, y.shape)
            yield x_new, y, [None]
        #print('finished epoch')

try:
    # net.fit(x=data_generator(batch_size, db, epochs), epochs=epochs, verbose=2,
    #         steps_per_epoch=(db.db_length() // batch_size) + 1)
    net.fit(x=db, epochs=epochs, verbose=2,
            steps_per_epoch=(db.db_length() // batch_size) + 1)
except BaseException as e:
    print(e)
