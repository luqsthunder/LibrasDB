from libras_classifiers.librasdb_loaders import DBLoader2NPY

import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight


def data_generator(batchsize, db, amount_epochs, mode='train', random=False,
                   use_seed=True, seed=5):
    for _ in range(amount_epochs):
        # return batches
        samples_idx = list(range(0, db.db_length()))
        if random:
            if use_seed:
                np.random.seed(seed)

            np.random.shuffle(samples_idx)

        for it in range(0, db.db_length(), batchsize):
            end = it + batchsize \
                if it + batchsize < db.db_length() \
                else db.db_length()

            samples_2_get = samples_idx[it:end]
            X, y = db.batch_load_samples(samples_2_get)
            X = X[:, :, 2:]

            yield X, y
        print('finished epoch')

db = DBLoader2NPY('C:/Users/lucas/Downloads/libras-db')
db.fill_samples_absent_frames_with_na()

max_len_seq = db.find_longest_sample()
amount_joints_used = len(db.joints_used()) - 2

lstm_layer = \
    tf.keras.layers.LSTM(units=120, activation='tanh',
                         recurrent_activation='sigmoid',
                         return_sequences=False,
                         input_shape=(max_len_seq, amount_joints_used))

lstm_layer2 = \
    tf.keras.layers.LSTM(units=80, activation='tanh',
                         recurrent_activation='sigmoid',
                         return_sequences=False)

amount_classes = db.amount_classes()

net = tf.keras.Sequential()
net.add(lstm_layer)
# net.add(lstm_layer2)
net.add(tf.keras.layers.Dense(units=amount_classes,
                              activation='softmax'))
try:
    net.compile(optimizer='Adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
except ValueError as e:
    print(e)

try:
    batch_size = 8
    epochs = 10

    net.fit(x=data_generator(batch_size, db, epochs),
            epochs=epochs, verbose=2,
            steps_per_epoch=db.db_length() // batch_size)
except BaseException as e:
    print(e)