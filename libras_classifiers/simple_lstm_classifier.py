from libras_classifiers.librasdb_loaders import DBLoader2NPY
import os
from tqdm import tqdm
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# %%
batch_size = 16
epochs = 20


def create_model(amount_time_steps, amount_classes):
    distributed_1d_conv1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(activation='relu', kernel_size=4,
                                                                                  filters=4),
                                                           )

    distributed_1d_pool1 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool1D(pool_size=2))

    distributed_1d_conv2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(activation='relu', kernel_size=2,
                                                                                  filters=42),
                                                           input_shape=(amount_time_steps, amount_joints_used, 2))
    distributed_1d_pool2 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool1D(pool_size=2))

    distributed_flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())

    lstm_layer = \
        tf.keras.layers.LSTM(units=80, activation='tanh',
                             return_sequences=True,
                             # input_shape=(max_len_seq, amount_joints_used)
                             )

    lstm_layer2 = \
        tf.keras.layers.LSTM(units=40, activation='tanh',
                             return_sequences=True)

    lstm_layer3 = \
        tf.keras.layers.LSTM(units=20, activation='tanh',
                             return_sequences=False)

    net = tf.keras.Sequential()
    # net.add(distributed_1d_conv1)
    # net.add(distributed_1d_pool1)
    net.add(distributed_1d_conv2)
    net.add(distributed_1d_pool2)
    net.add(distributed_flatten)
    net.add(lstm_layer)
    net.add(lstm_layer2)
    net.add(lstm_layer3)
    net.add(tf.keras.layers.Dense(units=amount_classes, activation='softmax'))
    # net.build()
    # net.summary()
    try:
        net.compile(optimizer='Adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    except ValueError as e:
        print(e)

    return net


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
    model_path_name = None

    def __init__(self, model_path='model.h5'):
        super().__init__()
        self.model_path_name = model_path

    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy'] > self.best_acc:
            self.best_acc = logs['accuracy']
            self.model.save(self.model_path_name)
        print(logs)


fit_res = None

try:
    all_classes = os.listdir('../sign_db_front_view')
    amount_joints_used = None

    res_df = pd.read_csv('res_xy_scaled.csv') if os.path.exists('res_xy_scaled.csv') else pd.DataFrame()
    all_combs = []
    for it in range(3, 4):
        class_comb_list = list(combinations(all_classes, it))
        all_combs.extend(class_comb_list)

    for curr_classes in tqdm(all_combs, total=len(all_combs)):
        if res_df.shape[0] != 0:
            if ' - '.join(curr_classes) in res_df.classes.values:
                continue

        db = DBLoader2NPY('../sign_db_front_view',
                          batch_size=batch_size,
                          not_use_pbar_in_load=True,
                          custom_internal_dir='',
                          only_that_classes=curr_classes,
                          scaler_cls=StandardScaler,
                          no_hands=False,
                          angle_pose=False)
        db.fill_samples_absent_frames_with_na()

        if amount_joints_used is None:
            amount_joints_used = len(db.joints_used()) - 1

        max_len_seq = db.find_longest_sample()
        amount_classes = db.amount_classes()

        net = create_model(max_len_seq, amount_classes)

        save_best_acc = SaveBestAcc()
        fit_res = net.fit(db, epochs=epochs, verbose=0)
        # res = net.predict(db, steps=160*len(curr_classes) // batch_size)
        c_df = pd.DataFrame(dict(
            classes=[' - '.join(curr_classes)],
            var_acc=[np.var(fit_res.history['accuracy'])],
            mean_acc=[np.mean(fit_res.history['accuracy'])],
            best_acc=[np.max(fit_res.history['accuracy'])],
            best_acc_epoch=[np.argmax(fit_res.history['accuracy'])]
        ))
        c_df.to_csv('res_xy_scaled.csv', index=False, header=False, mode='a')
        tf.keras.backend.clear_session()

    # class_map = {'1': 'nome', '2': 'nome2'}
    # for it, samples_item_path in enumerate(db.samples_path):
    #     if not np.argmax(res[it]) == samples_item_path[1]:
    #         print(samples_item_path, file=open('wrong-positivo-', mode='a'))

except BaseException as e:
    raise e

joints_to_use = ['frame', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist']
try:
    all_classes = os.listdir('../sign_db_front_view')
    amount_joints_used = None

    res_df = pd.read_csv('res_xy_scaled_some.csv') if os.path.exists('res_xy_scaled_some.csv') else pd.DataFrame()
    all_combs = []
    for it in range(3, 4):
        class_comb_list = list(combinations(all_classes, it))
        all_combs.extend(class_comb_list)

    for curr_classes in tqdm(all_combs, total=len(all_combs)):
        if res_df.shape[0] != 0:
            if ' - '.join(curr_classes) in res_df.classes.values:
                continue

        db = DBLoader2NPY('../sign_db_front_view',
                          batch_size=batch_size,
                          not_use_pbar_in_load=True,
                          custom_internal_dir='',
                          only_that_classes=curr_classes,
                          scaler_cls=StandardScaler,
                          joints_2_use=joints_to_use,
                          no_hands=False,
                          angle_pose=False)
        db.fill_samples_absent_frames_with_na()

        if amount_joints_used is None:
            amount_joints_used = len(db.joints_used()) - 1

        max_len_seq = db.find_longest_sample()
        amount_classes = db.amount_classes()

        net = create_model(max_len_seq, amount_classes)

        save_best_acc = SaveBestAcc()
        fit_res = net.fit(db, epochs=epochs, verbose=0)
        # res = net.predict(db, steps=160*len(curr_classes) // batch_size)
        c_df = pd.DataFrame(dict(
            classes=[' - '.join(curr_classes)],
            var_acc=[np.var(fit_res.history['accuracy'])],
            mean_acc=[np.mean(fit_res.history['accuracy'])],
            best_acc=[np.max(fit_res.history['accuracy'])],
            best_acc_epoch=[np.argmax(fit_res.history['accuracy'])]
        ))
        c_df.to_csv('res_xy_scaled_some.csv', index=False, header=False, mode='a')
        tf.keras.backend.clear_session()

    # class_map = {'1': 'nome', '2': 'nome2'}
    # for it, samples_item_path in enumerate(db.samples_path):
    #     if not np.argmax(res[it]) == samples_item_path[1]:
    #         print(samples_item_path, file=open('wrong-positivo-', mode='a'))

except BaseException as e:
    raise e

# %%
early_stopper = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=5,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

# %%
batch_size = 16
epochs = 120

joints_to_use = ['frame',
                 'Neck-RShoulder-RElbow',
                 'RShoulder-RElbow-RWrist',
                 'Neck-LShoulder-LElbow',
                 'LShoulder-LElbow-LWrist',
                 'RShoulder-Neck-LShoulder',
                 'left-Wrist-left-ThumbProximal-left-ThumbDistal',
                 'right-Wrist-right-ThumbProximal-right-ThumbDistal',
                 'left-Wrist-left-IndexFingerProximal-left-IndexFingerDistal',
                 'right-Wrist-right-IndexFingerProximal-right-IndexFingerDistal',
                 'left-Wrist-left-MiddleFingerProximal-left-MiddleFingerDistal',
                 'right-Wrist-right-MiddleFingerProximal-right-MiddleFingerDistal',
                 'left-Wrist-left-RingFingerProximal-left-RingFingerDistal',
                 'right-Wrist-right-RingFingerProximal-right-RingFingerDistal',
                 'left-Wrist-left-LittleFingerProximal-left-LittleFingerDistal',
                 'right-Wrist-right-LittleFingerProximal-right-LittleFingerDistal'
                 ]

db = DBLoader2NPY('../clean_sign_db_front_view', batch_size=batch_size,
                  shuffle=True, test_size=.25,
                  # add_angle_derivatives=True,
                  #only_that_classes=['HOMEM', 'PORQUE'],
                  no_hands=False, angle_pose=True, joints_2_use=joints_to_use)
db.fill_samples_absent_frames_with_na()

max_len_seq = db.find_longest_sample()
amount_joints_used = len(db.joints_used()) - 1

best_accs = []

# %%
for _ in range(1):
    lstm_layer = \
        tf.keras.layers.LSTM(units=30, activation='tanh',
                             return_sequences=True,
                             dropout=0.25, recurrent_dropout=0.25,
                             input_shape=(max_len_seq, amount_joints_used)
                             )

    lstm_layer2 = \
        tf.keras.layers.LSTM(units=15, activation='tanh',
                             return_sequences=False)

    lstm_layer3 = \
        tf.keras.layers.LSTM(units=15, activation='tanh',
                             return_sequences=False)

    net = tf.keras.Sequential()
    net.add(lstm_layer)
    # net.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(rate=0.2)))
    # net.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=30)))
    net.add(lstm_layer2)
    # net.add(lstm_layer3)
    # net.add(tf.keras.layers.Dense(units=40, activation='sigmoid'))
    # net.add(tf.keras.layers.Dropout(rate=0.3))
    net.add(tf.keras.layers.Dense(units=4, activation='softmax'))
    try:
        net.compile(optimizer='Adam',
                    loss='categorical_crossentropy',

                    metrics=['accuracy'])
    except ValueError as e:
        print(e)

    fit_res = net.fit(db.train(), epochs=epochs, verbose=0, shuffle=False, workers=0,
                      callbacks=[early_stopper],
                      validation_data=db.validation())
    print(dict(var_acc=[np.var(fit_res.history['accuracy'])],
               mean_acc=[np.mean(fit_res.history['accuracy'])],
               mean_acc_val=[np.mean(fit_res.history['val_accuracy'])],
               best_acc_val=[np.max(fit_res.history['val_accuracy'])],
               best_acc=[np.max(fit_res.history['accuracy'])],
               best_acc_val_epoch=[np.argmax(fit_res.history['val_accuracy'])],
               best_acc_epoch=[np.argmax(fit_res.history['accuracy'])]))
    best_accs.append(np.max(fit_res.history['accuracy']))

    plt.figure(0, dpi=720/9, figsize=(16, 9))
    plt.title('Accuracy')
    plt.axhline(y=0.99, linestyle='-.')
    plt.axhline(y=0.90, linestyle='--')
    plt.axhline(y=0.80, linestyle='--')
    plt.axhline(y=0.70, linestyle='--')
    plt.axhline(y=0.60, linestyle='--')
    plt.plot(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['val_accuracy'], label='val_accuracy')
    plt.scatter(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['val_accuracy'], label='val_accuracy')
    plt.plot(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['accuracy'], label='accuracy')
    plt.scatter(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['accuracy'], label='accuracy')

    plt.legend()
    plt.show()

    plt.figure(0, dpi=720/9, figsize=(16, 9))
    plt.title('Loss')
    plt.plot(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['val_loss'], label='val_loss')
    plt.scatter(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['val_loss'])
    plt.plot(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['loss'], label='loss')
    plt.scatter(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['loss'])
    plt.legend()
    plt.show()

# %%
svc_clf = SVC(kernel='rbf', probability=True)
X = []
y = []

for it, x in enumerate(db.train()):
    X.extend(x[0])
    y.extend(x[1])

x_val = []
y_val = []

for it, x in enumerate(db.validation()):
    x_val.extend(x[0])
    y_val.extend(x[1])

X = np.stack([x.reshape((1, -1)) for x in X])
X = X.reshape((X.shape[0], 735))
y = np.array([np.argmax(x) for x in np.stack(y)]).reshape((-1, 1)).reshape((-1))

x_val = np.stack([x.reshape((1, -1)) for x in x_val])
x_val = x_val.reshape((x_val.shape[0], 735))
y_val = np.array([np.argmax(x) for x in np.stack(y_val)]).reshape((-1, 1)).reshape((-1))

clf_list = [KNeighborsClassifier(n_neighbors=5), AdaBoostClassifier(), RandomForestClassifier(), DecisionTreeClassifier(),
            SVC(kernel='rbf', probability=True)]
for clf in clf_list:
    clf.fit(X=X, y=y)
    y_pred = clf.predict(x_val)
    print(clf, accuracy_score(y_val, y_pred))

