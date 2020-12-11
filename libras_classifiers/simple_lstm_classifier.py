import sys
import os
from libras_classifiers.librasdb_loaders import DBLoader2NPY
from tqdm import tqdm
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import itertools

# %%
early_stopper = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=8,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
batch_size = 100
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

db = DBLoader2NPY('../clean_sign_db_front_view',
                  batch_size=batch_size,
                  shuffle=True, test_size=.25,
                  add_derivatives=True,
                  no_hands=False,
                  angle_pose=True,
                  joints_2_use=joints_to_use
                  )
db.fill_samples_absent_frames_with_na()

db_no_dt = DBLoader2NPY('../clean_sign_db_front_view',
                        batch_size=batch_size,
                        shuffle=True, test_size=.25,
                        add_derivatives=True,
                        no_hands=False,
                        angle_pose=True,
                        joints_2_use=joints_to_use
                        )
db_no_dt.fill_samples_absent_frames_with_na()

db_xy = DBLoader2NPY('../clean_sign_db_front_view',
                     batch_size=batch_size,
                     shuffle=True, test_size=.25,
                     scaler_cls=StandardScaler,
                     add_derivatives=True,
                     custom_internal_dir='',
                     no_hands=False,
                     angle_pose=False,
                     )
db_xy.fill_samples_absent_frames_with_na()

db_no_dt_xy = DBLoader2NPY('../clean_sign_db_front_view',
                           batch_size=batch_size,
                           shuffle=True, test_size=.25,
                           scaler_cls=StandardScaler,
                           custom_internal_dir='',
                           no_hands=False,
                           angle_pose=False,
                           )
db_no_dt_xy.fill_samples_absent_frames_with_na()


# %%

def save_pdf_fig(fit_history: tf.keras.callbacks.History, base_folder: str, acc_name: str, loss_name: str):
    """

    Parameters
    ----------
    fit_history
    base_folder
    acc_name
    loss_name

    Returns
    -------

    """
    plt.figure(0, dpi=720 / 9, figsize=(16, 9))
    plt.title(acc_name)
    plt.axhline(y=0.99, linestyle='-.')
    plt.axhline(y=0.90, linestyle='--')
    plt.axhline(y=0.80, linestyle='--')
    plt.axhline(y=0.70, linestyle='--')
    plt.axhline(y=0.60, linestyle='--')
    plt.plot(np.arange(len(fit_history['val_accuracy'])), fit_history['val_accuracy'], label='val_accuracy')
    plt.scatter(np.arange(len(fit_history['val_accuracy'])), fit_history['val_accuracy'], label='val_accuracy')
    plt.plot(np.arange(len(fit_history['val_accuracy'])), fit_history['accuracy'], label='accuracy')
    plt.scatter(np.arange(len(fit_history['val_accuracy'])), fit_history['accuracy'], label='accuracy')

    plt.legend()
    plt.savefig(base_folder + acc_name + '.pdf')
    plt.close()

    plt.figure(0, dpi=720 / 9, figsize=(16, 9))
    plt.title(loss_name)
    plt.plot(np.arange(len(fit_history['val_accuracy'])), fit_history['val_loss'], label='val_loss')
    plt.scatter(np.arange(len(fit_history['val_accuracy'])), fit_history['val_loss'])
    plt.plot(np.arange(len(fit_history['val_accuracy'])), fit_history['loss'], label='loss')
    plt.scatter(np.arange(len(fit_history['val_accuracy'])), fit_history['loss'])
    plt.legend()
    plt.savefig(base_folder + loss_name + '.pdf')
    plt.close()


def create_lstms(lstm_units_list: list, dropout: float, dropout_recurrent: float, input_shape: tuple = None) -> list:
    """

    Parameters
    ----------
    lstm_units_list
    dropout
    dropout_recurrent
    input_shape

    Returns
    -------

    """
    lstms = []
    for it, units in enumerate(lstm_units_list):
        if input_shape is not None and it == 0:
            lstm_layer = tf.keras.layers.LSTM(units=units,
                                              activation='tanh',
                                              return_sequences=len(lstm_units_list) > it + 1,
                                              dropout=dropout,
                                              recurrent_dropout=dropout_recurrent,
                                              input_shape=input_shape)
            print(input_shape)
            lstms.append(lstm_layer)
            continue

        lstm_layer = tf.keras.layers.LSTM(units=units,
                                          activation='tanh',
                                          return_sequences=len(lstm_units_list) > it + 1,
                                          dropout=dropout,
                                          recurrent_dropout=dropout_recurrent)
        lstms.append(lstm_layer)

    return lstms


def train_angle_lstm(pose, amount_lstm, lstm_1_units, lstm_2_units, lstm_3_units, dropout, dropout_recurrent,
                     dense_units, use_derivative):
    epochs = 10000

    db_used = db if use_derivative else db_no_dt
    max_len_seq = db_used.find_longest_sample()
    amount_joints_used = len(db_used.joints_used()) - 1

    net = tf.keras.Sequential()

    if dense_units is not None:
        net.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=dense_units, activation='tanh',
                                                                      input_shape=(amount_joints_used,)),
                                                input_shape=(max_len_seq, amount_joints_used)))

    return make_base_classifier(net, pose_db=db_used, pose=pose, amount_lstm=amount_lstm,
                                lstm_1_units=lstm_1_units, lstm_2_units=lstm_2_units, lstm_3_units=lstm_3_units,
                                dropout=dropout, dropout_recurrent=dropout_recurrent, epochs=epochs,
                                dense_units=dense_units,
                                input_shape=(max_len_seq, amount_joints_used) if dense_units is None else None)


def train_lstm_xy_pose(pose, amount_lstm, lstm_1_units, lstm_2_units, lstm_3_units, dropout, dropout_recurrent,
                       dense_units, use_derivatives=False):
    epochs = 1000

    db_used = db_xy if use_derivatives else db_no_dt_xy
    max_len_seq = db_used.find_longest_sample()
    amount_joints_used = len(db_used.joints_used()) - 1

    net = tf.keras.Sequential()

    net.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(),
                                            input_shape=(max_len_seq, amount_joints_used, 2)))
    if dense_units is not None:
        net.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=dense_units, activation='tanh')))

    return make_base_classifier(net, pose_db=db_used, pose=pose, amount_lstm=amount_lstm, lstm_1_units=lstm_1_units,
                                lstm_2_units=lstm_2_units, lstm_3_units=lstm_3_units, dropout=dropout,
                                dense_units=dense_units, dropout_recurrent=dropout_recurrent, epochs=epochs)


def train_lstm_xy_pose_cnn(pose, amount_lstm, lstm_1_units, lstm_2_units, lstm_3_units, dropout, dropout_recurrent,
                           cnn_1_filters, cnn_2_filters, cnn_kernel, use_derivatives=False):
    epochs = 1000

    db_used = db_xy if use_derivatives else db_no_dt_xy
    max_len_seq = db_used.find_longest_sample()
    amount_joints_used = len(db_used.joints_used()) - 1

    net = tf.keras.Sequential()

    net.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv1D(input_shape=(max_len_seq, amount_joints_used, 2), kernel_size=amount_joints_used // 2,
                               activation='relu', filters=cnn_1_filters)
    ))
    net.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv1D(input_shape=(max_len_seq, amount_joints_used, 2), kernel_size=cnn_kernel,
                               activation='relu', filters=cnn_2_filters)
    ))
    net.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool1D()))
    net.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))

    return make_base_classifier(net, pose_db=db_used, pose=pose, amount_lstm=amount_lstm, lstm_1_units=lstm_1_units,
                                lstm_2_units=lstm_2_units, lstm_3_units=lstm_3_units, dropout=dropout,
                                dense_units=None, dropout_recurrent=dropout_recurrent, epochs=epochs,
                                cnn_kernel=cnn_kernel, cnn_1_filters=cnn_1_filters, cnn_2_filters=cnn_2_filters)


def make_base_classifier(net, pose_db, pose, amount_lstm, lstm_1_units, lstm_2_units, lstm_3_units, dropout,
                         dropout_recurrent, dense_units, epochs, input_shape=None, cnn_kernel=None,
                         cnn_1_filters=None, cnn_2_filters=None):
    lstm_units = [lstm_1_units, lstm_2_units, lstm_3_units]
    lstms = create_lstms([lstm_units[it] for it in range(amount_lstm)], dropout=dropout,
                         dropout_recurrent=dropout_recurrent, input_shape=input_shape)
    for lstm in lstms:
        net.add(lstm)

    net.add(tf.keras.layers.Dense(units=pose_db.amount_classes(), activation='softmax'))
    try:
        net.compile(optimizer='Adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        net.summary()
    except ValueError as e:
        print(e)

    fit_res = net.fit(pose_db.train(), epochs=epochs, verbose=0, shuffle=False, workers=2,
                      callbacks=[early_stopper], validation_data=pose_db.validation())

    cnn_str_params = f'CNN Kernel {cnn_kernel} Filters {cnn_1_filters} {cnn_2_filters}' \
                     if cnn_kernel is not None and cnn_1_filters is not None and cnn_2_filters is not None else ''

    acc_name = f'Accuracy {pose} lstm-{amount_lstm} lstm-{lstm_1_units} lstm-{lstm_2_units} lstm {lstm_3_units} ' \
               f'drop {dropout} rdrop {dropout_recurrent} dense {dense_units}' + cnn_str_params

    loss_name = f'Loss {pose} lstm-{amount_lstm} lstm-{lstm_1_units} lstm-{lstm_2_units} lstm {lstm_3_units} ' \
                f'drop {dropout} rdrop {dropout_recurrent} dense {dense_units}' + cnn_str_params

    base_folder = '../../first_batch_figs/'
    save_pdf_fig(fit_res.history, base_folder, acc_name, loss_name)
    return base_folder + acc_name + '.pdf', base_folder + loss_name + '.pdf', fit_res.history

# %%

########################################################################################################################

# pose angulo, sem densas, com derivadas
dt_pose = dict(
    amount_lstm=[1, 2, 3],
    lstm_1_units=[90, 45],
    lstm_2_units=[30, 20],
    lstm_3_units=[15, 10],
    dropout=[0.0, 0.35],
    dropout_recurrent=[0.0, 0.35],
)

keys, vals = zip(*dt_pose.items())
all_params_product = [{key: x[it] for it, key in enumerate(keys)} for x in list(itertools.product(*vals))]

exp_path = '../'
csv_angle_name = 'all_experiments_pose_angle_no_dense_derivatives_batch1.csv'

df_exp = pd.read_csv(exp_path + csv_angle_name) if os.path.exists(exp_path + csv_angle_name) else pd.DataFrame()
save_header = df_exp.shape[0] == 0
for param in tqdm(all_params_product):
    if df_exp.shape[0] != 0:
        if str(param) in df_exp.exp_name.unique().tolist():
            continue

    tf.keras.backend.clear_session()
    acc_pdf_file, loss_pdf_file, history = train_angle_lstm('angle-no-dense', dense_units=None,
                                                            use_derivative=True, **param)

    pd.DataFrame(dict(
        exp_name=[str(param)],
        acc=[np.max(history['accuracy'])],
        val_acc=[np.max(history['val_accuracy'])],
        best_epoch_acc=[np.argmax(history['accuracy'])],
        best_epoch_val_acc=[np.argmax(history['val_accuracy'])],
        dense_units=[0],
        amount_lstm=[param['amount_lstm']],
        lstm_1_units=[param['lstm_1_units']],
        lstm_2_units=[param['lstm_2_units']],
        lstm_3_units=[param['lstm_3_units']],
        dropout=[param['dropout']],
        dropout_recurrent=[param['dropout_recurrent']],
    )).to_csv(exp_path + csv_angle_name, mode='a', header=save_header)
    save_header = False

########################################################################################################################
# %% pose angulo, sem derivadas, com densas, sem densas
dt_pose = dict(
    dense_units=[None, 90, 45],
    amount_lstm=[1, 2, 3],
    lstm_1_units=[90, 45],
    lstm_2_units=[30, 20],
    lstm_3_units=[15, 10],
    dropout=[0.0, 0.35],
    dropout_recurrent=[0.0, 0.35],
)

keys, vals = zip(*dt_pose.items())
all_params_product = [{key: x[it] for it, key in enumerate(keys)} for x in list(itertools.product(*vals))]

exp_path = '../'
csv_angle_name = 'all_experiments_pose_angle_no_dense_no_derivatives_batch1.csv'

df_exp = pd.read_csv(exp_path + csv_angle_name) if os.path.exists(exp_path + csv_angle_name) else pd.DataFrame()
save_header = df_exp.shape[0] == 0
for param in tqdm(all_params_product):
    if df_exp.shape[0] != 0:
        if str(param) in df_exp.exp_name.unique().tolist():
            continue

    tf.keras.backend.clear_session()
    acc_pdf_file, loss_pdf_file, history = train_angle_lstm('angle-no-dense-no-derivative',
                                                            use_derivative=False, **param)

    pd.DataFrame(dict(
        exp_name=[str(param)],
        acc=[np.max(history['accuracy'])],
        val_acc=[np.max(history['val_accuracy'])],
        best_epoch_acc=[np.argmax(history['accuracy'])],
        best_epoch_val_acc=[np.argmax(history['val_accuracy'])],
        dense_units=[param['dense_units']],
        amount_lstm=[param['amount_lstm']],
        lstm_1_units=[param['lstm_1_units']],
        lstm_2_units=[param['lstm_2_units']],
        lstm_3_units=[param['lstm_3_units']],
        dropout=[param['dropout']],
        dropout_recurrent=[param['dropout_recurrent']],
    )).to_csv(exp_path + csv_angle_name, mode='a', header=save_header)
    save_header = False
########################################################################################################################
#%%

dt_pose = dict(
    amount_lstm=[1, 2, 3],
    lstm_1_units=[90, 45],
    lstm_2_units=[30, 20],
    lstm_3_units=[15, 10],
    dropout=[0.0, 0.35],
    dropout_recurrent=[0.0, 0.35],
)

keys, vals = zip(*dt_pose.items())
all_params_product = [{key: x[it] for it, key in enumerate(keys)} for x in list(itertools.product(*vals))]

exp_path = '../'
csv_xy_no_dense_name = 'all_experiments_pose_xy_no_dense_batch1.csv'

df_exp = pd.read_csv(exp_path + csv_xy_no_dense_name) \
         if os.path.exists(exp_path + csv_xy_no_dense_name) else pd.DataFrame()

save_header = df_exp.shape[0] == 0

for param in tqdm(all_params_product):
    if df_exp.shape[0] != 0:
        if str(param) in df_exp.exp_name.unique().tolist():
            continue

    tf.keras.backend.clear_session()
    try:
        acc_pdf_file, loss_pdf_file, history = train_lstm_xy_pose('xy-pose',dense_units=None, **param)
    except:
        continue

    pd.DataFrame(dict(
        exp_name=[str(param)],
        acc=[np.max(history['accuracy'])],
        val_acc=[np.max(history['val_accuracy'])],
        best_epoch_acc=[np.argmax(history['accuracy'])],
        best_epoch_val_acc=[np.argmax(history['val_accuracy'])],
        amount_lstm=[param['amount_lstm']],
        lstm_1_units=[param['lstm_1_units']],
        lstm_2_units=[param['lstm_2_units']],
        lstm_3_units=[param['lstm_3_units']],
        dropout=[param['dropout']],
        dropout_recurrent=[param['dropout_recurrent']],
    )).to_csv(exp_path + csv_xy_no_dense_name, mode='a', header=save_header)
    save_header = False

########################################################################################################################
#%%
dt_pose = dict(
    cnn_1_filters=[16],
    cnn_2_filters=[16, 8],
    cnn_kernel=[5, 9],
    amount_lstm=[1, 2, 3],
    lstm_1_units=[90, 45],
    lstm_2_units=[30, 20],
    lstm_3_units=[15, 10],
    dropout=[0.0, 0.35],
    dropout_recurrent=[0.0],
)

keys, vals = zip(*dt_pose.items())
all_params_product = [{key: x[it] for it, key in enumerate(keys)} for x in list(itertools.product(*vals))]

exp_path = '../'
csv_xy_cnn_name = 'all_experiments_pose_xy_cnn_batch1.csv'

df_exp = pd.read_csv(exp_path + csv_xy_cnn_name) if os.path.exists(exp_path + csv_xy_cnn_name) else pd.DataFrame()

save_header = df_exp.shape[0] == 0
for param in tqdm(all_params_product):
    if df_exp.shape[0] != 0:
        if str(param) in df_exp.exp_name.unique().tolist():
            continue

    tf.keras.backend.clear_session()
    acc_pdf_file, loss_pdf_file, history = train_lstm_xy_pose_cnn('xy-cnn', **param)

    pd.DataFrame(dict(
        exp_name=[str(param)],
        acc=[np.max(history['accuracy'])],
        val_acc=[np.max(history['val_accuracy'])],
        best_epoch_acc=[np.argmax(history['accuracy'])],
        best_epoch_val_acc=[np.argmax(history['val_accuracy'])],
        cnn_1_filters=[param['cnn_1_filters']],
        cnn_2_filters=[param['cnn_2_filters']],
        cnn_kernel=[param['cnn_kernel']],
        amount_lstm=[param['amount_lstm']],
        lstm_1_units=[param['lstm_1_units']],
        lstm_2_units=[param['lstm_2_units']],
        lstm_3_units=[param['lstm_3_units']],
        dropout=[param['dropout']],
        dropout_recurrent=[param['dropout_recurrent']],
    )).to_csv(exp_path + csv_xy_cnn_name, mode='a', header=save_header)
    save_header = False

# %%
#######################################################################################################################
dt_pose = dict(
    cnn_1_filters=[16],
    cnn_2_filters=[16, 8],
    cnn_kernel=[5, 9],
    amount_lstm=[1, 2, 3],
    lstm_1_units=[90, 45],
    lstm_2_units=[30, 20],
    lstm_3_units=[15, 10],
    dropout=[0.0, 0.35],
    dropout_recurrent=[0.35],
)

keys, vals = zip(*dt_pose.items())
all_params_product = [{key: x[it] for it, key in enumerate(keys)} for x in list(itertools.product(*vals))]

exp_path = '../'
csv_xy_cnn_name = 'all_experiments_pose_xy_cnn_batch1.csv'

df_exp = pd.read_csv(exp_path + csv_xy_cnn_name) if os.path.exists(exp_path + csv_xy_cnn_name) else pd.DataFrame()

save_header = df_exp.shape[0] == 0
for param in tqdm(all_params_product):
    if df_exp.shape[0] != 0:
        if str(param) in df_exp.exp_name.unique().tolist():
            continue

    tf.keras.backend.clear_session()
    acc_pdf_file, loss_pdf_file, history = train_lstm_xy_pose_cnn('xy-cnn', **param)

    pd.DataFrame(dict(
        exp_name=[str(param)],
        acc=[np.max(history['accuracy'])],
        val_acc=[np.max(history['val_accuracy'])],
        best_epoch_acc=[np.argmax(history['accuracy'])],
        best_epoch_val_acc=[np.argmax(history['val_accuracy'])],
        cnn_1_filters=[param['cnn_1_filters']],
        cnn_2_filters=[param['cnn_2_filters']],
        cnn_kernel=[param['cnn_kernel']],
        amount_lstm=[param['amount_lstm']],
        lstm_1_units=[param['lstm_1_units']],
        lstm_2_units=[param['lstm_2_units']],
        lstm_3_units=[param['lstm_3_units']],
        dropout=[param['dropout']],
        dropout_recurrent=[param['dropout_recurrent']],
    )).to_csv(exp_path + csv_xy_cnn_name, mode='a', header=save_header)
    save_header = False
# %%

