import sys
import os
sys.path.append('../')

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
from PyPDF2 import PdfFileMerger

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
                  add_angle_derivatives=True,
                  no_hands=False,
                  angle_pose=True,
                  joints_2_use=joints_to_use
                  )
db.fill_samples_absent_frames_with_na()

db_xy = DBLoader2NPY('../clean_sign_db_front_view',
                     batch_size=batch_size,
                     shuffle=True, test_size=.25,
                     scaler_cls=StandardScaler,
                     custom_internal_dir='',
                     no_hands=False,
                     angle_pose=False,
                     )
db_xy.fill_samples_absent_frames_with_na()


# %%
def train_lstm(pose, amount_lstm, lstm_1_units, lstm_2_units, lstm_3_units, dropout, dropout_recurrent, dense_units):
    epochs = 10000000

    max_len_seq = db.find_longest_sample()
    amount_joints_used = len(db.joints_used()) - 1

    lstm_layer = \
        tf.keras.layers.LSTM(units=lstm_1_units, activation='tanh',
                             return_sequences=amount_lstm > 1,
                             dropout=dropout, recurrent_dropout=dropout_recurrent,
                             # input_shape=(max_len_seq, amount_joints_used)
                             )

    lstm_layer2 = \
        tf.keras.layers.LSTM(units=lstm_2_units if lstm_2_units is not None else 1, activation='tanh',
                             dropout=dropout, recurrent_dropout=dropout_recurrent,
                             return_sequences=amount_lstm > 2)

    lstm_layer3 = \
        tf.keras.layers.LSTM(units=lstm_3_units if lstm_3_units is not None else 1, activation='tanh',
                             dropout=dropout, recurrent_dropout=dropout_recurrent,
                             return_sequences=amount_lstm > 3)

    net = tf.keras.Sequential()

    net.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=dense_units, activation='tanh',
                                                                  input_shape=(amount_joints_used, )),
                                            input_shape=(max_len_seq, amount_joints_used)))

    if amount_lstm >= 1:
        net.add(lstm_layer)
    if amount_lstm >= 2:
        net.add(lstm_layer2)
    if amount_lstm >= 3:
        net.add(lstm_layer3)

    net.add(tf.keras.layers.Dense(units=db.amount_classes(), activation='softmax'))
    try:
        net.compile(optimizer='Adam',
                    loss='categorical_crossentropy',

                    metrics=['accuracy'])
    except ValueError as e:
        print(e)

    fit_res = net.fit(db.train(), epochs=epochs, verbose=0, shuffle=False, workers=0,
                      callbacks=[early_stopper],
                      validation_data=db.validation())

    plt.figure(0, dpi=720 / 9, figsize=(16, 9))
    acc_name = f'Accuracy {pose} lstm-{amount_lstm} lstm-{lstm_1_units} lstm-{lstm_2_units} lstm {lstm_3_units} ' \
               f'drop {dropout} rdrop {dropout_recurrent} dense {dense_units}'
    loss_name = f'Loss {pose} lstm-{amount_lstm} lstm-{lstm_1_units} lstm-{lstm_2_units} lstm {lstm_3_units} ' \
                f'drop {dropout} rdrop {dropout_recurrent} dense {dense_units}'
    plt.title(acc_name)
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
    plt.savefig('../first_batch_figs/' + acc_name + '.pdf')
    plt.close()

    plt.figure(0, dpi=720 / 9, figsize=(16, 9))
    plt.title(loss_name)
    plt.plot(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['val_loss'], label='val_loss')
    plt.scatter(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['val_loss'])
    plt.plot(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['loss'], label='loss')
    plt.scatter(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['loss'])
    plt.legend()
    plt.savefig('../first_batch_figs/' + loss_name + '.pdf')
    plt.close()

    return '../first_batch_figs/' + acc_name + '.pdf', '../first_batch_figs/' + loss_name + '.pdf', fit_res.history


def train_lstm_xy_pose(pose, amount_lstm, lstm_1_units, lstm_2_units, lstm_3_units, dropout, dropout_recurrent,
                       dense_units):
    epochs = 10000000

    max_len_seq = db_xy.find_longest_sample()
    amount_joints_used = len(db_xy.joints_used()) - 1

    best_accs = []

    lstm_layer = \
        tf.keras.layers.LSTM(units=lstm_1_units, activation='tanh',
                             return_sequences=amount_lstm > 1,
                             dropout=dropout, recurrent_dropout=dropout_recurrent,
                             # input_shape=(max_len_seq, amount_joints_used)
                             )

    lstm_layer2 = \
        tf.keras.layers.LSTM(units=lstm_2_units if lstm_2_units is not None else 1, activation='tanh',
                             dropout=dropout, recurrent_dropout=dropout_recurrent,
                             return_sequences=amount_lstm > 2)

    lstm_layer3 = \
        tf.keras.layers.LSTM(units=lstm_3_units if lstm_3_units is not None else 1, activation='tanh',
                             dropout=dropout, recurrent_dropout=dropout_recurrent,
                             return_sequences=amount_lstm > 3)

    net = tf.keras.Sequential()

    net.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(),
                                            input_shape=(max_len_seq, amount_joints_used, 2)))

    net.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=dense_units, activation='tanh')))

    if amount_lstm >= 1:
        net.add(lstm_layer)
    if amount_lstm >= 2:
        net.add(lstm_layer2)
    if amount_lstm >= 3:
        net.add(lstm_layer3)

    net.add(tf.keras.layers.Dense(units=db_xy.amount_classes(), activation='softmax'))
    try:
        net.compile(optimizer='Adam',
                    loss='categorical_crossentropy',

                    metrics=['accuracy'])
    except ValueError as e:
        print(e)

    fit_res = net.fit(db_xy.train(), epochs=epochs, verbose=0, shuffle=False, workers=0,
                      callbacks=[early_stopper],
                      validation_data=db_xy.validation())
    # print(dict(var_acc=[np.var(fit_res.history['accuracy'])],
    #            mean_acc=[np.mean(fit_res.history['accuracy'])],
    #            mean_acc_val=[np.mean(fit_res.history['val_accuracy'])],
    #            best_acc_val=[np.max(fit_res.history['val_accuracy'])],
    #            best_acc=[np.max(fit_res.history['accuracy'])],
    #            best_acc_val_epoch=[np.argmax(fit_res.history['val_accuracy'])],
    #            best_acc_epoch=[np.argmax(fit_res.history['accuracy'])]))
    # best_accs.append(np.max(fit_res.history['accuracy']))

    plt.figure(0, dpi=720 / 9, figsize=(16, 9))
    acc_name = f'Accuracy {pose} lstm-{amount_lstm} lstm-{lstm_1_units} lstm-{lstm_2_units} lstm {lstm_3_units} ' \
               f'drop {dropout} rdrop {dropout_recurrent} dense {dense_units}'
    loss_name = f'Loss {pose} lstm-{amount_lstm} lstm-{lstm_1_units} lstm-{lstm_2_units} lstm {lstm_3_units} ' \
                f'drop {dropout} rdrop {dropout_recurrent} dense {dense_units}'
    plt.title(acc_name)
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
    plt.savefig('../first_batch_figs/' + acc_name + '.pdf')
    plt.close()

    plt.figure(0, dpi=720 / 9, figsize=(16, 9))
    plt.title(loss_name)
    plt.plot(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['val_loss'], label='val_loss')
    plt.scatter(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['val_loss'])
    plt.plot(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['loss'], label='loss')
    plt.scatter(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['loss'])
    plt.legend()
    plt.savefig('../first_batch_figs/' + loss_name + '.pdf')
    plt.close()

    return '../first_batch_figs/' + acc_name + '.pdf', '../first_batch_figs/' + loss_name + '.pdf', fit_res.history


def train_lstm_xy_pose_cnn(pose, amount_lstm, lstm_1_units, lstm_2_units, lstm_3_units, dropout, dropout_recurrent,
                           cnn_1_filters, cnn_2_filters, cnn_kernel):
    epochs = 10000000

    max_len_seq = db_xy.find_longest_sample()
    amount_joints_used = len(db_xy.joints_used()) - 1
    best_accs = []

    lstm_layer = \
        tf.keras.layers.LSTM(units=lstm_1_units, activation='tanh',
                             return_sequences=amount_lstm > 1,
                             dropout=dropout, recurrent_dropout=dropout_recurrent,
                             # input_shape=(max_len_seq, amount_joints_used)
                             )

    lstm_layer2 = \
        tf.keras.layers.LSTM(units=lstm_2_units if lstm_2_units is not None else 1, activation='tanh',
                             dropout=dropout, recurrent_dropout=dropout_recurrent,
                             return_sequences=amount_lstm > 2)

    lstm_layer3 = \
        tf.keras.layers.LSTM(units=lstm_3_units if lstm_3_units is not None else 1, activation='tanh',
                             dropout=dropout, recurrent_dropout=dropout_recurrent,
                             return_sequences=amount_lstm > 3)

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

    if amount_lstm >= 1:
        net.add(lstm_layer)
    if amount_lstm >= 2:
        net.add(lstm_layer2)
    if amount_lstm >= 3:
        net.add(lstm_layer3)

    net.add(tf.keras.layers.Dense(units=db_xy.amount_classes(), activation='softmax'))
    try:
        net.compile(optimizer='Adam',
                    loss='categorical_crossentropy',

                    metrics=['accuracy'])
    except ValueError as e:
        print(e)

    fit_res = net.fit(db_xy.train(), epochs=epochs, verbose=0, shuffle=False, workers=0,
                      callbacks=[early_stopper],
                      validation_data=db_xy.validation())
    # print(dict(var_acc=[np.var(fit_res.history['accuracy'])],
    #            mean_acc=[np.mean(fit_res.history['accuracy'])],
    #            mean_acc_val=[np.mean(fit_res.history['val_accuracy'])],
    #            best_acc_val=[np.max(fit_res.history['val_accuracy'])],
    #            best_acc=[np.max(fit_res.history['accuracy'])],
    #            best_acc_val_epoch=[np.argmax(fit_res.history['val_accuracy'])],
    #            best_acc_epoch=[np.argmax(fit_res.history['accuracy'])]))
    # best_accs.append(np.max(fit_res.history['accuracy']))

    plt.figure(0, dpi=720 / 9, figsize=(16, 9))
    acc_name = f'Accuracy {pose} lstm-{amount_lstm} lstm-{lstm_1_units} lstm-{lstm_2_units} lstm {lstm_3_units} ' \
               f'drop {dropout} rdrop {dropout_recurrent} cnn_kernel {cnn_kernel} filters {cnn_1_filters} ' \
               f'{cnn_2_filters}'

    loss_name = f'Loss {pose} lstm-{amount_lstm} lstm-{lstm_1_units} lstm-{lstm_2_units} lstm {lstm_3_units} ' \
                f'drop {dropout} rdrop {dropout_recurrent}  cnn_kernel {cnn_kernel} filters {cnn_1_filters} ' \
                f'{cnn_2_filters}'
    plt.title(acc_name)
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
    plt.savefig('../../first_batch_figs/' + acc_name + '.pdf')
    plt.close()

    plt.figure(0, dpi=720 / 9, figsize=(16, 9))
    plt.title(loss_name)
    plt.plot(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['val_loss'], label='val_loss')
    plt.scatter(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['val_loss'])
    plt.plot(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['loss'], label='loss')
    plt.scatter(np.arange(len(fit_res.history['val_accuracy'])), fit_res.history['loss'])
    plt.legend()
    plt.savefig('../first_batch_figs/' + loss_name + '.pdf')
    plt.close()

    return '../first_batch_figs/' + acc_name + '.pdf', '../first_batch_figs/' + loss_name + '.pdf', fit_res.history



# %%

dt_pose = dict(
    dense_units=[90, 45],
    amount_lstm=[1, 2, 3],
    lstm_1_units=[90, 45],
    lstm_2_units=[30, 20],
    lstm_3_units=[15, 10],
    dropout=[0.0, 0.35],
    dropout_recurrent=[0.0, 0.35],
)

keys, vals = zip(*dt_pose.items())
all_params_product = [{key: x[it] for it, key in enumerate(keys)} for x in list(itertools.product(*vals))]
merger = PdfFileMerger()

#####################
exp_df = pd.DataFrame()
df_exp = pd.read_csv('../../all_experiments_pose_angle_batch1.csv') \
    if os.path.exists('../../all_experiments_pose_angle_batch1.csv') else pd.DataFrame()
save_header = True
for param in tqdm(all_params_product):
    if df_exp.shape[0] != 0:
        if str(param) in df_exp.exp_name.unique().tolist():
            continue

    tf.keras.backend.clear_session()
    acc_pdf_file, loss_pdf_file, history = train_lstm('angle', **param)
    merger.append(acc_pdf_file)
    merger.append(loss_pdf_file)

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
    )).to_csv('../../all_experiments_pose_angle_batch1.csv', mode='a', header=save_header)
    save_header = False
#############


exp_df = pd.DataFrame()
save_header = True
df_exp = pd.read_csv('../../all_experiments_pose_xy_batch1.csv') \
    if os.path.exists('../../all_experiments_pose_xy_batch1.csv') else pd.DataFrame()

for param in tqdm(all_params_product):
    if df_exp.shape[0] != 0:
        if str(param) in df_exp.exp_name.unique().tolist():
            continue

    tf.keras.backend.clear_session()
    acc_pdf_file, loss_pdf_file, history = train_lstm_xy_pose('xy-pose', **param)
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
    )).to_csv('../../all_experiments_pose_xy_batch1.csv', mode='a', header=save_header)
    save_header = False

##########

dt_pose = dict(
    cnn_1_filters=[16, 8],
    cnn_2_filters=[16, 8],
    cnn_kernel=[5, 9],
    amount_lstm=[1, 2, 3],
    lstm_1_units=[90, 45],
    lstm_2_units=[30, 20],
    lstm_3_units=[15, 10],
    dropout=[0.0, 0.35],
    dropout_recurrent=[0.0, 0.35],
)

keys, vals = zip(*dt_pose.items())
all_params_product = [{key: x[it] for it, key in enumerate(keys)} for x in list(itertools.product(*vals))]

###########
exp_df = pd.DataFrame()
df_exp = pd.read_csv('../../all_experiments_pose_xy_cnn_batch1.csv') \
    if os.path.exists('../../all_experiments_pose_xy_cnn_batch1.csv') else pd.DataFrame()

save_header = True
for param in tqdm(all_params_product):
    if df_exp.shape[0] != 0:
        if str(param) in df_exp.exp_name.unique().tolist():
            continue

    tf.keras.backend.clear_session()
    acc_pdf_file, loss_pdf_file, history = train_lstm('angle', **param)

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
    )).to_csv('../../all_experiments_pose_xy_cnn_batch1.csv', mode='a', header=save_header)
    save_header = False