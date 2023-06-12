import tensorflow as tf
from libras_classifiers.librasdb_loader_online_sequences import DBLoaderOnlineSequences

db = DBLoaderOnlineSequences(db_path="../libras-db-folders-online-debug", batch_size=1)


max_len_seq, amount_joints_used, = 60, 22

net = tf.keras.Sequential()
net.add(
    tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten(), input_shape=(max_len_seq, amount_joints_used, 2)
    )
)
net.add(tf.keras.layers.LSTM(units=10))
net.add(tf.keras.layers.LSTM(units=4, activation="softmax"))
net.compile(optimizer="Adam", loss="categorical_crosentropy", metrics=["accuracy"])

net.fit(db, epochs=1, shuffle=False, workers=1)
