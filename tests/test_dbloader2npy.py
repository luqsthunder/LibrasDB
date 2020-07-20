from libras_classifiers.librasdb_loaders import DBLoader2NPY

#import tensorflow as tf


class TestDBLoader2npy:
    db = None
    BATCH_SIZE = 8

    def setup(self):
        self.db = DBLoader2NPY('../libras-db-folders',
                               angle_pose=False,
                               no_hands=False,
                               batch_size=self.BATCH_SIZE)
        self.dbxy = DBLoader2NPY('../libras-db-folders',
                                 batch_size=self.BATCH_SIZE,
                                 angle_pose=False,
                                 no_hands=False,)

    def test_constructor(self):
        try:
            self.db = DBLoader2NPY('../libras-db-folders', angle_pose=False,
                                   no_hands=False,
                                   batch_size=self.BATCH_SIZE)
            assert len(self.db.cls_dirs) > 0
            assert len(self.db.samples_path) > 0

            self.dbxy = DBLoader2NPY('../libras-db-folders',
                                     no_hands=False,
                                     batch_size=self.BATCH_SIZE,
                                     angle_pose=False)
            assert len(self.db.cls_dirs) > 0
            assert len(self.db.samples_path) > 0

        except BaseException:
            assert False

    def test_max_len(self):
        try:
            max_len = self.db.find_longest_sample()
            assert max_len > 0

            max_len = self.dbxy.find_longest_sample()
            assert max_len > 0

        except BaseException:
            assert False

    def test_joint_names(self):
        try:
            keys = self.db.joints_used()
            assert len(keys) > 0

            keys = self.dbxy.joints_used()
            assert len(keys) > 0

        except BaseException:
            assert False

    def test_amount_classes(self):
        try:
            cls_amount = self.db.amount_classes()
            assert cls_amount > 0

            cls_amount = self.dbxy.amount_classes()
            assert cls_amount > 0

        except BaseException:
            assert False

    def test_can_classify(self):
        max_len_seq = self.db.find_longest_sample()
        amount_joints_used = len(self.db.joints_used()) - 2

        lstm_layer = \
            tf.keras.layers.LSTM(units=120, activation='tanh',
                                 recurrent_activation='sigmoid',
                                 return_sequences=False,
                                 input_shape=(max_len_seq, amount_joints_used))

        amount_classes = self.db.amount_classes()
        net = tf.keras.Sequential()
        net.add(lstm_layer)
        net.add(tf.keras.layers.Dense(units=amount_classes,
                                      activation='softmax'))
        try:
            net.compile(optimizer='Adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        except ValueError as e:
            print(e)
            assert False

        try:
            self.db.fill_samples_absent_frames_with_na()
            net.fit(x=self.db, epochs=3, verbose=2)

        except (RuntimeError, ValueError) as e:
            print('Error in Fit :', e)
            assert False

        assert True

    def test_can_classify_pytorch(self):
        assert True

    def test_class_weigths(self):
        try:
            dt_weigths = self.db.make_class_weigth()
            assert len(dt_weigths) > 0
        except BaseException:
            assert False
