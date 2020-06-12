from libras_classifiers.librasdb_loaders import DBLoader2NPY
from libras_classifiers.generate_dataframe_person_2_sign import \
    DataframePerson2Sign
import tensorflow as tf


class TestDataframePerson2Sign:

    db = None
    check_signer = None
    BATCH_SIZE = 8

    def setup(self):
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        self.db = DBLoader2NPY('C:/Users/lucas/Downloads/libras-db',
                               batch_size=self.BATCH_SIZE,
                               angle_pose=False)
        self.check_signer = DataframePerson2Sign(self.db)

    def test_constructor(self):
        try:
            sample, _ = self.db.batch_load_samples([1], as_npy=False)
            sample = sample[0]
            signer = DataframePerson2Sign(self.db, 'grad')
            signer.process_single_sample(sample)
            signer = DataframePerson2Sign(self.db, 'dist')
            signer.process_single_sample(sample)

            assert True
        except BaseException as e:
            print(e)
            assert False

    def test_process_single_sample(self):
        try:
            sample, _ = self.db.batch_load_samples([1], as_npy=False)
            self.check_signer.process_single_sample(sample[0])
            assert True

        except BaseException as e:
            print(e)
            assert False
