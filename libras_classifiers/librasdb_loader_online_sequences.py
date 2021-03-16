import sys
sys.path.append('.')

from libras_classifiers.librasdb_loaders import DBLoader2NPY
from tensorflow.keras.utils import Sequence


class DBLoaderOnlineSequences(DBLoader2NPY):



    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def on_epoch_end(self):
        pass

    def validation(self):
        return



class InternalKerasSequenceOnlineSequence(Sequence):
    pass