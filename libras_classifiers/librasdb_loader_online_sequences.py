import sys
sys.path.append('.')

import numpy as np
import itertools
from libras_classifiers.librasdb_loaders import DBLoader2NPY
import os
import pandas as pd
from tensorflow.keras.utils import Sequence


# libras-db-folders-online-debug

class DBLoaderOnlineSequences:

    def __init__(self, db_path, batch_size, joints_2_use=None, angle_pose=False, shuffle=False, add_derivatives=False,
                 make_k_fold=False, k_fold_amount=None, only_that_classes=None, scaler_cls=None, const_none_angle_rep=0,
                 const_none_xy_rep=np.array([0, 0, 0])):
        """
        Parameters
        ----------
        db_path : str
            Path para os diretorios onde se encontram as poses.

        joints_2_use: List or None
            lista com nome das juntas a serem usadas. Caso none, todas as juntas seram usadas.

        make_k_fold : bool
            Informa a classe se existe necessidade de fazer um k-fold

        k_fold_amount : float
            valor da divisão de samples por classe que ficará para teste
            no k-fold.
        """
        self.count = 0
        self.none_rep = const_none_xy_rep if angle_pose else const_none_angle_rep
        self.db_path = db_path
        self.batch_size = batch_size
        self.make_k_fold = make_k_fold
        self.k_fold_amount = k_fold_amount
        self.joints_2_use = joints_2_use
        self.add_derivatives = add_derivatives
        self.scaler_cls = scaler_cls

        self.cls_dirs = []
        self.only_that_classes = only_that_classes
        self.shuffle = shuffle

        self.max_sample_length = None
        self.samples, self.cls_dirs = self.read_all_db_folders(db_path, only_that_classes)

        print(self.samples)

    def read_all_db_folders(self, db_path, only_that_classes):
        """

        Parameters
        ----------
        db_path
        only_that_classes
        angle_or_xy
        custom_internal_dir

        Returns
        -------

        """

        class_dict = {}
        cls_dirs = DBLoader2NPY.get_classes_dir(db_path, only_that_classes)
        samples_loaded = []
        class_names_in_pairs = []
        for it, class_dir in enumerate(cls_dirs):
            # aqui setamos o diretorio interno caso exista um, no caso de não existir seguimos o padrão de: xy_pose
            # para poses completas, angle_pose para as poses com angulos, no_hand-xy_pose para as poses xy sem as mãos.
            all_samples_in_class = [os.path.join(class_dir, x) for x in os.listdir(class_dir)]
            only_class_names = class_dir.replace('\\', '/').split('/')[-1].split('---')
            for cls in only_class_names:
                if cls not in class_dict:
                    class_dict.update({cls: len(class_dict)})

            for folder_sample in all_samples_in_class:
                sings_at_folder_sample = [os.path.join(folder_sample, x)
                                          for x in os.listdir(folder_sample) if '.csv' in x]
                sings_at_folder_sample = sorted(sings_at_folder_sample, key=lambda x: x.split('---')[2])
                classes = [class_dict[x.replace('\\', '/').split('/')[-1].split('---')[1]]
                           for x in sings_at_folder_sample]

                sings_at_folder_sample = list(map(pd.read_csv, sings_at_folder_sample))
                samples_loaded.append(dict(signs=sings_at_folder_sample, classes=classes))

        # achar o sample com maior frames
        samples_frame = list(map(lambda x: [sp.shape[0] for sp in x['signs']], samples_loaded))
        samples_frame = list(itertools.chain(*samples_frame))
        max_frame_count = max(samples_frame)
        self.max_sample_length = max_frame_count

        joints_name = samples_loaded[0]['signs'][0].keys() if self.joints_2_use is None else self.joints_2_use
        # padding em todos com numero de frames inferior ao maior sample
        for s_idx in range(len(samples_loaded)):
            for s in range(len(samples_loaded[s_idx]['signs'])):
                amount_absent_frames = max_frame_count - samples_loaded[s_idx]['signs'][s].shape[0]
                if amount_absent_frames > 0:
                    empty_df = pd.DataFrame({
                        f: [self.none_rep] * amount_absent_frames
                        for f in joints_name
                    })
                    samples_loaded[s_idx]['signs'][s] = empty_df.append(samples_loaded[s_idx]['signs'][s])
        # e ajustar os Y para frame
        return samples_loaded, cls_dirs

    def joints_used(self):
        """
        Mostra quais juntas estão sendo usadas.

        Returns
        -------
        joint_names: List
            Lista com nome das juntas utilizadas no carregador.
        """
        joint_names = self.samples[0][0].keys() if self.joints_2_use is None else self.joints_2_use
        return joint_names

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

