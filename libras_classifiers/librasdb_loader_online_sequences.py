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

    def __init__(self, db_path, batch_size, amount_frames_per_sequence=None, joints_2_use=None, angle_pose=False, shuffle=False, add_derivatives=False,
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

        self.pad_class = 0

        self.max_sample_length = None
        self.angle_pose = angle_pose
        self.amount_frames_per_sequence = amount_frames_per_sequence
        self.samples, self.classes = self._read_all_db_folders_internal(self.db_path, self.only_that_classes)

    def _read_all_db_folders_internal(self, db_path, only_that_classes, clean_nan=True):
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

        class_dict = {'---PADDING---': 0}
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
                # os sinais precisam ser ordenados pelo atributo begin ou informação de begin no seu arquivo.
                sings_at_folder_sample = sorted(sings_at_folder_sample, key=lambda x: x.split('---')[2])
                classes = [class_dict[x.replace('\\', '/').split('/')[-1].split('---')[1]]
                           for x in sings_at_folder_sample]

                sings_at_folder_sample = list(map(pd.read_csv, sings_at_folder_sample))
                samples_loaded.append(dict(signs=sings_at_folder_sample, classes=classes))

        # achar o sample com maior frames
        samples_frame = list(map(lambda x: sum([sp.shape[0] for sp in x['signs']]), samples_loaded))
        max_frame_count = max(samples_frame)
        self.max_sample_length = max_frame_count

        samples_join = []
        classes_per_frame = []
        for sample in samples_loaded:
            x = pd.DataFrame()
            y = []
            for signs_in_sample, cls in zip(sample['signs'], sample['classes']):
                x = x.append(signs_in_sample)
                y.extend([cls] * signs_in_sample.shape[0])

            samples_join.append(x)
            classes_per_frame.append(y)

        joints_in_use = samples_join[0].keys() if self.joints_2_use is None else self.joints_2_use
        for idx, sample in enumerate(samples_join):
            amount_absent_frames = self.max_sample_length - sample.values.shape[0]

            if amount_absent_frames > 0:
                empty_df = pd.DataFrame({
                    f: [self.none_rep] * amount_absent_frames
                    for f in joints_in_use
                })
                samples_join[idx] = empty_df.append(sample)
                classes_per_frame[idx] = [self.pad_class] * amount_absent_frames + classes_per_frame[idx]

        self.amount_frames_per_sequence = self.amount_frames_per_sequence \
            if self.amount_frames_per_sequence is not None else self.max_sample_length

        online_sequence = []
        online_sequence_classes = []

        for sample, idx in zip(samples_join, classes_per_frame):
            curr_frame_amount_online = []
            curr_frame_amount_online_classes = []
            for frame_idx in range(sample.shape[0]):
                curr_frame_amount_online.append(sample.iloc[frame_idx])
                print(curr_frame_amount_online)




        return samples_join, classes_per_frame

    def __clean_sample(self, sample):
        """
        Remove da amostra atual

        Parameters
        ----------
        sample : Pandas Dataframe
            Dataframe que deve conter uma amostra de um video ja previamente
            extraido ou convertido para angulos.

        Returns
        -------

        Pandas Dataframe
            dataframe com nan preenchidos pelo valor padrão definido no
            construtor.
        """

        # No formato XY cada junta da pose é uma str de um np.array.
        # Como vai ser convertido de str para vetor apos ser carregado, isso
        # evita o erro do pandas não aceitar preencher nulos com um np.array.
        xy_str_rep = str(self.const_none_xy_rep)
        m_sample = sample.copy()

        if not self.angle_pose:
            zero_rep = str(np.array([0., 0., 0.]))
            m_sample = m_sample.replace(zero_rep, np.nan)


        return m_sample.fillna(self.const_none_angle_rep
                               if self.angle_pose else xy_str_rep)

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

    def batch_load_samples(self, samples_idxs, as_npy=True, clean_nan=True):
        """
        Parameters
        ----------
        samples_idxs : list
            Lista dos ids a serem retornados.

        as_npy: bool
            Boleano que indica se deve retornar um dataframe ou uma array de
            numpy arrays.

        clean_nan : bool
            Boleano indicando se deve ser limpo as amostras com nulos.

        pbar: tqdm
            Progress bar to use while loading samples

        Returns
        -------

        """

        X = []
        Y = []
        for idx in samples_idxs:
            pass

        return X, Y

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


if __name__ == '__main__':
    DBLoaderOnlineSequences()