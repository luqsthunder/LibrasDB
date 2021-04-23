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

                    if clean_nan:
                        sample = self.__clean_sample(sample)

                    if not self.angle_pose:
                        sample = sample.applymap(DBLoader2NPY.parse_npy_vec_str)

                    if self.scaler_cls is not None and not self.angle_pose:
                        sample = DBLoader2NPY.scale_single_sample(sample, self.scaler_cls)

                    if self.add_derivatives and self.angle_pose:
                        sample = self.make_angle_derivative_sample(sample)

                    if self.add_derivatives and not self.angle_pose:
                        sample = self.make_xy_derivative_sample(sample)

        # e ajustar os Y para frame
        return samples_loaded, cls_dirs

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

        if pbar is not None:
            pbar.reset(total=len(samples_idxs))
            pbar.set_description('loading samples')
        X = []
        Y = []
        for idx in samples_idxs:
            x, y = self.__load_sample_by_pos(idx, clean_nan=clean_nan)
            if self.joints_2_use is not None:
                x = x[self.joints_2_use]
            if not self.angle_pose:
                x = x.applymap(lambda c: c[:2] if type(c) is np.ndarray else c)

                if self.samples_memory_xy_npy[idx] is None and as_npy:
                    x = self.__stack_xy_pose_2_npy(x)
                    self.samples_memory_xy_npy[idx] = x
                elif self.samples_memory_xy_npy[idx] is not None and as_npy:
                    x = self.samples_memory_xy_npy[idx]

                X.append(x)
            else:
                X.append(x.drop(columns=['frame']).values if as_npy else x)
            Y.append(y)

        if as_npy:
            shape_before = Y[0].shape
            Y = np.concatenate(Y).reshape(len(samples_idxs), shape_before[0], shape_before[1])

        if as_npy and self.angle_pose:
            shape_before = X[0].shape
            new_shape = [len(samples_idxs)]
            new_shape.extend(list(shape_before))
            new_shape = tuple(new_shape)
            X = np.concatenate(X).reshape(new_shape)

        if as_npy and not self.angle_pose:
            X = np.stack(X, axis=0)

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

