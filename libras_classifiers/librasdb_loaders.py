import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from pose_extractor.all_parts import *
import sklearn as sk


class DBLoader2NPY(tf.keras.utils.Sequence):
    """
    Responsavem por carregar as poses dos CSVs para formato usavel pelo keras
    pytorch ou outros frameworks que utilizam numpy.
    Responsavel também por pazer limpeza nos dados, representação dos nulos.
    Diferenciar representação das poses XY para ângulo.
    """

    def __init__(self, db_path, batch_size, angle_pose=True, no_hands=True, joints_2_use=None,
                 maintain_memory=True, make_k_fold=False, k_fold_amount=None, const_none_angle_rep=0,
                 const_none_xy_rep=np.array([0, 0, 0])):
        """
        Parameters
        ----------
        db_path : str
            Path para os diretorios onde se encontram as poses.

        angle_pose : bool
            boleano para informar se carrega as poses XY ou em formato de
            ângulo. True para pose em formato de ângulo, Falso para XY.

        no_hands : bool
            Caso exista pose das mãos.

        joints_2_use: List or None
            lista com nome das juntas a serem usadas. Caso none, todas as juntas seram usadas.

        maintain_memory : bool
            carregar todo dataset na memoria

        make_k_fold : bool
            Informa a classe se existe necessidade de fazer um k-fold

        k_fold_amount : float
            valor da divisão de samples por classe que ficará para teste
            no k-fold.

        const_none_angle_rep : float
            Reresentação que deve ser usada caso um ângulo nulo exista.

        const_none_xy_rep : float
            Reresentação que deve ser usada caso uma pose xy nula exista.

        """
        self.db_path = db_path; self.const_none_angle_rep = const_none_angle_rep
        self.const_none_xy_rep = const_none_xy_rep
        self.angle_pose = angle_pose
        self.batch_size = batch_size
        self.make_k_fold = make_k_fold; self.k_fold_amount = k_fold_amount
        self.joints_2_use = joints_2_use

        angle_or_xy = 'angle' if angle_pose else 'xy'
        angle_or_xy = 'no_hands-' + angle_or_xy \
            if no_hands else 'hands-' + angle_or_xy
        self.angle_or_xy = angle_or_xy
        self.cls_dirs = []

        try:
            self.cls_dirs = [os.path.join(db_path, x)
                             for x in os.listdir(db_path)]
            self.cls_dirs = list(filter(os.path.isdir, self.cls_dirs))
        except (FileNotFoundError, NotADirectoryError) as e:
            error_msg = '\n error in constructor DBLoader2NPY ' \
                        'using db_path {} \n '\
                        'using pose as {}'.format(db_path, angle_or_xy)
            print(e, error_msg)
            raise RuntimeError(error_msg)

        self.samples_path = []
        for it, class_dir in enumerate(self.cls_dirs):
            dir_to_walk = os.path.join(class_dir, angle_or_xy)
            all_samples_in_class = [os.path.join(dir_to_walk, x)
                                    for x in os.listdir(dir_to_walk)]
            all_samples_in_class = list(filter(os.path.isfile,
                                               all_samples_in_class))
            samples_class = [it] * len(all_samples_in_class)
            all_samples_in_class = list(zip(all_samples_in_class,
                                            samples_class))
            self.samples_path.extend(all_samples_in_class)

        self.maintain_memory = maintain_memory
        self.samples_memory = [None for _ in range(len(self.samples_path))]
        self.samples_memory_xy_npy = [None for _ in range(len(self.samples_path))]
        self.longest_sample = None
        self.longest_sample = self.find_longest_sample()
        self.weight_2_samples = None

        self.k_fold_iteration = 0
        self.k_folder = None
        if self.make_k_fold:
            self.k_folder = sk.model_selection.KFold(n_splits=self.k_fold_amount)
            self.X_train, self.X_test = self.k_fold_samples()

    def k_fold_samples(self):
        return next(self.k_folder.split(self.samples_memory))

    def find_longest_sample(self):
        """

        Returns
        -------

        int
            Tamanho da amostra com maior quantidade de frames.
        """
        if self.longest_sample is None:
            db_idx = [x for x in range(len(self.samples_path))]
            len_size, y = self.batch_load_samples(db_idx, as_npy=False, pbar=tqdm())
            len_size = max(len_size, key=lambda x: x.shape[0]).shape[0]
            return len_size
        else:
            return self.longest_sample

    def make_class_weigth(self):
        """

        Returns
        -------
        Dict
            Pesos para ser usado no keras para classificação.
        """
        db_idx = [x for x in range(len(self.samples_path))]
        len_size, y = self.batch_load_samples(db_idx, as_npy=False)
        class_count = [0] * len(self.cls_dirs)
        cls_as_labels = []
        for cls in y:
            cls_num = int(np.argmax(cls))
            class_count[cls_num] += 1
            cls_as_labels.append(cls_num)

        cls_as_labels = np.array(cls_as_labels)

        cls_weight = \
            sk.utils.class_weight.compute_class_weight('balanced',
                                                       np.unique(cls_as_labels),
                                                       cls_as_labels)
        return cls_weight

    def db_length(self):
        """
        Returns
        -------

        int
            Quantidade de amostras na base de dados.
        """
        return len(self.samples_path)

    def joints_used(self):
        """
        Mostra quais juntas estão sendo usadas.

        Returns
        -------
        joint_names: List
            Lista com nome das juntas utilizadas no carregador.
        """
        joint_names = self.__load_sample_by_pos(0)[0].keys() if self.joints_2_use is None else self.joints_2_use
        return joint_names

    def amount_classes(self):
        return len(self.cls_dirs)

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

    def __load_sample(self, class_name, num, clean_nan=True):
        """

        Parameters
        ----------
        class_name : str
            Nome do sinal ou classe.

        num : int
          Numero ou posição do sinal dentro da classe/pasta.

        Returns
        -------
        Pandas Dataframe
           Dataframe contendo o sinal em poses.
        """
        cls_begin_pos = 0
        for x in self.cls_dirs:
            if x[len(self.db_path):] == class_name:
                break
            dir_to_walk = os.path.join(x, self.angle_or_xy)
            cls_begin_pos += len(os.listdir(dir_to_walk))

        return self.__load_sample_by_pos(cls_begin_pos + num,
                                         clean_nan=clean_nan)

    @staticmethod
    def parse_npy_vec_str(str_array_like):
        if not isinstance(str_array_like, str):
            return str_array_like

        res = str_array_like[1:len(str_array_like) - 1].split(' ')
        recovered_np_array = []
        for r in res:
            if r == '' or ' ' in r:
                continue
            f = float(r)
            recovered_np_array.append(f)
        return np.array(recovered_np_array)

    def __load_sample_by_pos(self, pos, clean_nan=True) -> (pd.DataFrame, np.array):
        """

        Parameters
        ----------
        pos : int
            Posição da amostra.

        Returns
        -------
        Pandas Dataframe, np.array
           Dataframe contendo o sinal em poses e vector indicando a classe da
           pose.
        """
        sample = None
        if self.samples_memory[pos] is None and self.maintain_memory:
            sample = pd.read_csv(self.samples_path[pos][0])
            sample = sample.set_index('Unnamed: 0')
            if clean_nan:
                sample = self.__clean_sample(sample)

            if not self.angle_pose:
                sample = sample.applymap(self.parse_npy_vec_str)

            self.samples_memory[pos] = sample

        elif self.samples_memory is not None and self.maintain_memory:
            sample = self.samples_memory[pos]
        else:
            sample = pd.read_csv(self.samples_path[pos][0])
            sample = sample.set_index('Unnamed: 0')
            if clean_nan:
                sample = self.__clean_sample(sample)

            if not self.angle_pose:
                sample = sample.applymap(self.parse_npy_vec_str)

        class_vec = np.zeros((len(self.cls_dirs), ))
        idx_test = self.samples_path[pos][1]
        class_vec[idx_test] = 1
        return sample, class_vec.reshape((-1, 1))

    def fill_samples_absent_frames_with_na(self):
        """

        Parameters
        ----------
        sample : Pandas DataFrame
        """

        for it, sample in enumerate(self.samples_memory):
            if sample is None:
                continue

            amount_absent_frames = self.longest_sample - sample.values.shape[0]

            if amount_absent_frames > 0:

                na_rep = self.const_none_angle_rep if self.angle_pose \
                    else self.const_none_xy_rep

                empty_df = pd.DataFrame({
                    f: [na_rep] * amount_absent_frames
                    for f in self.joints_used()
                })

                self.samples_memory[it] = sample.append(empty_df,
                                                        ignore_index=True)

    @staticmethod
    def __stack_xy_pose_2_npy(sample: pd.DataFrame):
        """
        Enfileira os valores de uma amostra de pose-XY em um formato numpy. Descartando o atributo relacionado aos
        quadros (frames).

        O formato de cada amostra, das Poses-XY, não converte diretamente para um np.array, pois para cada célula o
        dataframe é tratado como um objeto separado e a coerção converte elas para um np.array de objetos. Logo Para
        cada objeto ser tratado como parte do np.array final é necessario enfileirar cada um deles, de modo que,
        existem 61 juntas por amostra, uma quantidade X de frames por amostra, com a amostra com maior numero de frames
        pussindo 2860, logo, cada amostra deve ter um formato (shape) igual a (numero de frames, numero de juntas, 2).


        Parameters
        ----------
        sample: pd.DataFrame

        Returns
        -------
        np.array
            a amostra convertida para np.array
        """
        sample_in_npy = []
        for row in sample.iterrows():
            row = row[1]
            sample_in_npy.append(np.stack(row.values[1:], axis=0))

        sample_in_npy = np.stack(sample_in_npy, axis=0)

        return np.stack(sample_in_npy, axis=0)

    def batch_load_samples(self, samples_idxs, as_npy=True, clean_nan=True, pbar: tqdm = None):
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
            if not self.angle_pose and self.joints_2_use is not None:
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
                X.append(x.values if as_npy else x)
            Y.append(y)

            if pbar is not None:
                pbar.update(1)
                pbar.refresh()

        if as_npy:
            shape_before = Y[0].shape
            Y = np.concatenate(Y).reshape(len(samples_idxs), shape_before[0], shape_before[1])

        if as_npy and self.angle_pose:
            shape_before = X[0].shape
            new_shape = [len(samples_idxs)]
            new_shape.extend(list(shape_before))
            new_shape = tuple(new_shape)
            X = np.concatenate(X).reshape(new_shape)

        return X, Y

    def __getitem__(self, index):
        beg = index * self.batch_size

        end = (index + 1) * self.batch_size \
            if (index + 1) * self.batch_size < self.db_length() \
            else self.db_length()

        x, y = self.batch_load_samples(list(range(beg, end)))
        return x, y, [None]

    def __len__(self):
        """

        Returns
        -------
        int
            Tamanho do lote
        """
        return math.ceil(self.db_length() / self.batch_size)

