import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tqdm.auto import tqdm
from pose_extractor.all_parts import *
import sklearn as sk
from sklearn.model_selection import StratifiedKFold
import random


class DBLoader2NPY(Sequence):
    """
    Responsavem por carregar as poses dos CSVs para formato usavel pelo keras
    pytorch ou outros frameworks que utilizam numpy.
    Responsavel também por pazer limpeza nos dados, representação dos nulos.
    Diferenciar representação das poses XY para ângulo.
    """

    def __init__(self, db_path, batch_size, angle_pose=True, no_hands=True, joints_2_use=None,
                 shuffle=False, test_size=0.3, add_angle_derivatives=False,
                 maintain_memory=True, make_k_fold=False, k_fold_amount=None, only_that_classes=None,
                 scaler_cls=None, not_use_pbar_in_load=False, custom_internal_dir=None, const_none_angle_rep=0,
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
        self.count = 0
        self.db_path = db_path; self.const_none_angle_rep = const_none_angle_rep
        self.const_none_xy_rep = const_none_xy_rep
        self.angle_pose = angle_pose
        self.batch_size = batch_size
        self.make_k_fold = make_k_fold; self.k_fold_amount = k_fold_amount
        self.joints_2_use = joints_2_use
        self.add_angle_derivatives = add_angle_derivatives
        self.scaler_cls = scaler_cls

        angle_or_xy = 'angle' if angle_pose else 'xy'
        angle_or_xy = 'no_hands-' + angle_or_xy \
            if no_hands else 'hands-' + angle_or_xy
        self.angle_or_xy = angle_or_xy
        self.cls_dirs = []
        self.only_that_classes = only_that_classes
        self.shuffle = shuffle

        self.samples_path, self.cls_dirs = DBLoader2NPY.read_all_db_folders(db_path, only_that_classes, angle_or_xy,
                                                                            custom_internal_dir)
        self.all_samples_separated_ids = DBLoader2NPY.separate_samples_with_their_real_ids(self.samples_path,
                                                                                           len(self.cls_dirs))
        self.all_samples_ids = []

        self.maintain_memory = maintain_memory
        self.samples_memory = [None for _ in range(len(self.samples_path))]
        self.samples_memory_xy_npy = [None for _ in range(len(self.samples_path))]
        self.longest_sample = None
        self.longest_sample = self.find_longest_sample(no_pbar=not_use_pbar_in_load)
        self.weight_2_samples = None

        if self.angle_pose and self.add_angle_derivatives and self.joints_2_use is not None:
            self.joints_2_use = self.joints_2_use + [f'DP-DT-{x}' for x in self.joints_2_use if x != 'frame']

        self.k_fold_iteration = 0
        self.k_folder = None

        self.y = np.array([x[1] for x in self.samples_path])
        self.x = np.array([it for it in range(len(self.samples_path))])
        self._train_ids, self._val_ids, self._y_train, self._y_test = \
            sk.model_selection.train_test_split(self.x, self.y, test_size=test_size, random_state=5, stratify=self.y)
        if self.make_k_fold:
            self.k_folder = StratifiedKFold(n_splits=self.k_fold_amount, shuffle=False)
           #self.train_set, self.val_set = self.k_fold_samples()

        print('separated_samples')

    @staticmethod
    def separate_samples_with_their_real_ids(all_samples, amount_classes, get_y_array=False):
        all_sample_separated = [
            [key for key, x in enumerate(all_samples) if x[1] == it] for it in range(amount_classes)
        ]

        if get_y_array:
            pass

        return all_sample_separated

    @staticmethod
    def read_all_db_folders(db_path, only_that_classes, angle_or_xy, custom_internal_dir=None):
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
        try:
            cls_dirs = [x for x in os.listdir(db_path)]
            if only_that_classes is not None:
                cls_dirs = list(filter(lambda x: x in only_that_classes, cls_dirs))
            if len(cls_dirs) == 0:
                raise RuntimeError(f'Classes that was required it was not found. This was required classes that was '
                                   f'given {only_that_classes}, and this is the db_path: {db_path}')

            cls_dirs = [os.path.join(db_path, x) for x in cls_dirs]
            cls_dirs = list(filter(os.path.isdir, cls_dirs))
        except (FileNotFoundError, NotADirectoryError) as e:
            error_msg = '\n error in constructor DBLoader2NPY ' \
                        'using db_path {} \n ' \
                        'using pose as {}'.format(db_path, angle_or_xy)
            print(e, error_msg)
            raise RuntimeError(error_msg)

        samples_path = []
        for it, class_dir in enumerate(cls_dirs):
            # aqui setamos o diretorio interno caso exista um, no caso de não existir seguimos o padrão de: xy_pose
            # para poses completas, angle_pose para as poses com angulos, no_hand-xy_pose para as poses xy sem as mãos.
            internal_dir = angle_or_xy if custom_internal_dir is None else custom_internal_dir
            dir_to_walk = os.path.join(class_dir, internal_dir)
            all_samples_in_class = [os.path.join(dir_to_walk, x)
                                    for x in os.listdir(dir_to_walk)]
            all_samples_in_class = list(filter(os.path.isfile,
                                               all_samples_in_class))
            samples_class = [it] * len(all_samples_in_class)
            all_samples_in_class = list(zip(all_samples_in_class,
                                            samples_class))
            samples_path.extend(all_samples_in_class)

        return samples_path, cls_dirs

    def k_fold_samples(self):
        return next(self.k_folder.split(self.all_samples_separated_ids, y=self.y))

    def find_longest_sample(self, no_pbar=False):
        """

        Returns
        -------

        int
            Tamanho da amostra com maior quantidade de frames.
        """
        if self.longest_sample is None:
            db_idx = [x for x in range(len(self.samples_path))]
            len_size, y = self.batch_load_samples(db_idx, as_npy=False, pbar=tqdm() if not no_pbar else None)
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

            if self.scaler_cls is not None and not self.angle_pose:
                sample = DBLoader2NPY.scale_single_sample(sample, self.scaler_cls)

            if self.add_angle_derivatives and self.angle_pose:
                sample = self.make_angle_derivative_sample(sample)

            self.samples_memory[pos] = sample

        elif self.samples_memory is not None and self.maintain_memory:
            sample = self.samples_memory[pos]
        else:
            sample = pd.read_csv(self.samples_path[pos][0])
            sample = sample.set_index('Unnamed: 0')
            if clean_nan:
                sample = self.__clean_sample(sample)

            if self.scaler_cls is not None and not self.angle_pose:
                sample = DBLoader2NPY.scale_single_sample(sample, self.scaler_cls)

            if self.add_angle_derivatives and self.angle_pose:
                sample = self.make_angle_derivative_sample(sample)

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

                self.samples_memory[it] = empty_df.append(sample,
                                                          ignore_index=True)

    @staticmethod
    def scale_single_sample(sample, scale_cls, scale_kwargs={}):
        all_frames = sample['frame'].unique().tolist()
        for frame in all_frames:
            curr_pose = sample[sample.frame == frame]
            all_x_from_this_frame = [x[0] for x in curr_pose.values[0][2:]]
            all_y_from_this_frame = [x[1] for x in curr_pose.values[0][2:]]
            xscaler = scale_cls(**scale_kwargs)
            res_x = xscaler.fit(np.array(all_x_from_this_frame).reshape((-1, 1))) \
                           .transform(np.array(all_x_from_this_frame).reshape((-1, 1)))

            res_y = xscaler.fit(np.array(all_y_from_this_frame).reshape((-1, 1))) \
                           .transform(np.array(all_y_from_this_frame).reshape((-1, 1)))

            for it in range(len(curr_pose.values[0][2:])):
                curr_pose.values[0][it + 2][:2] = [res_x[it], res_y[it]]

        return sample

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
        # um row do sample é um frame apenas, como queremos desconsiderar os frames para o classificador, pegamos o
        # row da posição 1 em diante pois o "frame" esta na posição 0 do row
        sample_to_npy = sample.drop(columns=["frame"])
        for row in sample_to_npy.iterrows():
            row = row[1]
            sample_in_npy.append(np.stack(row.values, axis=0))

        sample_in_npy = np.stack(sample_in_npy, axis=0)

        return np.stack(sample_in_npy, axis=0)

    def make_angle_derivative_sample(self, sample, padding='zeros') -> pd.DataFrame:
        """

        Parameters
        ----------
        sample
        padding

        Returns
        -------

        """
        # colocar todos os dados da amostra em um dicionario para construir um novo dataframe.
        sample_data_in_dict = {key: sample[key].values.tolist() for key in sample.keys()}

        # obter todas as linhas da amostra para iterar nelas na construição das derivadas
        all_rows_in_sample = [x[1] for x in sample.iterrows()]
        new_keys = all_rows_in_sample[1] - all_rows_in_sample[0]
        sample_data_in_dict.update({
            f'DP-DT-{key}': [] for key in new_keys.keys() if key != 'frame'
        })

        for it in range(0, sample.shape[0] - 1):
            # obtém cada linha para calcular a derivada finita.
            row_0 = all_rows_in_sample[it]
            row_1 = all_rows_in_sample[it + 1]

            # derivada é construida aki
            new_row_dt = row_1 - row_0
            for key in new_row_dt.keys():
                if key == 'frame':
                    continue

                sample_data_in_dict[f'DP-DT-{key}'].append(new_row_dt[key])

        if padding == 'zeros':
            for key in new_keys.keys():
                if key == 'frame':
                    continue
                sample_data_in_dict[f'DP-DT-{key}'].append(0)
        elif padding == 'cut_last':
            for key in sample.keys():
                if key == 'frame':
                    continue
                sample_data_in_dict[key].pop()
        else:
            raise TypeError(f'worng padding type PADDING TYPE -> {padding}')

        return pd.DataFrame(sample_data_in_dict)

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

        if as_npy and not self.angle_pose:
            X = np.stack(X, axis=0)

        return X, Y

    def __getitem__(self, index):
        beg = index * self.batch_size

        end = np.min([(index + 1) * self.batch_size, self.db_length()])

        x, y = self.batch_load_samples(self.x[beg: end])
        return x, y, [None]

    def __len__(self):
        """

        Returns
        -------
        int
            Tamanho do lote
        """
        return math.ceil(self.db_length() / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.x)

    def validation(self):
        return InternalBaseKerasIterator(self, self._val_ids)

    def train(self):
        return InternalBaseKerasIterator(self, self._train_ids)


class InternalBaseKerasIterator(Sequence):
    def __init__(self, parent, ids):
        self.parent = parent
        self.ids = ids

    def __getitem__(self, index):
        beg = index * self.parent.batch_size

        end = np.min([(index + 1) * self.parent.batch_size, len(self.ids)])

        x, y = self.parent.batch_load_samples(self.ids[beg: end])

        return x, y #[None] if tf.__version__.split('.')[1] == 1 else x, y

    def __len__(self):
        return math.ceil(len(self.ids) / self.parent.batch_size)

    def on_epoch_end(self):
        np.random.shuffle(self.ids)
