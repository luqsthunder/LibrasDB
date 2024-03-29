import os
import elementpath
import xml.etree.ElementTree as et
import pandas as pd
from tqdm import tqdm
from copy import copy
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from copy import deepcopy
from colorama import init
import argparse


class AllEAFParser2CSV:
    def __init__(self, base_db_path):
        """
        Parameters
        ----------
        base_db_path : String
            Diretorio onde esta os outros diretorios crawladados que contem os
            videos e legendas respectivos de cada um.
        """
        # nome de todas as pastas presentes na base de dados.
        self.estates_path_in_db = [
            os.path.join(base_db_path, x) for x in os.listdir(base_db_path)
        ]
        self.estates_path_in_db = list(
            filter(lambda x: os.path.isdir(x), self.estates_path_in_db)
        )
        self.bad_videos = []
        self.bad_subs = []
        self.amount_workers = multiprocessing.cpu_count()
        self.thread_executor = ThreadPoolExecutor(
            max_workers=self.amount_workers
        )

    def amount_items(self):
        """
        Quantiade total de itens a serem processados.


        Returns
        -------
        total_items: int
            Total de itens a serem processados.
        """
        total_items = 0

        for estates_path in self.estates_path_in_db:
            for proj_path in os.listdir(estates_path):
                proj_path = os.path.join(estates_path, proj_path)

                proj_dirs = os.listdir(proj_path)

                total_items += len(proj_dirs)

        return total_items

    def __process_eaf_async(self, fn, data_list):
        """

        Parameters
        ----------


        Returns
        -------


        """

        if len(data_list) >= self.amount_workers:
            raise RuntimeError(
                "data_list must be less or equals than"
                "{self.amount_workers}, else "
                "split your data_list"
            )
        futures = [self.thread_executor.submit(fn, x) for x in data_list]
        all_done = [False] * len(futures)
        while not all(all_done):
            all_done = [f.done() for f in futures]

        res = [f.result() for f in futures]
        return res

    def process(
        self,
        pbar=None,
        pbar_dup=None,
        path_to_save_sign_df="./",
        sign_df_name="all_videos5.csv",
        check_n_remove_dups=True,
    ):
        """
        Funcao principal que processa todas as legendas.

        Parameters
        ----------
        pbar: tqdm or None
            Barra de progresso para o processamento dos EAFs. Caso None o
            a função cria uma própria pbar.

        pbar_dup:
            Barra de progresso para o processamento dos sinais em duplicatas.
            Caso None o a função cria uma própria pbar.

        path_to_save_sign_df: str
            Localidade onde deve ser salvo o dataframe/csv que organiza a base de dados.

        """
        gen_subs = self.__gen_xmls_base_subs()
        amount_folders = self.amount_items()
        df_cols = ["sign", "beg", "end", "folder_name", "talker_id", "hand"]
        libras_df = pd.DataFrame(columns=df_cols)

        pbar = (
            tqdm(total=amount_folders, desc="Processing EAFs") if pbar is None else pbar
        )

        amount_dups = 0
        for subs_xml, videos, estate_name, proj_name, folder_path in gen_subs:

            if len(subs_xml) == 0 or len(videos) == 0:
                pbar.update(1)
                pbar.refresh()
                continue

            time_stamps = self.__process_eaf_async(
                self.__get_parsed_timestamps, subs_xml
            )
            # time_stamps = [self.__get_parsed_timestamps(x) for x in subs_xml]

            subs = self.__process_eaf_async(self.__get_subs, subs_xml)
            # subs = [self.__get_subs(x) for x in subs_xml]

            if len(subs) > 0:
                has_subs_diff = False
                l_s = subs[0]
                for s in subs[1:]:
                    if not l_s.equals(s):
                        print("found subs with differences in ", videos[0])
                        has_subs_diff = True

                if has_subs_diff:
                    #                    for it in range(len(subs)):
                    #                        dif_df = pd.DataFrame(columns=df_cols)
                    #                        name_df = f'v-({amount_dups})diff_df-it-{it}.csv'
                    #                        name_df = os.path.join(path_to_save_sign_df, name_df)
                    #                        self.__update_sign_df(dif_df, subs[it], time_stamps[it], videos[0],
                    #                                              estate_name, proj_name, folder_path).to_csv(name_df)
                    #                        amount_dups += 1

                    pbar.update(1)
                    pbar.refresh()
                    continue

            # iterrows retorna uma tupla com (idx: int, row: pd.Series)
            libras_df = self.__update_sign_df(
                libras_df,
                subs[0],
                time_stamps[0],
                videos[0],
                estate_name,
                proj_name,
                folder_path,
            )
            pbar.update(1)
            pbar.refresh()

        # Nos arquivos EAF do Corpus de Libras cada falante quando executa um
        # sinal com ambas as mãos é colocado em duplicidade no EAF.
        # Nessa etapa abaixo removemos as duplicidades pois não é interessante
        # saber ja que vamos extrair o esqueleto posteriormente.
        row_2_drop = []
        libras_corpus_index_string = (
            libras_df.folder_name.iloc[0].replace("\\", "/").split("/")
        )
        libras_corpus_index_string = libras_corpus_index_string.index("LibrasCorpus")

        libras_df.folder_name = libras_df.folder_name.map(
            lambda x: os.path.join(
                *(x.replace("\\", "/").split("/")[libras_corpus_index_string:])
            )
        )

        libras_df.folder = libras_df.folder.map(
            lambda x: os.path.join(
                *(x.replace("\\", "/").split("/")[libras_corpus_index_string:])
            )
        )

        libras_df.estate = libras_df.estate.map(
            lambda x: os.path.join(
                *(x.replace("\\", "/").split("/")[libras_corpus_index_string + 1 :])
            )
        )

        if check_n_remove_dups:
            path_2_dup_all_videos = os.path.join(
                path_to_save_sign_df, f"../dupl-{sign_df_name}"
            )
            libras_df.to_csv(path_2_dup_all_videos, index=False)

            if pbar_dup is not None:
                pbar_dup.reset(total=libras_df.shape[0])
                pbar_dup.set_description("dups")
            for it, row in enumerate(libras_df.iterrows()):
                row = row[1]
                res = libras_df.loc[
                    (libras_df["beg"] == row["beg"])
                    & (libras_df["end"] == row["end"])
                    & (libras_df["talker_id"] == row["talker_id"])
                    & (libras_df["folder_name"] == row["folder_name"])
                    & (libras_df["sign"] == row["sign"])
                ]
                if res.shape[0] > 1:
                    libras_df.loc[it, "hand"] = 2
                    row_2_drop.append(res.index)
                else:
                    libras_df.loc[it, "hand"] = 1

                if pbar_dup is not None:
                    pbar_dup.update(1)

            single_list_drop = list(map(lambda x: x[1], row_2_drop))
            single_list_drop = list(set(single_list_drop))
            libras_df = libras_df.drop(single_list_drop)

        path_2_all_videos = os.path.join(path_to_save_sign_df, f"../{sign_df_name}")
        libras_df.to_csv(path_2_all_videos, index=False)

    def remove_db_df_path_specific(self, df):
        """

        Parameters
        ----------
        df

        Returns
        -------

        """
        folder_count_to_db = len(self.estates_path_in_db[0].split("/")) - 1
        db_df = pd.read_csv(df) if isinstance(df, str) else df
        db_df["folder_name"] = db_df["folder_name"].applymap(
            lambda x: x.split("/")[folder_count_to_db:]
        )
        db_df["folder"] = db_df["folder"].applymap(
            lambda x: x.split("/")[folder_count_to_db:]
        )
        db_df["estate"] = db_df["estate"].applymap(
            lambda x: x.split("/")[folder_count_to_db + 1 :]
        )
        return db_df

    @staticmethod
    def __update_sign_df(
        df: pd.DataFrame,
        subs: pd.DataFrame,
        time_stamps: dict,
        video: str,
        estate_name: str,
        proj_name: str,
        folder_path: str,
    ):
        """
        Atualiza o dataframe com as seguintes colunas:

        sinais: str nome do sinal; beg: int

        (inicio em milisegundo do sinal no video de acordo com a legenda);

        end: int (final em milisegundo do sinal no video de acordo com a
                  legenda);

        folder_name: str nome da pasta / video;

        talker_id: int ID da de quem sinaliza de acordo com a legenda;

        hand: int quantas mãos são utilizadas na execução do sinal.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame a ser atualizado. Que deve/irá conter os sinais.

        subs:
            Legendas/Nome dos sinais.

        time_stamps: dict
            dicionario de timestamps. No formato { 'tsX': int_value }, com 'tsX'
            indicando a chave no tempo e int_value o valor de tempo em
            milisegundos.

        video: str
            Nome do video a quem as legendas pertencem.

        Returns
        -------
        df: pd.DataFrame
            DataFrame atualizado com as novas entradas.

        """
        signs = [x[1]["sub"] for x in subs.iterrows()]
        begs = [x[1]["beg"] for x in subs.iterrows()]
        ends = [x[1]["end"] for x in subs.iterrows()]

        begs = list(map(lambda x: time_stamps[x], begs))
        ends = list(map(lambda x: time_stamps[x], ends))

        talker_id = [x[1]["talker_id"] for x in subs.iterrows()]
        hand = [x[1]["hand"] for x in subs.iterrows()]
        video = [video] * len(signs)
        proj = [proj_name] * len(signs)
        folder = [folder_path] * len(signs)
        estate = [estate_name] * len(signs)

        data_dict = dict(
            sign=signs,
            beg=begs,
            end=ends,
            folder_name=video,
            talker_id=talker_id,
            hand=hand,
            project=proj,
            folder=folder,
            estate=estate,
        )

        return df.append(pd.DataFrame(data=data_dict), ignore_index=True)

    @staticmethod
    def __get_subs(tree):
        """
        Dentro do XML da legenda encontra cada palavra referente ao sinal
        executado, ignorando comentarios em formato de legenda e traducoes em
        formato de legenda.

        Parameters
        ----------
        tree : xml.etree.ElementTree
        Arvore do XML que contem a legenda.

        Returns
        -------
        retorna uma lista de dicionarios com as chaves 'beg' indicando
        inicio, 'end' indicando fim em timestamps, e a chave 'sub' contendo
        a legenda dentro desse intervalo de tempo.
        """

        tier_xpath = "//TIER[contains(normalize-space(@TIER_ID),'SinaisD')]" "/@TIER_ID"
        all_tiers_with_sing = elementpath.select(tree, tier_xpath)

        signs = pd.DataFrame(columns=["beg", "end", "sub", "talker_id", "hand"])

        base_xpath_tier = (
            "//TIER[contains(normalize-space(@TIER_ID), '{}')]" "/ANNOTATION"
        )
        for tier_name in all_tiers_with_sing:
            tier_xpath = copy(base_xpath_tier)
            tier_xpath = tier_xpath.format(tier_name)

            beg_slots_xpath = \
                tier_xpath + "/ALIGNABLE_ANNOTATION" "/@TIME_SLOT_REF1"
            time_slots_beg = elementpath.select(tree, beg_slots_xpath)

            end_slots_xpath = \
                tier_xpath + "/ALIGNABLE_ANNOTATION" "/@TIME_SLOT_REF2"
            time_slots_end = elementpath.select(tree, end_slots_xpath)

            subs_xpath = \
                tier_xpath + "/ALIGNABLE_ANNOTATION" "/ANNOTATION_VALUE"
            subs = elementpath.select(tree, subs_xpath)
            subs = [x.text for x in subs]

            talkers_id = [int(tier_name[0])] * len(subs)
            hands = [tier_name[-1]] * len(subs)

            signs = signs.append(
                pd.DataFrame(
                    data=dict(
                        beg=time_slots_beg,
                        end=time_slots_end,
                        sub=subs,
                        talker_id=talkers_id,
                        hand=hands,
                    )
                ),
                ignore_index=True,
            )
        return signs

    @staticmethod
    def __get_parsed_timestamps(tree):
        """
        Lê os timestams presentes num XML de legenda passado como argumento

        Parameters
        ----------
        tree : xml.etree.ElementTree
            arvore contendo o XML/EAF da legenda no video.

        Returns
        -------

        Um dicionario contendo os time stamps, o dicionario esta organizado de
        forma a ter um ts<numero> para chave e em alguma unidade de tempo
        o valor.
        """

        time_unit = elementpath.select(tree, "//HEADER/@TIME_UNITS")
        if time_unit[0] != "milliseconds":
            print(time_unit)

        time_stamps = elementpath.select(tree, "//@TIME_VALUE")
        time_stamps = list(map(int, time_stamps))

        time_stamps_key = elementpath.select(tree, "//@TIME_SLOT_ID")

        last_part = 0
        for tsk in time_stamps_key:
            int_part = int(tsk[2:])
            if int_part - last_part != 1:
                print("at parser", tsk)
            last_part = int_part

        # print('at parser len', len(time_stamps_key), len(time_stamps))

        return dict(zip(time_stamps_key, time_stamps))

    def __gen_xmls_base_subs(self):
        """
        Funcao para criar os xpaths e iterar com generator sobre toda a base.

        Yields
        -------

        Generator com cada video e legenda dentro da base.
        """

        for estates_path in self.estates_path_in_db:
            for proj_path in os.listdir(estates_path):
                proj_name = deepcopy(proj_path)
                proj_path = os.path.join(estates_path, proj_path)

                proj_dirs = os.listdir(proj_path)
                proj_dirs = list(map(lambda x: os.path.join(proj_path, x), proj_dirs))

                for item_path in proj_dirs:
                    item_name = deepcopy(item_path)
                    items_dir = os.listdir(item_path)
                    items_dir = list(
                        map(lambda x: os.path.join(item_path, x), items_dir)
                    )

                    subs = list(filter(lambda x: ".EAF" in x, items_dir))
                    subs_parsed = []
                    failed_subs_count = 0
                    for s in subs:
                        try:
                            s = et.parse(s)
                            subs_parsed.append(s)
                        except et.ParseError as e:
                            failed_subs_count += 1
                            self.bad_subs.append((s, item_path))

                    if failed_subs_count == len(subs):
                        yield [], [], "", "", " "
                        continue

                    videos = list(filter(lambda x: "1.mp4" in x, items_dir))
                    has_new_video = any(list(map(lambda x: "new" in x, videos)))

                    if len(videos) == 0:
                        yield [], [], "", "", ""
                        continue

                    good_videos = []
                    for it, v in enumerate(videos):
                        try:
                            vid = cv.VideoCapture(v)
                            ret, frame = vid.read()
                            if not vid.isOpened() or (not ret):
                                self.bad_videos.append((videos, item_path))
                                vid.release()
                                print(f"found a bad video {v}")
                                continue
                        except cv.error:
                            self.bad_videos.append((videos, item_path))
                            vid.release()
                            continue
                        else:
                            good_videos.append(v)
                            vid.release()

                    good_videos = list(filter(lambda x: "1.mp4" in x, good_videos))
                    if len(good_videos) == 0:
                        yield [], [], "", "", ""
                        continue

                    new_video_path = (
                        list(filter(lambda x: "new" in x, videos))[0]
                        if has_new_video
                        else None
                    )
                    if new_video_path is not None:
                        good_videos = [new_video_path] + good_videos

                    yield subs_parsed, good_videos, estates_path, proj_name, item_name


if __name__ == "__main__":
    init()

    parser = argparse.ArgumentParser(description="All eaf parser")
    parser.add_argument(
        "--db_path",
        type=str,
        help="a path to Libras Corpus",
        default="D:/libras corpus b2/LibrasCorpus",
    )

    parser.add_argument(
        "--sign_df", type=str, help="sign df name", default="all_videos.csv"
    )
    args = parser.parse_args()

    db_cut_videos = AllEAFParser2CSV(args.db_path)
    db_cut_videos.process(
        path_to_save_sign_df="./",
        check_n_remove_dups=False,
        sign_df_name=args.sign_df
    )
