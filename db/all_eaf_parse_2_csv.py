import os
import elementpath
import xml.etree.ElementTree as et
import pandas as pd
from tqdm import tqdm
from copy import copy
import cv2 as cv


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
        self.estates_path_in_db = [os.path.join(base_db_path, x)
                                   for x in os.listdir(base_db_path)]
        self.estates_path_in_db = list(filter(lambda x: os.path.isdir(x),
                                              self.estates_path_in_db))
        self.bad_videos = []
        self.bad_subs = []

    def hiho(self):
        print('hiho')

    def amount_items(self):
        total_items = 0

        for estates_path in self.estates_path_in_db:
            for proj_path in os.listdir(estates_path):
                proj_path = os.path.join(estates_path, proj_path)

                proj_dirs = os.listdir(proj_path)

                total_items += len(proj_dirs)

        return total_items

    def process(self, pbar=None):
        """
        Funcao principal que processa todas as legendas.

        Parameters
        ----------

        """
        gen_subs = self.__gen_xmls_base_subs()
        amount_folders = self.amount_items()
        df_cols = ['sign', 'beg', 'end', 'folder_name', 'talker_id', 'hand']
        libras_df = pd.DataFrame(columns=df_cols)

        pbar = tqdm(total=amount_folders, desc='Processing EAFs') \
            if pbar is None else pbar

        for subs_xml, videos in gen_subs:

            if len(subs_xml) == 0 or len(videos) == 0:
                continue

            time_stamps = [self.__get_parsed_timestamps(x) for x in subs_xml]
            subs = [self.__get_subs(x) for x in subs_xml]

            if len(subs) > 0:
                has_subs_diff = False
                l_s = subs[0]
                for s in subs[1:]:
                    if not l_s.equals(s):
                        print('found subs with differences in ', videos[0])
                        has_subs_diff = True

                if has_subs_diff:
                    for it in range(len(subs)):
                        dif_df = pd.DataFrame(columns=df_cols)
                        name_df = f'{videos[0]}-diff_df-it-{it}.csv'
                        self.__update_sign_df(dif_df, subs[it],
                                              time_stamps[it],
                                              videos[0]).to_csv(name_df)
                    continue

            # iterrows retorna uma tupla com (idx: int, row: pd.Series)
            libras_df = self.__update_sign_df(libras_df, subs[0], time_stamps[0],
                                              videos[0])
            pbar.update(1)
            pbar.refresh()

        # Nos arquivos EAF do Corpus de Libras cada falante quando executa um
        # sinal com ambas as mãos é colocado em duplicidade no EAF.
        # Nessa etapa abaixo removemos as duplicidades pois não é interessante
        # saber ja que vamos extrair o esqueleto posteriormente.
        row_2_drop = []
        libras_df.to_csv('dupl-all_videos.csv')
        pbar.reset(total=libras_df.shape[0])
        pbar.set_description('dups')
        for it, row in enumerate(libras_df.iterrows()):
            row = row[1]
            res = libras_df.loc[(libras_df['beg'] == row['beg']) &
                                (libras_df['end'] == row['end']) &
                                (libras_df['talker_id'] == row['talker_id']) &
                                (libras_df['folder_name'] == row['folder_name']) &
                                (libras_df['sign'] == row['sign'])]
            if res.shape[0] > 1:
                libras_df.loc[it, 'hand'] = 2
                row_2_drop.append(res.index)
            else:
                libras_df.loc[it, 'hand'] = 1
        single_list_drop = list(map(lambda x: x[1], row_2_drop))
        single_list_drop = list(set(single_list_drop))
        libras_df = libras_df.drop(single_list_drop)
        pbar.update(1)
        pbar.refresh()

        libras_df.to_csv('all_videos.csv')

    @staticmethod
    def __update_sign_df(df: pd.DataFrame, subs: pd.DataFrame,
                         time_stamps: dict, video: str):
        """
        Atualiza o dat;,aframe com as seguintes colunas:
        sinais: str nome do sinal;
        beg: int (inicio em milisegundo do sinal no video de acordo com a
        legenda);
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
        signs = [x[1]['sub'] for x in subs.iterrows()]
        begs = [x[1]['beg'] for x in subs.iterrows()]
        ends = [x[1]['end'] for x in subs.iterrows()]

        begs = list(map(lambda x: time_stamps[x], begs))
        ends = list(map(lambda x: time_stamps[x], ends))

        talker_id = [x[1]['talker_id'] for x in subs.iterrows()]
        hand = [x[1]['hand'] for x in subs.iterrows()]
        video = [video] * len(signs)

        data_dict = dict(sign=signs, beg=begs, end=ends, folder_name=video,
                         talker_id=talker_id, hand=hand)

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

        tier_xpath = "//TIER[contains(normalize-space(@TIER_ID),'Sinais')]" \
                     "/@TIER_ID"
        all_tiers_with_sing = elementpath.select(tree, tier_xpath)

        signs = pd.DataFrame(columns=['beg', 'end', 'sub', 'talker_id',
                                      'hand'])

        base_xpath_tier = "//TIER[contains(normalize-space(@TIER_ID), '{}')]" \
                          "/ANNOTATION"
        for tier_name in all_tiers_with_sing:
            tier_xpath = copy(base_xpath_tier)
            tier_xpath = tier_xpath.format(tier_name)

            beg_slots_xpath = tier_xpath + '/ALIGNABLE_ANNOTATION' \
                                           '/@TIME_SLOT_REF1'
            time_slots_beg = elementpath.select(tree, beg_slots_xpath)

            end_slots_xpath = tier_xpath + '/ALIGNABLE_ANNOTATION' \
                                           '/@TIME_SLOT_REF2'
            time_slots_end = elementpath.select(tree, end_slots_xpath)

            subs_xpath = tier_xpath + '/ALIGNABLE_ANNOTATION' \
                                      '/ANNOTATION_VALUE'
            subs = elementpath.select(tree, subs_xpath)
            subs = [x.text for x in subs]

            talkers_id = [int(tier_name[0])] * len(subs)
            hands = [tier_name[-1]] * len(subs)

            signs = signs.append(pd.DataFrame(data=dict(beg=time_slots_beg,
                                                        end=time_slots_end,
                                                        sub=subs,
                                                        talker_id=talkers_id,
                                                        hand=hands)),
                                 ignore_index=True)
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

        time_unit = elementpath.select(tree, '//HEADER/@TIME_UNITS')
        if time_unit[0] != 'milliseconds':
            print(time_unit)

        time_stamps = elementpath.select(tree, '//@TIME_VALUE')
        time_stamps = list(map(int, time_stamps))

        time_stamps_key = elementpath.select(tree,
                                             '//@TIME_SLOT_ID')

        last_part = 0
        for tsk in time_stamps_key:
            int_part = int(tsk[2:])
            if int_part - last_part != 1:
                print('at parser', tsk)
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
                proj_path = os.path.join(estates_path, proj_path)

                proj_dirs = os.listdir(proj_path)
                proj_dirs = list(map(lambda x: os.path.join(proj_path, x),
                                     proj_dirs))

                for item_path in proj_dirs:
                    items_dir = os.listdir(item_path)
                    items_dir = \
                        list(map(lambda x: os.path.join(item_path, x),
                                 items_dir))

                    subs = list(filter(lambda x: '.EAF' in x, items_dir))
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
                        yield [], []
                        continue

                    videos = list(filter(lambda x: '.mp4' in x, items_dir))
                    if len(videos) == 0:
                        yield [], []
                        continue

                    failed_video = [False] * len(videos)
                    good_videos = []
                    for it, v in enumerate(videos):
                        try:
                            vid = cv.VideoCapture(v)
                            ret, frame = vid.read()
                            if not vid.isOpened() or (not ret):
                                self.bad_videos.append((videos, item_path))
                                vid.release()
                                continue
                        except cv.error:
                            self.bad_videos.append((videos, item_path))
                            continue
                        else:
                            good_videos.append(v)
                            vid.release()
                    if failed_video[0]:
                        yield [], []
                        continue

                    # if len(good_videos) == 0 or len(subs_parsed) == 0:
                    #     yield [], []
                    #     continue

                    yield subs_parsed, good_videos


if __name__ == '__main__':
    db_cut_videos = AllEAFParser2CSV('../LibrasCorpusScrapy/db')
    db_cut_videos.process()
    print(db_cut_videos.bad_subs, file=open('bad_subs.txt', mode='w'))
    print(db_cut_videos.bad_videos, file=open('bad_videos.txt', mode='w'))
