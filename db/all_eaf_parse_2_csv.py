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
        res = os.listdir('.')
        self.estates_path_in_db = [os.path.join(base_db_path, x)
                                   for x in os.listdir(base_db_path)]
        self.estates_path_in_db = list(filter(lambda x: os.path.isdir(x),
                                              self.estates_path_in_db))

    def process(self):
        """
        Funcao principal que processa todas as legendas.

        Parameters
        ----------

        """
        gen_subs = self.__gen_xmls_base_subs()
        amount_folders = next(gen_subs)
        libras_df = pd.DataFrame(columns=['sign', 'beg', 'end', 'folder_name',
                                          'talker_id', 'hand'])

        for subs_xml, videos in tqdm(gen_subs, total=amount_folders):
            # TODO: destrinxar as legendas e comparar se elas são iguais

            if len(subs_xml) == 0 or len(videos) == 0:
                continue

            time_stamps = [self.__get_parsed_timestamps(x) for x in subs_xml]
            subs = [self.__get_subs(x) for x in subs_xml]

            if len(subs) > 0:
                l_s = subs[0]
                for s in subs[1:]:
                    if not l_s.equals(s):
                        print('found subs with differences in ', videos[0])

            signs = [x[1]['sub'] for x in subs[0].iterrows()]
            begs = [x[1]['beg'] for x in subs[0].iterrows()]
            ends = [x[1]['end'] for x in subs[0].iterrows()]

            begs = list(map(lambda x: time_stamps[0][x], begs))
            ends = list(map(lambda x: time_stamps[0][x], ends))

            talker_id = [x[1]['talker_id'] for x in subs[0].iterrows()]
            hand = [x[1]['hand'] for x in subs[0].iterrows()]
            video = [videos[0]] * len(signs)

            data_dict = dict(sign=signs, beg=begs, end=ends, folder_name=video,
                             talker_id=talker_id, hand=hand)
            libras_df = libras_df.append(pd.DataFrame(data=data_dict),
                                         ignore_index=True)
        libras_df.to_csv('all_videos.csv')

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
        indexes_2_drop = []
        for row in signs.iterrows():
            # vem como uma tupla (idx, Series)
            row = row[1]

            ret = signs.loc[(signs['beg'] == row['beg']) &
                            (signs['end'] == row['end']) &
                            (signs['sub'] == row['sub']) &
                            (signs['talker_id'] == row['talker_id'])]
            print(ret.shape)
            if ret.shape[0] > 1:
                signs.loc[ret['hand'].index, 'hand'] = 2
                indexes_2_drop.append(ret.index[1])

        return signs


    @staticmethod
    def __get_parsed_timestamps(tree):
        """
        Lê os timestams presentes num XML de legenda passado como argumento

        Parameters
        ----------

        tree : xml.etree.ElementTree

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

        Returns
        -------

        Generator com cada video e legenda dentro da base.
        """

        total_items = 0

        for estates_path in self.estates_path_in_db:
            for proj_path in os.listdir(estates_path):
                proj_path = os.path.join(estates_path, proj_path)

                proj_dirs = os.listdir(proj_path)

                total_items += len(proj_dirs)

        yield total_items

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

                    subs = list(filter(lambda x: '.xml' in x, items_dir))
                    subs_parsed = []
                    failed_subs_count = 0
                    for s in subs:
                        try:
                            s = et.parse(s)
                            subs_parsed.append(s)
                        except et.ParseError as e:
                            failed_subs_count += 1

                    if failed_subs_count == len(subs):
                        yield [], []
                        continue

                    videos = list(filter(lambda x: '.mp4' in x, items_dir))
                    failed_video_count = 0
                    good_videos = []
                    for v in videos:
                        try:
                            vid = cv.VideoCapture(v)
                            ret, frame = vid.read()
                            if not vid.isOpened() or (not ret):
                                failed_video_count += 1
                                vid.release()
                                continue
                        except cv.error:
                            failed_video_count += 1
                        else:
                            good_videos.append(v)
                            vid.release()

                    if len(good_videos) == 0 or len(subs_parsed) == 0:
                        yield [], []

                    yield subs_parsed, good_videos

    def parse_time_slots(tree):
        pass


if __name__ == '__main__':
    db_cut_videos = AllEAFParser2CSV('./db')
    db_cut_videos.process()
