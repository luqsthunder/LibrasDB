import os
import elementpath
import xml.etree.ElementTree as et
import pandas as pd
from tqdm import tqdm
import cv2 as cv


class DBCutVideos:
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

    def process(self):
        """
        Funcao principal que processa todas as legendas.

        Parameters
        ----------

        """
        gen_subs = self.__gen_xmls_base_subs()
        amount_folders = next(gen_subs)
        libras_df = pd.DataFrame(columns=['sign', 'beg', 'end', 'video_file'])
        for subs_xml, videos in tqdm(gen_subs, total=amount_folders):
            # TODO: destrinxar as legendas e comparar se elas são iguais

            time_stamps = [self.__get_parsed_timestamps(x) for x in subs_xml]
            subs = [self.__get_subs(x) for x in subs_xml]

            signs = []
            begs = []
            ends = []
            video = []

            for s in subs[0]:
                cur_sign = None; cur_beg = None; cur_end = None;
                try:
                    cur_sign = s['sub']
                    cur_beg = time_stamps[0][s['beg']]
                    cur_end = time_stamps[0][s['end']]
                except KeyError as e:
                    print(e)
                    continue

                signs.append(cur_sign); begs.append(cur_beg)
                ends.append(cur_end); video.append(videos[0])

            libras_df = libras_df.append(pd.DataFrame(data=dict(sign=signs,
                                                                beg=begs,
                                                                end=ends,
                                                                video_file=video)))
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

        base_xpath_anotation =\
            "//TIER[contains(normalize-space(@TIER_ID),'Sinais')]/ANNOTATION"
        time_stots_beg =\
            elementpath.select(tree, base_xpath_anotation +
                               "/ALIGNABLE_ANNOTATION/@TIME_SLOT_REF1")
        time_stots_end = \
            elementpath.select(tree, base_xpath_anotation +
                               "/ALIGNABLE_ANNOTATION/@TIME_SLOT_REF2")
        subs = elementpath.select(tree, base_xpath_anotation +
                                  "/ALIGNABLE_ANNOTATION/"
                                  "ANNOTATION_VALUE/text()")

        return list(map(lambda x: {'beg': x[0], 'end': x[1], 'sub': x[2]},
                        zip(time_stots_beg, time_stots_end, subs)))

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
        for estates_path in self.estates_path_in_db:
            for proj_path in os.listdir(estates_path):
                proj_path = os.path.join(estates_path, proj_path)

                proj_dirs = os.listdir(proj_path)
                proj_dirs = list(map(lambda x: os.path.join(proj_path, x),
                                     proj_dirs))

                yield len(proj_dirs)
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
                            print('sub error {}'.format(e))
                            failed_subs_count += 1

                    if failed_subs_count == len(subs):
                        continue

                    videos = list(filter(lambda x: '.mp4' in x, items_dir))
                    failed_video_count = 0
                    good_videos = []
                    for v in videos:
                        try:
                            vid = cv.VideoCapture(v)
                            if not vid.isOpened():
                                print('video error: {}'.format(v))
                                failed_subs_count += 1
                                continue
                        except cv.error:
                            print('video error: {}'.format(v))
                            failed_video_count += 1
                        else:
                            good_videos.append(v)
                            vid.release()

                    if len(good_videos) == 0 or len(subs_parsed) == 0:
                        continue

                    yield subs_parsed, good_videos

    def parse_time_slots(tree):
        pass


if __name__ == '__main__':
    db_cut_videos = DBCutVideos('/gdrive/My Drive/LibrasCorpus')
    db_cut_videos.process()
