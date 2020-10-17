from libras_classifier.librasdb_loaders import DBLoader2NPY
from tqdm import tqdm
import pandas as pd

class LibrasImageLoader(DBLoader2NPY):
    """
    Aqui é utilizado a funcionalidade ja presente do DBLoader2NPY para carregar as poses e extrair a imagem relacionada
    a cada junta. Os videos ja presentes de cada amostra é carregada por essa mesma classe (LibrasImageLoader) e
    fornece-las como um generator para tf-keras (tensorflow 2 API do keras) e o keras (api antiga com tensorflow 1.14).
    """

    def __init__(self, video_db_path, all_videos_csv_path, **kwargs)
        """
        Os kwargs são os mesmos da DBLoader2NPY, cheque o arquivo libras_classifier/librasdb_loaders.py para maiores
        detalhes.
        """
        self.super().__init__(**kwargs)

        self.video_db_path = video_db_path
        self.all_videos = pd.read_csv(all_videos_csv_path)
        self.sample_2_video = {}
        self.video_files= {}

    def batch_load_sample(self, samples_idx, as_npy=True, clean_nan=True, pbar: tqdm = None):
        X, y = super().batch_load_sample(samples_idx, as_npu=False, clean_nan=False)
        for sample_id in samples_idx:
            pass
        return X, y

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
