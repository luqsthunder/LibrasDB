import numpy as np
import zipfile
import pandas as pd
from tqdm import tqdm
import os

all_videos = pd.read_csv("all_videos.csv")
folders = all_videos.folder.unique().tolist()

folders_batch = []

db_path = "d:/gdrive"


def zip_batch_folders(folders, zip_name):
    zipf = zipfile.ZipFile(f"d:/{zip_name}", "w", zipfile.ZIP_DEFLATED)
    for f in tqdm(folders):
        f_path = os.path.join(db_path, f)
        zipf.write(f_path, arcname=f)
        for file_in_folder in os.listdir(f_path):
            zipf.write(
                os.path.join(f_path, file_in_folder), os.path.join(f, file_in_folder)
            )


zip_batch_folders(folders[0 : len(folders) // 4], "b1.zip")
