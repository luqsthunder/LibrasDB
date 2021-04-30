import os
import sys
import pandas as pd
from tqdm import tqdm

db_path = '../sign_db_front_view/sign_db_rig/'

all_db_folders = os.listdir(db_path)
all_db_folders = list(map(lambda x: os.path.join(db_path, x), all_db_folders))

csvs_with_zeros = []

try:
    for db_folder in tqdm(all_db_folders):
        for db_folder_csv in os.listdir(db_folder):
            db_folder_csv = os.path.join(db_folder, db_folder_csv)
            sign_df = pd.read_csv(db_folder_csv)
            if sign_df.shape[0] == 0:
                print(db_folder_csv)
except BaseException as e:
    print(e)
    sys.exit(-1)

