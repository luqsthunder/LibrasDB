import os
import pandas as pd

db_path = '../sign_db_front_view/sign_db_rig/'

all_db_folders = os.listdir(db_path)
all_db_folders = list(map(lambda x: os.path.join(db_path, x), all_db_folders))

csvs_with_zeros = []

for db_folder in all_db_folders:
    for db_folder_csvs in os.listdir(db_folder):
        db_folder_csvs = list(map(lambda x: os.path.join(db_folder, x),
                                   db_folder_csvs))
        db_folders_csvs = list(map(pd.read_csv, db_folder_csvs))
        if any(list(map(lambda x: x.shape[0] == 0, db_folder_csvs))):
            print(db_folder_csvs)
        #csvs_with_zeros.extend([x for x in db_folders_csvs if x.shape[0] == 0])
