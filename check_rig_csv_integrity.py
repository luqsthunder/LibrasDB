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
        folder_content = os.listdir(db_folder)
        files_csv = list(filter(os.path.isfile, files_csv))
        more_folders = list(filter(os.path.isdir, files_csv))

        if len(more_folders) > 0:
            csvs_in_more = []
            more_folders = list(map(lambda x: os.path.join(db_folder, x),
                                    more_folders))
            for folder_in_more in more_folders:
                files_in_more = list(map(
                    lambda x: os.path.join(x, folder_in_more),
                    os.listdir(folder_in_more)
                ))
                csvs_in_more.extend(files_in_more)
            files_csv.extend(csvs_in_more)

        for db_folder_csv in files_csv:
            db_folder_csv = os.path.join(db_folder, db_folder_csv)
            sign_df = pd.read_csv(db_folder_csv)
            if sign_df.shape[0] == 0:
                print(db_folder_csv)
except BaseException as e:
    print(e)
    sys.exit(-1)

