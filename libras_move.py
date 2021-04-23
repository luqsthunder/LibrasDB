import os
from tqdm import tqdm
import shutil

db_path = 'D:/gdrive/LibrasCorpus/Santa Catarina/Inventario Libras/'
db_folders_path = [os.path.join(db_path, x) for x in os.listdir(db_path)]
db_folders_path = sorted(db_folders_path, key=lambda x: int(x.split(' v')[-1]), reverse=True)
for db_folder in tqdm(db_folders_path):
    non_ended_in_cam = sorted(list(filter(lambda x: '1.mp4' in x, os.listdir(db_folder))),
                              key=lambda x: int(x.split('.mp4')[0][-1]))
    non_ended_in_cam = list(filter(lambda x: 'FLN' in x, non_ended_in_cam))

    ended_in_cam = list(filter(lambda x: 'm.mp4' in x, os.listdir(db_folder)))
    new_video = list(filter(lambda x: 'new_video_' in x, os.listdir(db_folder)))

    if len(ended_in_cam) > 0:
        #shutil.copy(os.path.join(db_folder, ended_in_cam[0]), os.path.join(db_folder, non_ended_in_cam[0]))
        #print(os.path.join(db_folder, ended_in_cam[0]), os.path.join(db_folder, non_ended_in_cam[0]))
        #os.remove(os.path.join(db_folder, ended_in_cam[0]))
    elif len(new_video) > 0:
        #shutil.copy(os.path.join(db_folder, new_video[0]), os.path.join(db_folder, non_ended_in_cam[0]))
        #print(os.path.join(db_folder, new_video[0]),' ---- ' ,os.path.join(db_folder, non_ended_in_cam[0]))
        #os.remove(os.path.join(db_folder, new_video[0]))