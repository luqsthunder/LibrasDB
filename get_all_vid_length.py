import sys
import seaborn as sns
import os
from tqdm import tqdm
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import subprocess

def get_length(input_video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return float(result.stdout)

all_videos = pd.read_csv('all_videos.csv')
duration = list(map(lambda x: x[1]['end'] - x[1]['beg'], all_videos.iterrows()))
mean_duration = sum(duration) // len(duration)

db_path = '/mnt/d/gdrive/LibrasCorpus/Santa Catarina/Inventario Libras'
itens = os.listdir(db_path)
itens = list(map(lambda x: os.path.join(db_path, x), itens))
video_durations = []
for item in tqdm(itens):
    videos = list(filter(lambda x: '.mp4' in x, os.listdir(item)))
    videos = list(map(lambda x: os.path.join(item, x), videos))
    for v in videos:
        try:
            video_durations.append(get_length(v))
        except:
            continue

mean_length = sum(video_durations) // len(video_durations)
print(sum(video_durations))
print(mean_length)
print(sorted(video_durations, reverse=True)[0])
