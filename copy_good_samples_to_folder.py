import os
import shutil

sinal = 'TRABALHAR'
end =  'semelhante'

folder_vid_base = f'./vid-folder/{sinal}/{end}'

video_list = os.listdir(f"./vid-folder/{sinal}/{end}")

#for video in video_list:
#    new_video_name = video.split('---')
#    new_video_name[1] = 'NAO'
#    new_video_name = '---'.join(new_video_name)
#    print(new_video_name)
#    shutil.move(os.path.join(folder_vid_base, video), 
#                os.path.join(folder_vid_base, new_video_name))


video_list = list(filter(lambda x: '.mp4' in x, video_list))
video_list = list(map(lambda x: x[:-4], video_list))

base_db_folder1 = f'./sign_db_front_view/{sinal}/'
base_db_folder2 = f'./cut_folders/{sinal}/'

base_db_2_copy = f'./clean_sign_db_front_view/{sinal}'
if not os.path.exists(base_db_2_copy):
    os.makedirs(base_db_2_copy, exist_ok=True)

samples = list(map(lambda x: os.path.join(base_db_folder1, x), 
                   os.listdir(base_db_folder1))) + \
          list(map(lambda x: os.path.join(base_db_folder2, x), 
                   os.listdir(base_db_folder2)))

for video in video_list:
    for s in samples:
        if video in s:
            print(video, s)
            shutil.copyfile(src=s, dst=os.path.join(base_db_2_copy, video + '.csv'))

