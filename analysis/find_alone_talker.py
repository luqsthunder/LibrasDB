import pandas as pd
from tqdm import tqdm

all_videos = pd.read_csv('all_videos.csv')

print(all_videos.keys())
folders = all_videos.folder_name.unique().tolist()

def find_where_signaling_is_quiet(all_video_df, folder_name, talker_id):
    """
    Acha onde um sinalizador de uma legenda na pasta do projeto não fala nada.

    parameters
    ----------
    all_video_df: pd.DataFrame
        base de dados.

    folder_name: str
        nome da pasta na base de dados.

    talker_id: int
        id do sinalizador na legenda

    returns: List
        holes, lista com os locais onde o sinalizador não fala.
    """
    talker_1_signs = all_videos_df[all_videos_df.folder_name == folder_name]
    talker_1_signs = talker_1_signs[talker_1_signs.talker_id == talker_id]
    talker_1_signs = talker_1_signs[talker_1_signs.hand == 1]
    talker_1_times =[(x[1].beg, x[1].end)
                     for x in talker_1_signs.iterrows() if x[1].talker_id == 1]
    talker_1_times = sorted(talker_1_times, key=lambda x: x[0])

    last_time_sec = None
    times_talking = [talker_1_times[0][0]]
    last_end = talker_1_times[0][1]
    holes = []

    for time_talk in tqdm(talker_1_times[1:]):
        if last_end >= time_talk[0]:
            last_end = time_talk[1] if last_end < time_talk[1] else last_end
        else:
            times_talking.extend([last_end, time_talk[0]])
            last_end = time_talk[1]

    for time in range(0, len(times_talking), 3):
                try:
                    hole = times_talking[time + 2] - times_talking[time + 1]
                    holes.append(dict(beg=times_talking[time],
                          end=times_talking[time + 2],
                          hole=hole))
        except IndexError:
            continue

    holes = sorted(list(filter(lambda x: x['hole'] > 1000, holes)),
                   key=lambda x: x['hole'])
    return holes

signer_1_holes = find_where_signaling_is_quiet(all_videos, folders[0], 1)
signer_2_holes = find_where_signaling_is_quiet(all_videos, folders[0], 2)


