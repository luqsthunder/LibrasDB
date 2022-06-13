from pose_extractor.openpose_extractor import OpenposeExtractor
from pose_extractor.pose_centroid_tracker import PoseCentroidTracker
from pose_extractor.df_utils import update_xy_pose_df_single_person
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
from copy import deepcopy
import pose_extractor.all_parts as all_parts


class ExtractSignsFromVideo:
    def __init__(self, db_path):
        pass

    def process(self):
        pass


def process_single_sample(
    extractor,
    curr_video,
    beg,
    end,
    person_needed_id,
    pose_tracker,
    pbar=None,
    num_gpus=1,
):

    num_gpus = int(num_gpus)
    frame_time = 1000 / curr_video.get(cv.CAP_PROP_FPS)
    frame_end_pos = end / frame_time
    succ = curr_video.set(cv.CAP_PROP_POS_FRAMES, float(beg / frame_time))
    if not succ:
        raise RuntimeError(f"Could not set video position")

    df_cols = (
        ["frame"]
        + all_parts.BODY_PARTS_NAMES
        + ["left-" + x for x in all_parts.HAND_PARTS]
        + ["right-" + x for x in all_parts.HAND_PARTS]
    )

    video_df = pd.DataFrame(columns=df_cols)

    beg_frame_pos = beg / frame_time

    # aparentemente as pessoas estão sorteadas da esquerda para direita.
    real_id_if_sorted = person_needed_id - 1

    if pbar is not None:
        pbar.reset(total=int(frame_end_pos - beg_frame_pos) + 1)

    first_x_mid = None

    def update_single_pose_in_df(pose_dt, curr_df, first_x_mid_, curr_msec):
        if first_x_mid_ is None:
            curr_centroids = list(
                map(pose_tracker.make_xy_centroid, pose_dt.poseKeypoints)
            )
            if len(curr_centroids) < 2:
                return None

            x_mid_point = sum(map(lambda x: x[0], curr_centroids))
            first_x_mid_ = x_mid_point / len(curr_centroids)

        left_sorted_persons = pose_tracker.filter_persons_by_x_mid(
            pose_dt, first_x_mid_
        )
        new_df = update_xy_pose_df_single_person(
            pose_dt,
            curr_df,
            curr_msec,
            left_sorted_persons[real_id_if_sorted][0],
            left_sorted_persons[real_id_if_sorted][1],
            pose_tracker.body_parts,
            pose_tracker.hands_parts,
        )
        return new_df, first_x_mid_

    while curr_video.get(cv.CAP_PROP_POS_FRAMES) <= frame_end_pos:
        curr_update_amount = num_gpus
        if num_gpus > 0:
            frames = []
            msecs = []
            for it in range(num_gpus):
                ret, frame = curr_video.read()
                curr_msec_pos = curr_video.get(cv.CAP_PROP_POS_MSEC)

                if curr_msec_pos >= end:
                    break

                if not ret and it == 0:
                    return None

                msecs.append(curr_msec_pos)
                frames.append(frame)
            poses = extractor.extract_multiple_gpus(frames)
            for p, msec in zip(poses, msecs):
                video_df, first_x_mid = update_single_pose_in_df(
                    p, video_df, first_x_mid, msec
                )

            curr_update_amount = len(frames)

        else:
            ret, frame = curr_video.read()
            curr_msec_pos = curr_video.get(cv.CAP_PROP_POS_MSEC)
            if not ret:
                return None
            dt = extractor.extract_poses(frame)
            video_df, first_x_mid = update_single_pose_in_df(
                dt, video_df, first_x_mid, curr_msec_pos
            )

        if pbar is not None:
            pbar.update(curr_update_amount)

    return video_df


if __name__ == "__main__":
    # Path to DB
    db_path = "D:/gdrive/LibrasCorpus/"
    if db_path == "":
        raise RuntimeError("esqueceu de setar o db_path")

    # pose_extractor = OpenposeExtractor('../openpose')
    centroids_df = pd.read_csv("centroids.csv")
    pose_tracker = PoseCentroidTracker(
        "all_videos.csv",
        "db_path",
        openpose_path="../../Libraries/repos/openpose",
        centroids_df_path="centroids.csv",
    )

    all_videos = pd.read_csv("all_videos.csv")[
        ["sign", "beg", "end", "folder_name", "talker_id", "hand"]
    ]
    signs_names = all_videos.sign.unique()
    # count_signs = []
    # for sign in tqdm(signs_names):
    #     amount_sign = all_videos[all_videos.sign == sign].sign.count()
    #
    #     count_signs.append(dict(name=sign,
    #                             count=amount_sign))
    #
    # sorted_signs = sorted(count_signs, reverse=True, key=lambda x: x['count'])
    needed_signs = ["IX(eu)", "E(então)", "IX(você)", "E(positivo)"]

    no_unicode_need_signs = ["IX(eu)", "E(entao)", "IX(voce)", "E(positivo)"]
    no_unicode_need_signs = {
        needed_signs[it]: x for it, x in enumerate(no_unicode_need_signs)
    }
    # [x['name'] for x in sorted_signs[:4]]
    # print(needed_signs)

    if os.path.exists("bad_video.csv"):
        bad_video_df = pd.read_csv("bad_video.csv")
    else:
        bad_video_df = pd.DataFrame()

    # folders_name_map = {x.split('v')[-1]: x
    #                     for x in os.listdir(os.path.join(db_path, 'Inventario Libras'))}

    centroid_folder_names = sorted(list(centroids_df.folder.unique()))
    processed_signs_count = {x: 0 for x in needed_signs}

    ms_window = 0

    folders = sorted(list(all_videos.folder_name.unique()))
    for f_name in tqdm(folders):

        v_part = f_name.split(" v")[-1].split("/")[0].split("/")[0]
        if bad_video_df.shape[0] > 0:
            if f_name in bad_video_df.folder.unique():
                continue

        needed_sings_in_video = all_videos[
            (all_videos["folder_name"] == f_name)
            & (all_videos["sign"].isin(needed_signs))
        ]

        video_path = os.path.join(db_path, f_name).replace("\\", "/")

        only_folder_name = video_path.split("/")[-2]
        folder_name = os.path.join(*(video_path.split("/")[:-1])).replace("\\", "/")
        sign_path = deepcopy(folder_name)
        video = cv.VideoCapture(video_path)

        for it, sign in tqdm(
            enumerate(needed_sings_in_video.iterrows()),
            total=needed_sings_in_video.shape[0],
            desc="signs",
        ):
            sign = sign[1]
            if processed_signs_count[sign.sign] >= 120:
                continue

            sign_name = sign.sign
            sample_name = f"sample-xy-{ms_window}ms-{only_folder_name}-{sign_name}-beg-{sign.beg}-end-{sign.end}.csv"
            sample_path = os.path.join(sign_path, sample_name)
            if os.path.exists(sample_path):
                processed_signs_count[sign.sign] += 1
                continue

            print(f"\n beg sign {sign.sign}")
            df = process_single_sample(
                pose_tracker.pose_extractor,
                video,
                sign.beg - ms_window,
                sign.end + ms_window,
                sign.talker_id,
                pbar=tqdm(),
            )
            if df is not None:
                df.to_csv(sample_path)
                processed_signs_count[sign.sign] += 1

        video.release()
