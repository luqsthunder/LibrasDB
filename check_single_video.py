import sys
import os
import cv2 as cv
from tqdm.auto import tqdm
import argparse


def check_single_video(name):
    video = cv.VideoCapture(name)
    end = video.get(cv.CAP_PROP_FRAME_COUNT)

    for _ in tqdm(range(int(end) - 1), position=2):
        frame = video.get(cv.CAP_PROP_POS_FRAMES)
        ret, frame = video.read()
        if not ret:
            print(frame)
            video.release()
            return False
    video.release()
    return True


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Process files in dir or single file \n can remove bad files if you'
    # parser.add_argument('file_or_dir', metavar='file_or_dir', type=str)
    # parser.add_argument('--rm_bad', metavar='rm_bad')
    # args = parser.parse_args()

    rm_bad = True
    if os.path.isdir(sys.argv[1]):
        bad_list = []
        for file in tqdm(os.listdir(sys.argv[1]), position=1):
            if ".mp4" not in file:
                continue

            file_path = os.path.join(sys.argv[1], file)
            if not check_single_video(file_path):
                bad_list.append(file_path)

        if rm_bad:
            for b_file in bad_list:
                os.remove(b_file)
    else:
        check_single_video(sys.argv[1])
# 487
