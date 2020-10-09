# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
import shutil
import face_recognition


# %%
db_path = '/media/usuario/Others/gdrive/LibrasCorpus/Santa Catarina/Inventario Libras'
db_folders_path = [os.path.join(db_path, x) for x in os.listdir(db_path)]
db_folders_path = sorted(db_folders_path, key=lambda x: int(x.split(' v')[-1]), reverse=True)


# %%
all_videos_in_folder = [cv.VideoCapture(os.path.join(db_folders_path[2], x))
                        for x in os.listdir(db_folders_path[2]) if '.mp4' in x]
for it in range(len(all_videos_in_folder)):
    all_videos_in_folder[it].set(cv.CAP_PROP_POS_FRAMES, 30)


# %%
def face_rec_create_encodings(vc, amount_faces: int = 20, show=False, ret_amount_faces=False, pbar: tqdm = None):
    encondings = []

    if pbar is not None:
        pbar.reset(total=amount_faces)

    count = 0
    while count < amount_faces:
        ret, frame = vc.read()

        if not ret:
            return [], 0


        face_location = face_recognition.face_locations(frame[:, :, ::-1], number_of_times_to_upsample=1, model='cnn')
        if len(face_location) > 1:
            continue

        curr_face_encoding = face_recognition.face_encodings(frame[:, :, ::-1], face_location)
        encondings.append(curr_face_encoding)

        if show:
            face = face_location[0]
            x1 = face[3]
            y1 = face[0]

            x2 = face[1]
            y2 = face[2]
            frame = cv.rectangle(frame, (x1, y1), (x2, y2), (50, 205, 50), 2)
            cv.imshow('win', frame)
            key = cv.waitKey(-1)
            if key == 27:  # exit on ESC
                break

        if pbar is not None:
            pbar.update(1)
            pbar.refresh()

        count += 1
    cv.destroyAllWindows()
    return encondings if not ret_amount_faces else encondings, len(face_location)


# %% test
for it in range(len(all_videos_in_folder)):
    all_videos_in_folder[it].set(cv.CAP_PROP_POS_FRAMES, 30 * 60 * 1)

#person_1 = create_face_images(0, dual=True, show=True)
try:
    p1_encodings = face_rec_create_encodings(vc=all_videos_in_folder[1], pbar=tqdm())
    p2_encodings = face_rec_create_encodings(vc=all_videos_in_folder[2], pbar=tqdm())
except RuntimeError:
    cv.destroyAllWindows()


# %%
def gen_train_set_from_encodings(p1_encode_vecs, p2_encode_vecs):
    embeddings = []
    y_train = []
    lbl = 0
    for person_encodings in [p1_encode_vecs, p2_encode_vecs]:
        if person_encodings is None:
            continue

        for encode_vec in person_encodings:
            if len(encode_vec) > 0:
                embeddings.append(encode_vec[0])
                y_train.append(lbl)
        lbl = lbl + 1

    x_train = np.array(embeddings)
    x_train = x_train.reshape(x_train.shape[0], 128)

    return x_train, y_train
# %%
def make_embedings_from_reference_video(reference_vc, show_frame=False):
    reference_embedings = []
    while len(reference_embedings) < 2:
        ret, reference_frame = reference_vc.read()
        if not ret:
            return [], [], None

        if show_frame:
            cv.imshow('w', reference_frame)
            key = cv.waitKey(-1)
            if key == 27:
                break
        faces_in_reference = face_recognition.face_locations(reference_frame[:, :, ::-1],
                                                             number_of_times_to_upsample=1,
                                                             model='cnn')
        reference_embedings = face_recognition.face_encodings(reference_frame[:, :, ::-1], faces_in_reference)

    if show_frame:
        cv.destroyAllWindows()

    return reference_embedings

def make_bounding_box_from_encodings(vc, encodings_list, amount_frames=20, reference_embedings=None):
    x_train, y_train = gen_train_set_from_encodings(encodings_list[0], encodings_list[1])
    clf_knn = KNN(n_neighbors=5)
    clf_knn.fit(x_train, y_train)
    font = cv.FONT_HERSHEY_SIMPLEX

    for _ in range(amount_frames):
        ret, frame = vc.read()

        faces_in_frame = face_recognition.face_locations(frame[:, :, ::-1], number_of_times_to_upsample=1, model='cnn')

        identities = []
        left_id = [0, 0]
        right_id = [0, 0]
        for face in faces_in_frame:
            embedding = face_recognition.face_encodings(frame[:, :, ::-1], [face])
            identity = clf_knn.predict(embedding)
            identity_score = clf_knn.predict_proba(embedding)[0]

            identities.append([identity_score])

            x1 = face[3]
            y1 = face[0]

            x2 = face[1]
            y2 = face[2]

            curr_max = None
            if x1 <= frame.shape[0] // 2:
                left_id[identity[0]] += 1
                curr_max = np.argmax(left_id)
            elif x1 > frame.shape[0] // 2:
                right_id[identity[0]] += 1
                curr_max = np.argmax(right_id)

            frame = cv.rectangle(frame, (x1, y1), (x2, y2), (50, 205, 50), 2)
            cv.putText(frame, str(f'person_{np.argmax(identity)} {identity_score}'),
                       (x1 + 5, y1 - 5), font, 0.5, (50, 205, 50), 2)

        if len(faces_in_frame) == 1:
            break

    print(left_id, right_id)
    return frame, identities, np.sum(left_id) > np.sum(right_id)


# %% all vids encodings
all_encodings = []
join_all_vids_folder_df = pd.DataFrame()
for db_folder in tqdm(db_folders_path):
    vpart = db_folder.replace('\\', '/').split('/')[-1]

    all_videos_in_folder = sorted(list(filter(lambda x: '.mp4' in x, os.listdir(db_folder))),
                                  key=lambda x: int(x.split('.mp4')[0][-1]))
    all_videos_in_folder = [cv.VideoCapture(os.path.join(db_folder, x)) for x in all_videos_in_folder]
    if len(all_videos_in_folder) <= 2:
        continue

    for it in range(len(all_videos_in_folder)):
        all_videos_in_folder[it].set(cv.CAP_PROP_POS_FRAMES, 30 * 60 * 1)

    v1_frame = all_videos_in_folder[0].read()[1]
    v2_frame = all_videos_in_folder[1].read()[1]
    v3_frame = all_videos_in_folder[2].read()[1]



    p1_encodings, amount1_faces = face_rec_create_encodings(vc=all_videos_in_folder[1], ret_amount_faces=True)
    p2_encodings, amount2_faces = face_rec_create_encodings(vc=all_videos_in_folder[2], ret_amount_faces=True)
    all_encodings.append(dict(encodings=[p1_encodings, p2_encodings],
                              amount_faces=[amount1_faces, amount2_faces],
                              v1_frame=v1_frame,
                              v2_frame=v2_frame,
                              v3_frame=v3_frame))
    # TODO:
    #  GARANTIR QUE O ENCODING PRO 1 É SEMPRE ESQUERDA E O ENCODING PRO 2 É SEMPRE DIREITA.
    if amount1_faces == 1 and amount2_faces == 1:
        v1_frame, ids1, lf1 = make_bounding_box_from_encodings(all_videos_in_folder[0], [p1_encodings, p2_encodings], 1)
        reference_embedings = make_embedings_from_reference_video(all_videos_in_folder[0])
        v2_frame, ids2, lf2 = make_bounding_box_from_encodings(all_videos_in_folder[1], [p1_encodings, p2_encodings],
                                                               reference_embedings=reference_embedings)
        v3_frame, ids3, lf3 = make_bounding_box_from_encodings(all_videos_in_folder[2], [p1_encodings, p2_encodings])
        print(ids1, ids2, ids3)
        fig, axes = plt.subplots(1, 3, dpi=720//9, figsize=(21, 9))
        axes[0].imshow(v1_frame[:, :, ::-1])
        axes[1].imshow(v2_frame[:, :, ::-1])
        axes[2].imshow(v3_frame[:, :, ::-1])
        #plt.savefig(f'../figs_for_2_videos/{vpart}.pdf')
        plt.show()

    for it in range(len(all_videos_in_folder)):
        all_videos_in_folder[it].release()

    break


# %%
x_train_, y_train_ = gen_train_set_from_encodings(p1_encodings, p2_encodings)

# %%
svc = SVC(gamma='auto')
rfc = RandomForestClassifier(n_estimators=50, random_state=1)
knn = KNN(n_neighbors=5)
adb = AdaBoostClassifier(n_estimators=100, random_state=1)

clf = VotingClassifier(estimators=[
    ('svc', svc), ('rfc', rfc), ('knn', knn)], voting='hard'
)
clf = knn
clf.fit(x_train_, y_train_)
print(clf.score(x_train_, y_train_))

# %%
for it in range(len(all_videos_in_folder)):
    all_videos_in_folder[it].set(cv.CAP_PROP_POS_FRAMES, 30 * 60 * 1)

    cv.namedWindow("Face Recognizer")
    vc_ = all_videos_in_folder[1]

    font = cv.FONT_HERSHEY_SIMPLEX
    face_cascade = cv.CascadeClassifier('analysis/haarcascade_frontalface_default.xml')
    left_embeddings = []
    right_embeddings = []

    left_id = [0, 0]
    right_id = [0, 0]
    while True:
        ret, img = vc_.read()
        # print('Faces Detected: ', len(faces))
        frame = img

        identities = []
        faces = face_recognition.face_locations(frame[:, :, ::-1], number_of_times_to_upsample=1, model='cnn')
        for face in faces:

            embedding = face_recognition.face_encodings(frame[:, :, ::-1], [face])
            identity = clf.predict(embedding)
            identity_score = clf.predict_proba(embedding)[0]

            x1 = face[3]
            y1 = face[0]

            x2 = face[1]
            y2 = face[2]

            curr_max = None
            if x1 <= frame.shape[0] // 2:
                left_id[identity[0]] += 1
                curr_max = np.argmax(left_id)
            elif x1 > frame.shape[0] // 2:
                right_id[identity[0]] += 1
                curr_max = np.argmax(right_id)

            # if x1 <= frame.shape[0] // 2 and int(np.mean(left_id)) == identity[0] and len(left_id) > 10:
            #     p1_encodings.append(embedding)
            # elif x1 > frame.shape[0] // 2 and int(np.mean(right_id)) == identity[0] and len(left_id) > 10:
            #     right_embeddings.append(embedding)

            img = cv.rectangle(frame, (x1, y1), (x2, y2), (50, 205, 50), 2)
            cv.putText(img, str(f'person_{curr_max + 1} {identity_score}'),
                       (x1 + 5, y1 - 5), font, 0.5, (50, 205, 50), 2)

        cv.imshow("Face Recognizer", img)
        key = cv.waitKey(30)

        if key == 27:  # exit on ESC
            break


    cv.destroyAllWindows()

# %%
for it in range(len(all_videos_in_folder)):
    all_videos_in_folder[it].release()


# %%
font = cv.FONT_HERSHEY_SIMPLEX
def find_faces_ids_and_embedings_sorted_left_2_right(captures, show_video1_frame=False, fig_name=None):
    ret, frame = captures[0].read()
    if not ret:
        return [], []

    faces = face_recognition.face_locations(frame[:, :, ::-1], number_of_times_to_upsample=1, model='cnn')
    f1_mid = np.abs((faces[0][1] - faces[0][3]) / 2) + faces[0][3]
    f2_mid = np.abs((faces[1][1] - faces[1][3]) / 2) + faces[1][3]
    x_middle = np.abs((f1_mid - f2_mid) / 2) + np.min([f1_mid, f2_mid])

    left_id = 0 if f1_mid <= x_middle else 1
    right_id = 0 if left_id == 1 else 1

    left_encodings = face_recognition.face_encodings(frame[:, :, ::-1], [faces[left_id]])
    right_encodings = face_recognition.face_encodings(frame[:, :, ::-1], [faces[right_id]])
    left_encodings = left_encodings[0]; right_encodings = right_encodings[0]

    p1_encodings, amount1_faces = face_rec_create_encodings(vc=captures[1], ret_amount_faces=True)#, pbar=tqdm())
    p2_encodings, amount2_faces = face_rec_create_encodings(vc=captures[2], ret_amount_faces=True)#, pbar=tqdm())
    xtrain, ytrain = gen_train_set_from_encodings(p1_encodings, p2_encodings)
    knn = KNN(n_neighbors=5)
    knn.fit(xtrain, ytrain)

    if show_video1_frame:
        cv.circle(frame, center=(int(f1_mid), faces[0][2]), radius=5, color=(255, 0, 0), thickness=-1)
        p1_str = 'left' if f1_mid <= x_middle else 'right'
        vid1_id = knn.predict_proba(left_encodings.reshape((1, -1))) if p1_str == 'left' \
            else knn.predict_proba(right_encodings.reshape((1, -1)))
        cv.putText(frame, str(f'person_{vid1_id} {p1_str}'), (int(f1_mid) + 5, faces[0][2] - 5), font, 0.5,
                   (50, 205, 50), 2)

        cv.circle(frame, center=(int(f2_mid),  faces[1][2]), radius=5, color=(0, 255, 0), thickness=-1)
        p2_str = 'left' if f2_mid <= x_middle else 'right'
        vid2_id = knn.predict_proba(left_encodings.reshape((1, -1))) if p2_str == 'left' \
            else knn.predict_proba(right_encodings.reshape((1, -1)))
        cv.putText(frame, str(f'person_{vid2_id} {p2_str}'), (int(f2_mid) + 5, faces[0][2] - 5), font, 0.5,
                   (50, 205, 50), 2)

        # print(x_middle, frame.shape, f1_mid, f2_mid)

        vid1_id = np.argmax(vid1_id) + 1
        vid2_id = np.argmax(vid2_id) + 1

        cv.circle(frame, center=(int(x_middle),  faces[0][2]), radius=5, color=(0, 0, 255), thickness=-1)
        ret, left_frame = captures[vid1_id].read() if p1_str == 'left' else captures[vid2_id].read()
        ret, right_frame = captures[vid1_id].read() if p2_str == 'left' else captures[vid2_id].read()
        frame = cv.hconcat([frame, left_frame, right_frame])
        plt.figure(0, dpi=720//9, figsize=(21, 9))
        plt.imshow(frame[:, :, ::-1])
        plt.savefig(f'../fig-folder-libras/{fig_name}.pdf')
        plt.show()

    left_id = np.argmax(knn.predict_proba(left_encodings.reshape((1, -1))))
    right_id = np.argmax(knn.predict_proba(right_encodings.reshape((1, -1))))

    return ([p1_encodings if left_id == 0 else p2_encodings, p1_encodings if right_id == 0 else p2_encodings],
            [left_id + 1, right_id + 1])

def draw_bbox_in_frame(frame, face, id, identity_score):
    x1 = face[3]
    y1 = face[0]

    x2 = face[1]
    y2 = face[2]
    id = identity[0]
    frame = cv.rectangle(frame, (x1, y1), (x2, y2), (50, 205, 50), 2)
    cv.putText(frame, str(f'person_{id} {identity_score}'),
               (x1 + 5, y1 - 5), font, 0.5, (50, 205, 50), 2)

    return frame

def check_person_in_single_videos(vc, clf, is_left, amount_frames=20, pbar=None):
    if pbar is not None:
        pbar.reset(total=amount_frames)

    ids = [0, 0]
    for _ in range(amount_frames):
        ret, frame = vc.read()
        faces_in_frame = face_recognition.face_locations(frame[:, :, ::-1], number_of_times_to_upsample=1, model='cnn')

        for face in faces_in_frame:
            embedding = face_recognition.face_encodings(frame[:, :, ::-1], [face])
            identity = clf.predict(embedding)
            identity_score = clf.predict_proba(embedding)[0]
            ids[identity[0]] += 1

        if pbar is not None:
            pbar.update(1)
            pbar.refresh()

    left = np.argmax(ids)
    print(left == 0 if is_left else left != 0)
    return ids

# %%
for db_folder in tqdm(db_folders_path):
    v_part = db_folder.split(' v')[-1]
    fig_name_path = f'../fig-folder-libras/{v_part}.pdf'
    if os.path.exists(fig_name_path):
        continue

    all_videos_in_folder = sorted(list(filter(lambda x: '.mp4' in x, os.listdir(db_folder))),
                                  key=lambda x: int(x.split('.mp4')[0][-1]))
    all_videos_in_folder = [cv.VideoCapture(os.path.join(db_folder, x)) for x in all_videos_in_folder]
    if len(all_videos_in_folder) < 3:
        for it in range(len(all_videos_in_folder)):
            all_videos_in_folder[it].release()

        continue

    for it in range(len(all_videos_in_folder)):
        all_videos_in_folder[it].set(cv.CAP_PROP_POS_FRAMES, 30 * 60 * 1)


    encodes, p_ids = find_faces_ids_and_embedings_sorted_left_2_right(all_videos_in_folder, show_video1_frame=True,
                                                                      fig_name=v_part)

    for it in range(len(all_videos_in_folder)):
        all_videos_in_folder[it].release()

# %%
clf = KNN(n_neighbors=5)
x_train_, y_train_ = gen_train_set_from_encodings(left_person_encodes, right_person_encodes)
clf.fit(x_train_, y_train_)

# %%
ids = check_person_in_single_videos(all_videos_in_folder[2], is_left=True, clf=clf, pbar=tqdm())


# %%
for it in range(len(all_videos_in_folder)):
    all_videos_in_folder[it].release()
# %%
db_path = '/home/lucasamaral/Documents/LibrasCorpus/Santa Catarina/Inventario Libras'
db_folders_path = [os.path.join(db_path, x) for x in os.listdir(db_path)]
db_folders_path = sorted(db_folders_path, key=lambda x: int(x.split(' v')[-1]), reverse=True)
print(db_folders_path)

# %%
for db_folder in tqdm(db_folders_path):
    non_ended_in_cam = sorted(list(filter(lambda x: '1.mp4' in x, os.listdir(db_folder))),
                              key=lambda x: int(x.split('.mp4')[0][-1]))
    non_ended_in_cam = list(filter(lambda x: 'FLN' in x, non_ended_in_cam))

    ended_in_cam = list(filter(lambda x: 'm.mp4' in x, os.listdir(db_folder)))
    new_video = list(filter(lambda x: 'new_video_' in x, os.listdir(db_folder)))

    if len(ended_in_cam) > 0:
        shutil.copy(os.path.join(db_folder, ended_in_cam[0]), os.path.join(db_folder, non_ended_in_cam[0]))
    elif len(new_video) > 0:
        shutil.copy(os.path.join(db_folder, new_video[0]), os.path.join(db_folder, non_ended_in_cam[0]))
    # else:
    #     print(non_ended_in_cam)
    #shutil.copy(ended_in_cam, non_ended_in_cam)
