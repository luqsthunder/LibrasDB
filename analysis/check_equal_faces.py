# %%
import pandas as pd
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
import face_recognition


# %%
db_path = 'D:/gdrive/LibrasCorpus/Santa Catarina/Inventario Libras'
db_folders_path = [os.path.join(db_path, x) for x in os.listdir(db_path)]

# %%
all_videos_in_folder = [cv.VideoCapture(os.path.join(db_folders_path[0], x))
                        for x in os.listdir(db_folders_path[0]) if '.mp4' in x]
for it in range(len(all_videos_in_folder)):
    all_videos_in_folder[it].set(cv.CAP_PROP_POS_FRAMES, 30 * 60 * 1)


# %%
def face_rec_create_encodings(vc, amount_faces: int = 20, show=False):
    encondings = []
    for _ in tqdm(range(amount_faces)):
        ret, frame = vc.read()

        if show:
            cv.imshow('win', frame)
            key = cv.waitKey(-1)
            if key == 27:  # exit on ESC
                break

        face_location = face_recognition.face_locations(frame[:, :, ::-1], number_of_times_to_upsample=1, model='cnn')
        curr_face_encoding = face_recognition.face_encodings(frame[:, :, ::-1], face_location)
        encondings.append(curr_face_encoding)

    cv.destroyAllWindows()
    return encondings


# %% test
for it in range(len(all_videos_in_folder)):
    all_videos_in_folder[it].set(cv.CAP_PROP_POS_FRAMES, 30 * 60 * 1)

#person_1 = create_face_images(0, dual=True, show=True)
try:
    p1_encodings = face_rec_create_encodings(vc=all_videos_in_folder[1])
    p2_encodings = face_rec_create_encodings(vc=all_videos_in_folder[2])
except RuntimeError:
    cv.destroyAllWindows()

# %%

embeddings = []
y_train = []
lbl = 0
for person_encodings in [p1_encodings, p2_encodings]:
    for encode_vec in person_encodings:
        embeddings.append(encode_vec[0])
        y_train.append(lbl)
    lbl = lbl + 1

x_train = np.array(embeddings)
x_train = x_train.reshape(x_train.shape[0], 128)
# %%

svc = SVC(gamma='auto')
rfc = RandomForestClassifier(n_estimators=50, random_state=1)
knn = KNN(n_neighbors=5)
adb = AdaBoostClassifier(n_estimators=100, random_state=1)

clf = VotingClassifier(estimators=[
    ('svc', svc), ('rfc', rfc), ('knn', knn)], voting='hard'
)
clf = knn
clf.fit(x_train, y_train)
print(clf.score(x_train, y_train))

# %%
for it in range(len(all_videos_in_folder)):
    all_videos_in_folder[it].set(cv.CAP_PROP_POS_FRAMES, 30 * 60 * 1)

cv.namedWindow("Face Recognizer")
vc = all_videos_in_folder[0]

font = cv.FONT_HERSHEY_SIMPLEX
face_cascade = cv.CascadeClassifier('analysis/haarcascade_frontalface_default.xml')
left_embeddings = []
right_embeddings = []

left_id = [0, 0]
right_id = [0, 0]
while True:
    ret, img = vc.read()
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