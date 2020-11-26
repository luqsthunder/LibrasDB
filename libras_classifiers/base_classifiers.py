import numpy as np

from libras_classifiers.librasdb_loaders import DBLoader2NPY
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

batch_size = 100
joints_to_use = ['frame',
                 'Neck-RShoulder-RElbow',
                 'RShoulder-RElbow-RWrist',
                 'Neck-LShoulder-LElbow',
                 'LShoulder-LElbow-LWrist',
                 'RShoulder-Neck-LShoulder',
                 'left-Wrist-left-ThumbProximal-left-ThumbDistal',
                 'right-Wrist-right-ThumbProximal-right-ThumbDistal',
                 'left-Wrist-left-IndexFingerProximal-left-IndexFingerDistal',
                 'right-Wrist-right-IndexFingerProximal-right-IndexFingerDistal',
                 'left-Wrist-left-MiddleFingerProximal-left-MiddleFingerDistal',
                 'right-Wrist-right-MiddleFingerProximal-right-MiddleFingerDistal',
                 'left-Wrist-left-RingFingerProximal-left-RingFingerDistal',
                 'right-Wrist-right-RingFingerProximal-right-RingFingerDistal',
                 'left-Wrist-left-LittleFingerProximal-left-LittleFingerDistal',
                 'right-Wrist-right-LittleFingerProximal-right-LittleFingerDistal'
                 ]

db = DBLoader2NPY('../../clean_sign_db_front_view',
                  batch_size=batch_size,
                  shuffle=True, test_size=.25,
                  add_angle_derivatives=True,
                  no_hands=False,
                  angle_pose=True,
                  joints_2_use=joints_to_use
                  )
db.fill_samples_absent_frames_with_na()

svc_clf = SVC(kernel='rbf', probability=True)
X = []
y = []

for it, x in enumerate(db.train()):
    X.extend(x[0])
    y.extend(x[1])

x_val = []
y_val = []

for it, x in enumerate(db.validation()):
    x_val.extend(x[0])
    y_val.extend(x[1])

X = np.stack([x.reshape((1, -1)) for x in X])
X = X.reshape((X.shape[0], 735))
y = np.array([np.argmax(x) for x in np.stack(y)]).reshape((-1, 1)).reshape((-1))

x_val = np.stack([x.reshape((1, -1)) for x in x_val])
x_val = x_val.reshape((x_val.shape[0], 735))
y_val = np.array([np.argmax(x) for x in np.stack(y_val)]).reshape((-1, 1)).reshape((-1))

clf_list = [KNeighborsClassifier(n_neighbors=5), AdaBoostClassifier(), RandomForestClassifier(),
            DecisionTreeClassifier(),
            SVC(kernel='rbf', probability=True)]
for clf in clf_list:
    clf.fit(X=X, y=y)
    y_pred = clf.predict(x_val)
    print(clf, accuracy_score(y_val, y_pred))