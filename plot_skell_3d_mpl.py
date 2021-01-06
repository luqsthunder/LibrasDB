from libras_classifiers.librasdb_loaders import DBLoader2NPY
from pose_extractor.all_parts import BODY_PAIRS

import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib.widgets import Slider, Button, RadioButtons

front_data_path = 'v1098_barely_front_view_left_person.csv'
side_data_path = 'v1098_side_view_left_person.csv'
#destination_path = f'./{person}_person_3d_{version}.csv'

left_front_data = pd.read_csv(front_data_path)
df = pd.DataFrame(left_front_data)
left_side_data = pd.read_csv(side_data_path)

left_front_data = left_front_data.applymap(DBLoader2NPY.parse_npy_vec_str)
left_side_data = left_side_data.applymap(DBLoader2NPY.parse_npy_vec_str)
left_side_data.frame = left_front_data.frame


# %%
all_joints_to_save = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
                      'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 'RAnkle', 'LHip',
                      'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar'
                     ]

all_joints_to_use = ['RShoulder', 'RElbow', 'RWrist','LShoulder', 'LElbow', 'LWrist']

threshold = 0.60
all_frames = left_front_data.frame.unique().tolist()
# all_frames = [all_frames[0]] + [all_frames[len(all_frames) // 2]] + [all_frames[len(all_frames) - 2]]

matches = []

for i in range(0,len(all_frames),10):
    row_front = left_front_data[left_front_data['frame'] == all_frames[i]]
    row_front = row_front[all_joints_to_use].values[0]
    row_front = [x if x[2] >= threshold else None for x in row_front]

    row_side = left_side_data[left_side_data['frame'] == all_frames[i]]
    row_side = row_side[all_joints_to_use].values[0]
    row_side = [x if x[2] >= threshold else None for x in row_side]

    for front, side in zip(row_front, row_side):
        if front is not None and side is not None:
            matches.append((front[:2], side[:2]))

    if len(matches) >= 100:
        break

def make_3dpoints(fx, fy, frame=0):
    camera_mat = np.identity(3)
    camera_mat[0][0] = fx
    camera_mat[1][1] = fy

    essential, inliers = cv.findEssentialMat(np.array([x[0] for x in matches]),
                                             np.array([x[1] for x in matches]),
                                             method=cv.RANSAC, cameraMatrix=camera_mat)
    retval, R, t, mask2 = cv.recoverPose(essential, np.array([x[0] for x in matches]),
                                         np.array([x[1] for x in matches]),
                                         mask=inliers)#, cameraMatrix=camera_mat)

    projection1 = np.zeros((3, 4))
    projection1[:3, :3] = camera_mat

    projection2 = np.zeros((3, 4))
    projection2[:3, :3] = R
    projection2[:, 3:] = t

    dic = {
        k: [] for k in left_front_data.keys() if k not in ['frame', 'Unnamed: 0']
    }

    min_frame = left_front_data.frame[frame]
    row_front = left_front_data[left_front_data['frame'] == min_frame]
    row_front = row_front.drop(columns=['frame', 'Unnamed: 0'])
    row_front = row_front.applymap(lambda x: np.array([0.0, 0.0, 0.0]) if isinstance(x, float) else x)
    row_front = [(x.values[0] if x.values[0][2] >= threshold else None, key)
                 for key, x in row_front.iteritems()]

    row_side = left_side_data[left_side_data['frame'] == min_frame]
    row_side = row_side.drop(columns=['frame', 'Unnamed: 0'])
    row_side = row_side.applymap(lambda x: np.array([0.0, 0.0, 0.0]) if isinstance(x, float) else x)
    row_side = [(x.values[0] if x.values[0][2] >= threshold else None, key)
                 for key, x in row_side.iteritems()]

    for front, side in zip(row_front, row_side):
        if front[0] is not None and side[0] is not None:
            points3d = cv.triangulatePoints(projection1, projection2,
                                            front[0][:2].reshape(1, 1, 2),
                                            side[0][:2].reshape(1, 1, 2))
            dic[front[1]].append(points3d.T[0])
        else:
            dic[front[1]].append(None)

    return pd.DataFrame(dic)

fig = plt.figure(figsize=(21, 9), dpi=720//9)
ax = fig.add_subplot(111, projection='3d')

x_scale=10
y_scale=10
z_scale=10

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
  return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj

plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)
a0 = 5
f0 = 1

delta_fx = 1.0
delta_fy = 1.0

df_res = make_3dpoints(1, 1, 0)
global chart
global lines

#-------------------------------------------------------------------------
chart = []
colors = ['r','b','g','y','c','m','k','#56004B','#0B2B6B']
bp = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist','LShoulder', 'LElbow', '?', '??']
i = 0
for key in df_res.keys():
    if df_res[key][0] is not None and df_res[key][0] is not None and df_res[key][0] is not None:
        chart.append(ax.scatter(df_res[key][0][0],df_res[key][0][1],df_res[key][0][2],color = colors[i], label = bp[i], s = 40))
        i+=1
#-------------------------------------------------------------------------

#chart = ax.scatter([df_res[key][0][0] for key in df_res.keys() if df_res[key][0] is not None],
#                   [df_res[key][0][1] for key in df_res.keys() if df_res[key][0] is not None],
#                   [df_res[key][0][2] for key in df_res.keys() if df_res[key][0] is not None],
#                   color = 'k')

lines = []
for pair in BODY_PAIRS:
    if df_res[pair[0]][0] is not None and df_res[pair[1]][0] is not None:
        lines.append(ax.plot(
            [df_res[pair[0]][0][0], df_res[pair[1]][0][0]],
            [df_res[pair[0]][0][1], df_res[pair[1]][0][1]],
            [df_res[pair[0]][0][2], df_res[pair[1]][0][2]],
            color = 'k'
        ))

x_min = min([df_res[key][0][0] for key in df_res.keys() if df_res[key][0] is not None])
x_max = max([df_res[key][0][0] for key in df_res.keys() if df_res[key][0] is not None])

y_min = min([df_res[key][0][1] for key in df_res.keys() if df_res[key][0] is not None])
y_max = max([df_res[key][0][1] for key in df_res.keys() if df_res[key][0] is not None])

z_min = min([df_res[key][0][2] for key in df_res.keys() if df_res[key][0] is not None])
z_max = max([df_res[key][0][2] for key in df_res.keys() if df_res[key][0] is not None])

ax.axes.set_xlim3d(left=x_min, right=x_max)
ax.axes.set_ylim3d(bottom=y_min, top=y_max)
ax.axes.set_zlim3d(bottom=z_min, top=z_max)

ax.margins(x=0)

axcolor = 'lightgoldenrodyellow'
ax_fx = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_fy = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
frame_num = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_fxy = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

slider_fx = Slider(ax_fx, 'Foco X (fx)', 1, 1000.0, valinit=f0, valstep=delta_fx)
slider_fy = Slider(ax_fy, 'Foco Y (fy)', 1, 1000.0, valinit=f0, valstep=delta_fy)
slider_frame = Slider(frame_num, 'Curr Frame', 0, 1000, valinit=f0, valstep=1.0)

slider_fxy = Slider(ax_fxy, 'Foco X e Y (fx e fy)', 1, 1000.0, valinit=f0, valstep=delta_fy)


def update(val):
    f_fx = slider_fx.val
    f_fy = slider_fy.val
    f_frame = int(slider_frame.val)
    update_fig(f_fx, f_fy, f_frame)


def update_fig(fx, fy, frame):
    global chart
    global lines
    df_res = make_3dpoints(fx, fy, frame)

    #-------------------------------------------------------------------------
    for it in range(len(chart)):
        chart[it].remove()
        
    chart = []
    colors = ['r','b','g','y','c','m','k','#56004B','#0B2B6B']
    bp = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist','LShoulder', 'LElbow', '?', '??']
    i = 0
    for key in df_res.keys():
        if df_res[key][0] is not None and df_res[key][0] is not None and df_res[key][0] is not None:
            chart.append(ax.scatter(df_res[key][0][0],df_res[key][0][1],df_res[key][0][2],color = colors[i], label = bp[i], s = 40))
            i+=1
    #-------------------------------------------------------------------------
    #chart.remove()
    #chart = ax.scatter([df_res[key][0][0] for key in df_res.keys() if df_res[key][0] is not None],
    #                   [df_res[key][0][1] for key in df_res.keys() if df_res[key][0] is not None],
    #                   [df_res[key][0][2] for key in df_res.keys() if df_res[key][0] is not None])


    for it in range(len(lines)):
        a = lines[it].pop(0)
        a.remove()

    lines = []
    for pair in BODY_PAIRS:
        if df_res[pair[0]][0] is not None and df_res[pair[1]][0] is not None:
            lines.append(ax.plot(
                [df_res[pair[0]][0][0], df_res[pair[1]][0][0]],
                [df_res[pair[0]][0][1], df_res[pair[1]][0][1]],
                [df_res[pair[0]][0][2], df_res[pair[1]][0][2]],
                color = 'k'
            ))

    x_min = min([df_res[key][0][0] for key in df_res.keys() if df_res[key][0] is not None])
    x_max = max([df_res[key][0][0] for key in df_res.keys() if df_res[key][0] is not None])

    y_min = min([df_res[key][0][1] for key in df_res.keys() if df_res[key][0] is not None])
    y_max = max([df_res[key][0][1] for key in df_res.keys() if df_res[key][0] is not None])

    z_min = min([df_res[key][0][2] for key in df_res.keys() if df_res[key][0] is not None])
    z_max = max([df_res[key][0][2] for key in df_res.keys() if df_res[key][0] is not None])

    ax.axes.set_xlim3d(left=x_min, right=x_max)
    ax.axes.set_ylim3d(bottom=y_min, top=y_max)
    ax.axes.set_zlim3d(bottom=z_min, top=z_max)
    plt.draw()

def update_fxy(val):
    fx = slider_fxy.val
    fy = slider_fxy.val
    frame = slider_frame.val
    slider_fx.set_val(fx)
    slider_fy.set_val(fy)
    update_fig(fx, fy, frame)

slider_fx.on_changed(update)
slider_fy.on_changed(update)
slider_frame.on_changed(update)
slider_fxy.on_changed(update_fxy)

resetax = plt.axes([0.8, 0.008, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    slider_fx.reset()
    slider_fy.reset()
    slider_frame.reset()
    slider_fxy.reset()

button.on_clicked(reset)

ax.legend()

plt.show()