import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from copy import copy
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm


class ConvertPoseToAngle:

    def __init__(self, db_pose_path, base_joint_xaxis=None,
                 base_joint_yaxis=None, joint_list=None):
        """

        Parameters
        ----------
        db_pose_path
           Path para base de dados contentdo os videos das
           poses em csv.

        base_joint_axis
            lista de juntas para montar o eixo
        """
        self.db_pose_path = db_pose_path
        if base_joint_xaxis is None:
            self.base_joint_xaxis = ['RShoulder', 'Neck', 'LShoulder']

        if base_joint_yaxis is None:
            self.base_joint_yaxis = ['Neck', 'Nose']
        if joint_list is None:
            df_cols = [
                "person", "frame",
                "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder",
                "LElbow",
                "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee",
                "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe",
                "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
            ]
            df_cols.extend(['hand{}l'.format(it) for it in range(21)])
            df_cols.extend(['hand{}r'.format(it) for it in range(21)])
            df_cols.extend(['head{}'.format(it) for it in range(68)])

            self.edges_list = [
                (0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8),
                (8, 9), (8, 12), (9, 10), (10, 11), (11, 24), (11, 22),
                (23, 22), (12, 13), (13, 14), (14, 21), (14, 19), (20, 19),
            ]

            self.edges_list = list(map(lambda x: (x[0] + 2, x[1] + 2),
                                       self.edges_list))

            hand_edges = [
                (0, 1), (0, 5), (0, 9), (0, 13), (0, 17), (1, 2), (2, 3),
                (3, 4), (5, 6), (6, 7), (7, 8), (9, 10), (10, 11), (11, 12),
                (13, 14), (14, 15), (15, 16), (17, 18), (18, 19), (19, 20)
            ]

            left_hand_position = df_cols.index('hand0l')
            right_hand_position = df_cols.index('hand0r')
            print(df_cols[left_hand_position], df_cols[right_hand_position])

            left_hand_edges = copy(hand_edges)
            left_hand_edges = [(x[0] + left_hand_position,
                                x[1] + left_hand_position)
                               for x in left_hand_edges]

            right_hand_edges = copy(hand_edges)
            right_hand_edges = [(x[0] + right_hand_position,
                                 x[1] + right_hand_position)
                                for x in right_hand_edges]

            head_position = df_cols.index('head0')
            # em volta do rosto
            head_edges = [(x, x + 1) for x in range(16)]

            # sobrancelhas
            head_edges.extend([(x, x + 1) for x in range(17, 21)])
            head_edges.extend([(x, x + 1) for x in range(12, 26)])

            # nariz
            head_edges.extend([(x, x + 1) for x in range(27, 30)])
            head_edges.extend([(x, x + 1) for x in range(31, 35)])

            # olhos
            head_edges.extend([(x, x + 1) for x in range(36, 41)])
            head_edges.extend([(x, x + 1) for x in range(42, 47)])

            # boca
            head_edges.extend([(x, x + 1) for x in range(48, 59)])
            head_edges.extend([(x, x + 1) for x in range(60, 67)])

            head_edges = [(x[0] + head_position, x[1] + head_position)
                          for x in head_edges]
            self.edges_list.extend(left_hand_edges)
            self.edges_list.extend(right_hand_edges)
            self.edges_list.extend(head_edges)

            self.edges_list = list(map(
                lambda x: (df_cols[x[0]], df_cols[x[1]]),
                self.edges_list
            ))

    def process(self):
        """
        Inicia o processo de converter todos os videos presentes na base de
        dados. De modo que, os videos estão no formato csv das poses em
        coordenadas cartezianas em pixels para ângulo.
        dados.

        """
        for signs_dir in os.listdir(self.db_pose_path):
            signs_dir = os.path.join(self.db_pose_path, signs_dir)

            for pose_video_file in os.listdir(signs_dir):
                pose_video_file = os.path.join(signs_dir, pose_video_file)
                single_video = pd.read_csv(pose_video_file)

                def parse_np_array_as_str(str_array_like):
                    if not isinstance(str_array_like, str):
                        return str_array_like

                    res = str_array_like[1:len(str_array_like) - 1].split(' ')
                    recovered_np_array = []
                    for r in res:
                        if r == '' or ' ' in r:
                            continue
                        f = float(r)
                        recovered_np_array.append(f)
                    return np.array(recovered_np_array)

                single_video = single_video[single_video.keys()].applymap(
                    lambda v: parse_np_array_as_str(v)
                )

                # hand 27, 48 68
                for it in range(len(single_video.index)):
                    black_img = np.ones((1000, 1000, 3)).astype('uint8') * 255
                    for it2, edge in enumerate(self.edges_list):
                        try:
                            first_point = single_video[edge[0]].iloc[it]
                            second_point = single_video[edge[1]].iloc[it]
                            if first_point[2] > .20 and second_point[2] > .20:
                                first_point = first_point[:2]
                                second_point = second_point[:2]

                                color = ((255//3) * (it2 % 3),
                                         (255//4) * (it2 % 4),
                                         (255//5) * (it2 % 5))
                                first_point = tuple(map(int, first_point))
                                second_point = tuple(map(int, second_point))
                                cv.line(black_img, first_point, second_point,
                                        color)
                        except IndexError as idxError:
                            continue
                    plt.imshow(black_img)
                    plt.show()

                # for frame, pose in single_video.iterrows():
                #     angle, orig, xaxis, yaxis = \
                #         self.convert_single_pose(pose)
                #     print(angle, orig, xaxis, yaxis)

    def convert_single_pose(self, curr_pose):
        """
        Converte um pose humana no formato de cada junta sendo (x, y) para
        formato de angulos direcionados 2D.

        Parameters
        ----------
        curr_pose :  Panda's Series contendo a uma pose.

        Returns
        -------
        juntas em representadas por angulo relacionado a junta anterior,
        origem de um eixo de coordenadas de e os eixos de coordenadas.
        """

        orig, x_axis, y_axis = self.make_base_axis(curr_pose)

        # note que orig esta na quantidade normal em PX nas coordenadas da foto.
        pose_in_angles = []

        for key in curr_pose.keys():
            if key in ['person', 'frame']:
                continue

            first_joint = curr_pose[key].value
            second_joint = curr_pose[self.edges_list[key]].value

            # angulo direcional entre dois vetores 2D.
            angle = np.arccos((np.linalg.norm(first_joint) *
                               np.linalg.norm(second_joint)) /
                              np.dot(first_joint, second_joint))
            angle_sign = np.sign(first_joint[0] * second_joint[1] -
                                 first_joint[1] * second_joint[0])
            pose_in_angles.append(angle_sign * angle)

        return pose_in_angles, orig, x_axis, y_axis

    def make_base_axis(self, curr_pose):
        """
        Constroi eixos para pose atual baseado nas juntas previamentes
        definidas em self.base_joint_xaxis para eixo dos X e
        self.base_joint_yaxis

        Params
        ------
        curr_pose : Pose atual para construção dos eixos previamente definidos

        :return Vetor contendo a origem do sistema de coordenadas, e os
                eixos x e y.
        """
        x_axis = list(map(lambda x: np.array(x),
                          curr_pose[self.base_joint_xaxis].values))
        y_axis = list(map(lambda x: np.array(x),
                          curr_pose[self.base_joint_yaxis].values))

        regressor = LinearRegression()

        regressor.fit(np.array([[x[0]] for x in x_axis]),
                      np.array([[x[1]] for x in x_axis]))
        x_axis_line = [regressor.coef_, regressor.intercept_]

        regressor.fit(np.array([y[0] for y in y_axis]),
                      np.array([y[1] for y in y_axis]))
        y_axis_line = [regressor.coef_, regressor.intercept_]

        # note que as duas eqs são, y = mx + b. Logo precisamos fazer
        # m1 * x + b1 = m2 * x + b2
        # (m1 - m2) * x = b2 - b1
        # x = (b2 - b1) / (m1 - m2)
        x_intercept = (y_axis_line[1] - x_axis_line[1]) / \
                      (x_axis_line[0] - y_axis_line[0])

        y_intercept = x_axis_line[0] * x_intercept + x_axis_line[1]
        return np.array([x_intercept, y_intercept]), x_axis, y_axis


if __name__ == '__main__':
    ConvertPoseToAngle('db/').process()
