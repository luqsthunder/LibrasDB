import numpy as np
import pandas as pd
from pose_extractor.all_parts import BODY_PARTS, HAND_PARTS


class PoseCentroidTracker:
    """
    Classe para rastrear os esqueletos e mãos, a classe funciona minimizando
    a distância entre os centroid dos esqueletos rastreados e atribuindo os
    esqueletos mais proximos aos seus respectivos ID, e um novo ID aos
    esqueletos mais distantes que um limitante (threshold).
    """

    def __init__(self, dist_threshold=20, persons_id=None, body_parts=None,
                 hand_parts=None, head_parts=None):
        """
        Parameters
        ----------
        persons_id: list[int] or None

        body_parts: list[str] or None

        hand_parts: list[str] or None

        head_parts: list[str] or None

        """
        self.persons_id = persons_id if persons_id is not None else []
        self.body_parts = BODY_PARTS if body_parts is not None else body_parts
        self.hands_parts = HAND_PARTS if hand_parts is not None else hand_parts
        self.head_parts = head_parts

    def register_persons_from_subtitle(self, subtitle_df, extractor):
        pass

    def __make_persons_list(self, poses):
        """
        Parameters
        ----------

        Returns
        -------
        """

        persons_list = []
        for it, persons in enumerate(poses.poseKeypoints):
            person = {
                'id': it,
                'centroid': self.make_xy_centroid(poses.poseKeypoints[it])}
            persons_list.append(person)
        return persons_list

    def update(self, poses):
        pass

    @staticmethod
    def make_xy_centroid(pose):
        """
        Função para calcular um centroid de um conjunto de juntas no formato xy.

        Parameters
        ----------
        pose: numpy.array or numpy.ndarray
            Contem um vetor de juntas contend.

        Returns
        -------
        np.array
            Centroid das juntas xy informadas.
        """

        length = pose.shape[0]
        x_part = np.sum(pose[:, 0])
        y_part = np.sum(pose[:, 1])

        centroid = np.array([x_part / length, y_part / length])

        return centroid


class PoseAngleDataframe:
    """
    Classe para axiliar na construção dos dataframes das poses em angulos.

    Essa classe pode ser utilizada com contexto do python e.g:

    ```
    with PoseAngleDataframe(['Neck', 'Nose', ...], 'my-pose-df.csv') as pad:
        # process
    ```

    """

    def __init__(self, joint_names, output_file):
        self.joint_names = joint_names
        self.df = pd.DataFrame()
        self.output_file = output_file

    def update_dataframe(self, new_poses):
        pass

    def close(self):
        self.__exit__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        self.df.to_csv(self.output_file)
