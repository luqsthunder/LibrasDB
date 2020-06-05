import numpy as np
import pandas as pd


class PoseCentroidTracker:
    """

    """
    def __init__(self, poses):
        self.poses = poses

    def update(self, poses):
        pass

    @staticmethod
    def make_centroid_from_xypose(pose, acc_treshold=0.3):
        """
        Função para calcular um centroid de um conjunto de juntas no formato xy.

        Parameters
        ----------
        pose: numpy.array or numpy.ndarray
            Contem um vetor de juntas.

        acc_treshold: Float32
            Limitante  para descartar juntas que possua um valor de acuracia
            inferior.

        Returns
        -------
        np.array
            Centroid das juntas xy informadas.
        """
        all_valid_points = np.where(np.all(pose > acc_treshold, axis=2))
        length = all_valid_points.shape[0]
        x_part = np.sum(all_valid_points[:, 0])
        y_part = np.sum(all_valid_points[:, 1])

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
