import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sys import platform


def from_datum_to_datumlike(datum):
    pass


class DatumLike:
    def __init__(self):
        self.poseKeypoints = np.array([])
        self.handKeypoints = [np.array([])]
        self.headKeypoints = np.array([])
        self.cvOutputData = np.array([])
        self.cvInputData = np.array([])


class OpenposeExtractor:
    def __init__(self, version='openpose', openpose_path='./'):
        self.version = version
        self.op_wrapper = None
        if self.version == 'openpose':
            openpose_build_path = os.path.join(openpose_path, 'build')
            self.try_import_openpose(openpose_build_path)
            params = dict()
            model_folder = os.path.join(openpose_path, 'models')
            params["model_folder"] = model_folder
            params["hand"] = True
            params["hand_detector"] = 0
            # params["body"] = True
            self.op_wrapper = op.WrapperPython()
            self.op_wrapper.configure(params)
            self.op_wrapper.start()

    @staticmethod
    def try_import_openpose(openpose_build_path):
        """
        Importa a biblioteca openpose de um diretorio especifico.

        Parameters
        ----------
        openpose_build_path: str
            Localidade de onde o openpose foi baixado. Dentro desse path
            deve conter uma pasta com nome build onde o openpose foi compilado.

        Raises
        -------
            Caso não ache o openpose dentro do `openpose_build_path` um
            ImportError será jogado.

        """
        global op
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder
                # (Release/x64 etc.)
                sys.path.append(os.path.join(openpose_build_path,
                                             'python/openpose/Release'))

                # No windows ';' é o separador de itens dentro da variavel do
                # path.
                os.environ['PATH'] = os.environ['PATH'] + ';' + \
                                     os.path.join(openpose_build_path, 'x64/Release') + ';' + \
                                     os.path.join(openpose_build_path, 'bin') + ';'

                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder
                # (Release/x64 etc.)
                sys.path.append('python')
                # If you run `make install`
                # (default path is `/usr/local/python` for Ubuntu), you can
                # also access the OpenPose/python module from there. This will
                # install OpenPose and the python library at your desired
                # installation path. Ensure that this is in your python path in
                # order to use it. sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable '
                  '`BUILD_PYTHON` in CMake and have this Python script in the '
                  'right folder?')
            raise e

    def extract_poses(self, img, filter_low_acc=None) -> DatumLike:
        datum = op.Datum()
        datum.cvInputData = img
        self.op_wrapper.emplaceAndPop([datum])
        return datum
