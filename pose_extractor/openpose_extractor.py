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
    def __init__(self, version='openpose', openpose_path='./', num_gpu=1):
        self.version = version
        self.op_wrapper = None
        if self.version == 'openpose':
            openpose_build_path = os.path.join(openpose_path, 'build/')
            self.try_import_openpose(openpose_build_path)
            params = dict()
            model_folder = os.path.join(openpose_path, 'models')
            params["model_folder"] = model_folder
            params["hand"] = 1
            params["hand_detector"] = 0
            params['face'] = 0
            params['render_pose'] = 0
            self.num_gpu = num_gpu
            if num_gpu > 1:
                params['num_gpu'] = num_gpu

            # params["body"] = True
            self.op_wrapper = op.WrapperPython()
            self.op_wrapper.configure(params)
            self.op_wrapper.start()
        self.op = op

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
                os.environ['PATH'] = os.environ['PATH'] + ';' \
                                     + openpose_build_path + 'x64/Release;' \
                                     + openpose_build_path + 'bin;'

                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder
                # (Release/x64 etc.)
                sys.path.append(os.path.join(openpose_build_path, 'python'))
                # If you run `make install`
                # (default path is `/usr/local/python` for Ubuntu), you can
                # also access the OpenPose/python module from there. This will
                # install OpenPose and the python library at your desired
                # installation path. Ensure that this is in your python path in
                # order to use it. sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except (ImportError, ModuleNotFoundError) as e:
            print('Error: OpenPose library could not be found. Did you enable '
                  '`BUILD_PYTHON` in CMake and have this Python script in the '
                  'right folder?')
            raise e

    def extract_poses(self, img, filter_low_acc=None):
        datum = op.Datum()
        datum.cvInputData = img
        self.op_wrapper.emplaceAndPop([datum])
        return datum

    def extract_multiple_gpus(self, im_list):
        # Read and push images into OpenPose wrapper
        pose_list = []
        try:
            datums = []
            for im in im_list:
                if im is None:
                    continue

                datum = op.Datum()
                datum.cvInputData = im
                datums.append(datum)
                self.op_wrapper.waitAndEmplace([datum])

            for d_id in range(len(datums)):
                datum = datums[d_id]
                self.op_wrapper.waitAndPop([datum])
                pose_list.append(datum)
        except BaseException as e:
            print(f' AT OP EXTRACTOR {e}')

        return pose_list
