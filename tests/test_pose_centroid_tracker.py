from pose_extractor.pose_centroid_tracker import PoseCentroidTracker


class TestPoseCentroidTracker:

    tracker = None

    def setup(self):
        self.tracker = PoseCentroidTracker('all_videos.csv')

    def test_constructor(self):
        pass

    def test_make_centroid_from_xypose(self):
        pass

    def test_registers_persons_from_sing_df(self):
        folder_name = '../LibrasCorpusScrapy/db\\Santa Catarina\\' \
                      'Inventario Libras' \
                      '\\ Inventário Nacional de Libras - Grande ' \
                      'Florianópolis (Parte II) v1074\\' \
                      'FLN_G1_M1_entrevista_CAM1.mp4'

        self.tracker.register_persons_from_sign_df(folder_name)
