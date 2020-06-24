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
        folder_name = \
            './db\\Alagoas\\Inventário de Libras Maceió\\' \
            'Inventário Nacional de Libras - Surdos de Referência v1004\\v0.mp4'
        self.tracker.register_persons_from_sign_df(folder_name)
