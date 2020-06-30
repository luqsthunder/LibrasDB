from pose_extractor.find_signaling_centroid import FindSignalingCentroid


class TestFindSignalingCentroid:

    def setup(self):
        self.talk_cent_finder = FindSignalingCentroid('all_videos.csv')

    def test_constructor(self):
        pass

    def test_find_centroid_each_singer_video(self):
        folder_name = \
            './db\\Alagoas\\Inventário de Libras Maceió\\' \
            'Inventário Nacional de Libras - Surdos de Referência v1004\\v0.mp4'
        self.talk_cent_finder.find_each_signaler_frame_talks_alone(folder_name)
