from pose_extractor.find_signaling_centroid import FindSignalingCentroid
from pose_extractor.pose_centroid_tracker import PoseCentroidTracker
from tqdm.auto import tqdm


class TestFindSignalingCentroid:

    def setup(self):
        self.talk_cent_finder = FindSignalingCentroid('all_videos.csv')
        self.centroid_finder = PoseCentroidTracker('all_videos.csv',
                                                   openpose_path='../../Libraries/repos/openpose')

    def test_constructor(self):
        pass

    def test_find_centroid_each_singer_video(self):
        folder_name = 'Santa Catarina\Inventario Libras\ Inventário Nacional de Libras - Grande Florianópolis ' \
                      '(Parte II) v1074\FLN_G1_M1_entrevista_CAM1.mp4'
        self.talk_cent_finder.find_where_signalers_talks_alone(folder_name)
