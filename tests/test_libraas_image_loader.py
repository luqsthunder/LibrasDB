from libras_classifiers._librasdb_image_loader import LibrasImageLoader

class TestDBLoader2npy:
    BATCH_SIZE = 8

    def setup(self):
        self.db_image = LibrasImageLoader('../libras-db-folders-debug',
                                          batch_size=self.BATCH_SIZE,
                                          angle_pose=False,
                                          no_hands=False)

    def test_constructor(self):
        try:
            self.db_image = LibrasImageLoader('../libras-db-folders-debug',
                                              batch_size=self.BATCH_SIZE,
                                              angle_pose=False,
                                              no_hands=False)
            assert len(self.db.cls_dirs) > 0
            assert len(self.db.samples_path) > 0

        except BaseException:
            assert False