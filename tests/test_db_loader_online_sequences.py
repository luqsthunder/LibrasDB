from libras_classifiers.librasdb_loader_online_sequences import DBLoaderOnlineSequences


class TestDBLoaderOnlineSequences:
    db = None

    def setup(self):
        self.db = DBLoaderOnlineSequences(db_path='../libras-db-folders-online-debug', batch_size=1)

    def test_constructor(self):
        try:
            db = DBLoaderOnlineSequences(db_path='../libras-db-folders-online-debug', batch_size=1)
            if self.db is None:
                self.db = db
            assert True
        except BaseException:
            assert False

    def test_batch_load_samples(self):
        self.db.batch_load_samples([0])
