import unittest

from libras_classifiers.librasdb_loader_online_sequences import DBLoaderOnlineSequences

class MyTestCase(unittest.TestCase):
    def test_constructor(self):
        DBLoaderOnlineSequences(db_path='../libras-db-folders-online-debug', batch_size=1)


if __name__ == '__main__':
    unittest.main()
