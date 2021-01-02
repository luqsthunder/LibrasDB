import pandas as pd
import os


class DBDirectoryTree:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_dir_tree = self.__make_db_directory_tree()

    def __make_db_directory_tree(self):
        db_tree = pd.DataFrame()
        for estate_name in os.listdir(self.db_path):
            estate_path = os.path.join(self.db_path, estate_name)

            for proj_name in os.listdir(estate_path):
                proj_path = os.path.join(estate_path, proj_name)
                for item_name in os.listdir(proj_path):
                    item_path = os.path.join(proj_path, item_name)
                    item_files = os.listdir(item_path)

                    db_tree = db_tree.append(pd.DataFrame(dict(
                        items=item_files, estate=[estate_name] * len(item_files),
                        project=[proj_name] * len(item_files), item_name=[proj_name] * len(item_files)
                        )
                    ))

        return db_tree
