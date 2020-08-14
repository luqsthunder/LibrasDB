import os
from shutil import copyfile

dest_dir = '../libras-db-folders/'
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

src_csv_dir = './only-csv'
csv_files_path = list(filter(lambda x: '.csv' in x, os.listdir(src_csv_dir)))

count_classes = {}
for f_name in csv_files_path:
    split_idx = 5 if 'Grande_Florianopolis' not in f_name else 7
    cls_name = f_name.split('-')
    cls_name = cls_name[split_idx]
    if cls_name not in count_classes.keys():
        count_classes.update({cls_name: 1})
    else:
        count_classes[cls_name] += 1

class_names = list(count_classes.keys())
for cls_name in class_names:
    cls_path = os.path.join(dest_dir, cls_name, 'hands-xy')
    if not os.path.exists(cls_path):
        os.makedirs(cls_path)

    cls_path = os.path.join(dest_dir, cls_name, 'hands-angle')
    if not os.path.exists(cls_path):
        os.makedirs(cls_path)

for f_name in csv_files_path:
    split_idx = 5 if 'Grande_Florianopolis' not in f_name else 7
    cls_name = f_name.split('-')[split_idx]
    f_name_path = os.path.join(src_csv_dir, f_name)
    f_name_dest = os.path.join(dest_dir, cls_name, 'hands-xy', f_name)
    copyfile(f_name_path, f_name_dest)
    #print()
print(count_classes)