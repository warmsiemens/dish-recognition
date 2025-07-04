import os
import shutil
import random


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
image_src_dir = os.path.join(BASE_DIR, 'data', 'raw_frames')
label_src_dir = os.path.join(BASE_DIR, 'data', 'cvat_labels', 'obj_train_data')
dst_dir = os.path.join(BASE_DIR, 'data', 'dataset')

subsets = ['train', 'val', 'test']
for subset in subsets:
    os.makedirs(os.path.join(dst_dir, 'images', subset), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, 'labels', subset), exist_ok=True)


txt_files = [f for f in os.listdir(label_src_dir) if f.endswith('.txt')]
random.shuffle(txt_files)


n = len(txt_files)
train_cut = int(n * 0.8)
val_cut = int(n * 0.9)

subset_map = {
    'train': txt_files[:train_cut],
    'val': txt_files[train_cut:val_cut],
    'test': txt_files[val_cut:]
}


for subset, files in subset_map.items():
    for txt_file in files:
        img_file = txt_file.replace('.txt', '.jpg')

        shutil.copy(os.path.join(image_src_dir, img_file), os.path.join(dst_dir, 'images', subset, img_file))
        shutil.copy(os.path.join(label_src_dir, txt_file), os.path.join(dst_dir, 'labels', subset, txt_file))

print('Подготовка данных завершена')
