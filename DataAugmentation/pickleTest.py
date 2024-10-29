import os
import pickle
import numpy as np
from PIL import Image as pil_image

# 读取 train.txt 文件
val_path = 'cifar100/splits/bertinetto/val.txt'
with open(val_path, 'r') as file:
    categories = [line.strip() for line in file.readlines()]

# 初始化字典来保存类别和对应的图像文件名
data_dict = {category: [] for category in categories}

# 设置数据集目录路径
data_dir = 'cifar100/data/'

# 遍历数据集目录中的每个类别目录
for category in categories:
    category_dir = os.path.join(data_dir, category)
    if os.path.isdir(category_dir):
        for filename in os.listdir(category_dir):
            if filename.endswith('.png'):
                data_dict[category].append(os.path.join(category_dir, filename))

# 图像预处理和保存
preprocessed_data = {}
for category, file_paths in data_dict.items():
    preprocessed_data[category] = []
    for path in file_paths:
        img = pil_image.open(path)
        img = img.convert('RGB')
        img = img.resize((32, 32), pil_image.Resampling.LANCZOS)
        img = np.array(img, dtype='float32')
        preprocessed_data[category].append(img)

# 保存预处理后的数据为 pickle 文件
pickle_file = 'dataset/cifar_fs_val.pickle'
with open(pickle_file, 'wb') as file:
    pickle.dump(preprocessed_data, file)

print(f'预处理后的数据已生成并保存为 {pickle_file}')

