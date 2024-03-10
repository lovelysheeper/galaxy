import os
from shutil import copy2

import pandas as pd

#Set the base paths
base_path = r'../dataset'
training_solutions = os.path.join(base_path, 'solutions.csv')
training_images    = os.path.join(base_path, 'images')

df = pd.read_csv(training_solutions)
# print(df)
elliptical = df[(df['Class1.1'] > 0.7) & (df['Class6.2'] > 0.7)]['GalaxyID'].tolist()
# print(elliptical)
face_on_disk = df[(df['Class1.2'] > 0.7) & (df['Class2.2']/(df['Class1.2'] + 0.0001) > 0.7)
                    & (df['Class6.2'] > 0.7)]['GalaxyID'].tolist()
# print(face_on_disk)
edge_on_disk = df[(df['Class1.2'] > 0.7) & (df['Class2.1']/(df['Class1.2'] + 0.0001) > 0.7)
                    & (df['Class6.2'] > 0.7)]['GalaxyID'].tolist()
# print(edge_on_disk)
merging = df[(df['Class6.1'] > 0.7) & (df['Class8.6']/(df['Class6.2'] + 0.0001) > 0.7)]['GalaxyID'].tolist()
# print(merging)

print('Total number of elliptical examples: ',  len(elliptical))
print('Total number of face_on_disk examples: ',  len(face_on_disk))
print('Total number of edge_on_disk examples: ',  len(edge_on_disk))
print('Total number of merging examples: ',  len(merging))

print("开始数据集划分")
# 在目标目录下创建文件夹
split_names = ['elliptical', 'face_on_disk', 'edge_on_disk', 'merging']
for split_name in split_names:
    split_path = os.path.join("../dataset", split_name)
    if os.path.isdir(split_path):
        pass
    else:
        os.mkdir(split_path)

i = 0
for category in [elliptical, face_on_disk, edge_on_disk, merging]:
    goal_path = os.path.join("../dataset", split_names[i])
    i = i + 1
    for j in category:
        img_path = os.path.join("../dataset/images", str(j) + ".jpg")
        copy2(img_path, goal_path)
print("划分完毕")