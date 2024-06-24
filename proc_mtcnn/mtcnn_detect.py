import os
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np
import random
import shutil

# 初始化MTCNN模型
mtcnn = MTCNN()

# 指定图片所在的目录
img_dir = './examples'

# 获取所有图片文件
img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

# 随机拆分为训练集和验证集
random.shuffle(img_files)
split_idx = int(len(img_files) * 0.8)
train_files = img_files[:split_idx]
val_files = img_files[split_idx:]

# 创建训练集和验证集文件夹名
train_folder_name = 'examples_train'
val_folder_name = 'examples_val'

# 创建训练集和验证集文件夹
os.makedirs(os.path.join(img_dir, train_folder_name), exist_ok=True)
os.makedirs(os.path.join(img_dir, val_folder_name), exist_ok=True)

# 将图片文件移动到对应的文件夹中
for f in train_files:
    train_folder = os.path.join(img_dir, train_folder_name)
    os.makedirs(train_folder, exist_ok=True)
    shutil.move(os.path.join(img_dir, f), os.path.join(train_folder, f))
for f in val_files:
    val_folder = os.path.join(img_dir, val_folder_name)
    os.makedirs(val_folder, exist_ok=True)
    shutil.move(os.path.join(img_dir, f), os.path.join(val_folder, f))

# 对每个集合中的图片进行关键点检测并保存结果
for folder in [os.path.join(img_dir, train_folder_name), os.path.join(img_dir, val_folder_name)]:
    os.makedirs(os.path.join(folder, 'detections'), exist_ok=True)
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        # 检查路径是否是文件
        if os.path.isfile(img_path):
            # 加载图片
            img = Image.open(img_path)

            # 如果图片是灰度图，跳过并处理下一张图片
            if img.mode != 'RGB':
                continue
            
            # 使用MTCNN进行关键点检测
            boxes, probs, points = mtcnn.detect(img, landmarks=True)
            
            # 如果检测到了关键点
            if points is not None:
                # 将关键点保存到txt文件中
                with open(os.path.join(folder, 'detections', os.path.splitext(img_file)[0] + '.txt'), 'w') as f:
                    for point in points[0]:
                        f.write(f'{point[0]:.2f}\t{point[1]:.2f}\n')