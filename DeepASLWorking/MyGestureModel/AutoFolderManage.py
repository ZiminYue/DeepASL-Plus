import os
import shutil
import random

def split_dataset(src_dir, dest_dir, train_size=50, val_size=20):
    """
    将每个类别文件夹中的图片，按指定比例划分为训练集和验证集，并复制到对应的文件夹。
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 获取类别文件夹（即字母或数字）
    classes = os.listdir(src_dir)

    for class_name in classes:
        class_path = os.path.join(src_dir, class_name)

        if not os.path.isdir(class_path):  # 只处理文件夹
            continue

        # 创建训练集和验证集文件夹
        train_class_dir = os.path.join(dest_dir, 'train', class_name)
        val_class_dir = os.path.join(dest_dir, 'val', class_name)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # 获取当前类别的所有图片
        images = os.listdir(class_path)
        random.shuffle(images)  # 打乱顺序

        # 按照指定的数量分配训练集和验证集
        train_images = images[:train_size]
        val_images = images[train_size:train_size+val_size]

        # 复制训练集图片
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))

        # 复制验证集图片
        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_class_dir, img))

    print("数据集拆分完成！")

# 数据集源路径
source_directory = './your_dataset'

# 数据集目标路径
destination_directory = './new_dataset'

# 调用拆分函数（每个类别 50 张训练集，20 张验证集）
split_dataset(source_directory, destination_directory, train_size=50, val_size=20)
