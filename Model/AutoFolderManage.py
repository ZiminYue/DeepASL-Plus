import os
import shutil
import random

def split_dataset(src_dir, dest_dir, train_size=50, val_size=20):
    """
    Divide images in each category folder into a training set and a validation set at a specified ratio, and copied to the corresponding folders.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Get category folders (alphabetical or numerical)
    classes = os.listdir(src_dir)

    for class_name in classes:
        class_path = os.path.join(src_dir, class_name)

        if not os.path.isdir(class_path):  # Only manage folders
            continue

        # Create taining set and validation set folders 
        train_class_dir = os.path.join(dest_dir, 'train', class_name)
        val_class_dir = os.path.join(dest_dir, 'val', class_name)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Get all images from this catagory 
        images = os.listdir(class_path)
        random.shuffle(images) 

        # Divide them into training set and validation set at certain ratio
        train_images = images[:train_size]
        val_images = images[train_size:train_size+val_size]

        # Copy training set images 
        for img in train_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))

        # Copy validation set images 
        for img in val_images:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_class_dir, img))

    print("Dataset folder managed！")

# Dataset Source Path
source_directory = './My Model/aslDataSet'

# Dataset target path 
destination_directory = './My Model/aslDataSet4Training'

# Call the split function (150 training sets and 20 validation sets per category)
split_dataset(source_directory, destination_directory, train_size=150, val_size=20)
