import os
import random
import shutil

# CONFIGURATION
DATA_DIR = "./data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
VAL_SPLIT = 0.2  # 20% for validation

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_class(class_name):
    src_dir = os.path.join(DATA_DIR, class_name)
    images = [f for f in os.listdir(src_dir) if f.lower().endswith('.jpeg')]
    random.shuffle(images)
    n_val = int(len(images) * VAL_SPLIT)
    val_images = images[:n_val]
    train_images = images[n_val:]

    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    val_class_dir = os.path.join(VAL_DIR, class_name)
    ensure_dir(train_class_dir)
    ensure_dir(val_class_dir)

    # Move images
    for img in train_images:
        shutil.move(os.path.join(src_dir, img), os.path.join(train_class_dir, img))
    for img in val_images:
        shutil.move(os.path.join(src_dir, img), os.path.join(val_class_dir, img))

def main():
    random.seed(42)
    classes = [d for d in os.listdir(DATA_DIR)
               if os.path.isdir(os.path.join(DATA_DIR, d)) and d not in ["train", "val"]]
    for class_name in classes:
        split_class(class_name)
    print("Splitting complete. Check the 'train' and 'val' folders inside data.")

if __name__ == "__main__":
    main()
