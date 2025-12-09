import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

# ------------------------------
# CONFIG
# ------------------------------
ORIGINAL_DATASET = "dataset/"
OUTPUT_DATASET = "dataset_split/"
TARGET_COUNT = 500
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

augmentations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# ------------------------------
# STEP 1: SPLIT DATASET
# ------------------------------
def split_dataset(test_size=0.2):
    print("Splitting dataset...")

    for cls in CLASSES:
        class_dir = os.path.join(ORIGINAL_DATASET, cls)
        images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        train_dir = os.path.join(OUTPUT_DATASET, "train", cls)
        test_dir = os.path.join(OUTPUT_DATASET, "test", cls)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        train_imgs, test_imgs = train_test_split(
            images,
            test_size=test_size,
            random_state=42,
            stratify=[cls] * len(images)
        )

        # Copy training images
        for img in train_imgs:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, img))

        # Copy testing images (NO AUGMENTATION)
        for img in test_imgs:
            shutil.copy(os.path.join(class_dir, img), os.path.join(test_dir, img))

    print("Dataset split completed.")


# ------------------------------
# STEP 2: AUGMENT TRAINING SET
# ------------------------------
def augment_training_data():
    print("Augmenting training data...")

    train_root = os.path.join(OUTPUT_DATASET, "train")

    for cls in CLASSES:
        class_dir = os.path.join(train_root, cls)
        images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        current_count = len(images)
        required = TARGET_COUNT - current_count

        print(f"{cls}: current={current_count}, required augmentation={required}")

        i = 0
        while i < required:
            img_name = images[i % current_count]
            img_path = os.path.join(class_dir, img_name)

            img = Image.open(img_path)
            aug_img = augmentations(img)

            aug_img.save(os.path.join(class_dir, f"{cls}_aug_{i}.jpg"))
            i += 1

    print("Augmentation completed.")


# ------------------------------
# RUN BOTH STEPS
# ------------------------------
if __name__ == '__main__':
    split_dataset()
    augment_training_data()

