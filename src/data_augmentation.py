import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

##############################################

INPUT_DATASET = "dataset/"
OUTPUT_DATASET = "dataset_split/"
TARGET_COUNT = 500
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

augmentations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),      #Randomly rotate the image by a degree between -10 and 10
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2),   #Randomly increase or decrease brightness/contrast by up to +-20%.

])

####################################################

def split_dataset(test_size = 0.2):
    print("Splitting dataset:")

    for cls in CLASSES:
        # get the class folder
        class_folder = os.path.join(INPUT_DATASET, cls)
        # get images in the class folder
        images = [file for file in os.listdir(class_folder) if file.endswith(('.jpg', '.png', '.jpeg'))]

        # define the target directories for train and test images for the class
        train_folder = os.path.join(OUTPUT_DATASET, "train", cls)
        test_folder = os.path.join(OUTPUT_DATASET, "test", cls)
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # split the images into train and test
        train_images, test_images = train_test_split(
            images,
            test_size=test_size,
            random_state=42,
            stratify=[cls] * len(images)
        )

        # copy training images to the train_folder
        for img in train_images:
            shutil.copy(os.path.join(class_folder, img), os.path.join(train_folder, img))

        # copy tetsing images to the train_folder
        for img in test_images:
            shutil.copy(os.path.join(class_folder, img), os.path.join(test_folder, img))

        print(f"Splitted {cls}")

    print("Dataset splitting is done.")

##################################################

def augment_data():
    print("Augmenting data:")

    train_dir = os.path.join(OUTPUT_DATASET, "train")

    for cls in CLASSES:
        # get the class folder from the training images folder
        class_folder = os.path.join(train_dir, cls)
        # get images in the class folder
        images = [file for file in os.listdir(class_folder) if file.endswith(('.jpg', '.png', '.jpeg'))]

        current_count = len(images)
        required = TARGET_COUNT - current_count

        print(f"{cls}: original # = {current_count}, required # = {required}")

        i = 0
        while i < required:
            image_name = images[i % current_count]  # % ensures that the index wraps the around the images list, incase you want to generate images more than the number of original reuse the images in a loop safely.
            image_path = os.path.join(class_folder, image_name)

            image = Image.open(image_path)

            # apply transformations on the image
            augmented = augmentations(image)

            # save the augmented image to the training directory
            augmented.save(os.path.join(class_folder, f"{cls}_aug{i}.jpg"))

            i += 1
    
    print("Augmentation completed")

##################################################

def get_img_count_per_class(split_type = "train"):
    class_counts = {}

    directory = os.path.join(OUTPUT_DATASET, split_type)
    for cls in CLASSES:
        # get the folder of the class
        folder_path = os.path.join(directory, cls)
        # get the images from the class folder
        images = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[cls] = len(images)

    print("Image counts per class:")
    total_images = 0
    for cls, cls_count in class_counts.items():
        print(f"{cls}: {cls_count}")
        total_images += cls_count

    print(f"Total number of images in the {split_type} dataset: {total_images}")

##############################################

if __name__ == '__main__':
    split_dataset()
    augment_data()
    get_img_count_per_class("train")
    get_img_count_per_class("test")
