from torchvision import transforms
from PIL import Image
import os
import shutil


augmentations = transforms.Compose([
    transforms.RandomRotation(20),  # Rotate the image randomly in the range of (-20, +20) 
    transforms.RandomHorizontalFlip(),  # Flip the image horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2),   # Randomly change the brightness, contrast
    transforms.RandomResizedCrop(
        size=(384, 512),    # image size
        scale=(0.8, 1.2),   # 0.8 -> zoom-out, 1.0 -> original, 1.2 -> zoom-n
        ratio=(4/3, 4/3),   # fixed aspect ratio for 512*384
    ),
])

# directory of the original dataset
INPUT_FOLDER = "dataset/"

# directory to save the augmented data
OUTPUT_FOLDER = "dataset_augmented/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# threshold / target per class
TARGET_COUNT = 500

def augment_data():
    for cls in CLASSES:
        # get folder of the class
        folder_path = os.path.join(INPUT_FOLDER, cls)
        # get images in the class folder
        images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]


        current_count = len(images)   # current number of images in the class
        required_number = TARGET_COUNT - current_count    # required number of images to augment to reach the target class size
        i = 0

        # make a directody inside the output directory for the class
        output_dir = os.path.join(OUTPUT_FOLDER, cls)
        os.makedirs(output_dir, exist_ok=True)
        
        while i < required_number:
            img_name = images[i % current_count]
            img = Image.open(os.path.join(folder_path, img_name))

            # apply transformations on the image
            augmented_img = augmentations(img)

            # save the transformed image in the target directory
            augmented_img.save(os.path.join(output_dir, f"{cls}_aug_{i}.jpg"))
            i += 1
        
        # copy the original images as well to the target directory
        shutil.copytree(folder_path, output_dir, dirs_exist_ok=True)


def get_img_count_per_class():
    class_counts = {}

    for cls in CLASSES:
        # get the folder of the class
        folder_path = os.path.join(OUTPUT_FOLDER, cls)
        # get the images from the class folder
        images = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[cls] = len(images)

    print("Image counts per class:")
    total_images = 0
    for cls, cls_count in class_counts.items():
        print(f"{cls}: {cls_count}")
        total_images += cls_count

    print(f"Total number of images in the dataset: {total_images}")


if __name__ == '__main__':
    augment_data()
    get_img_count_per_class()