import os, numpy as np
import torch    # used for deep learning operations
from torchvision import models, transforms      # torchvision.models -> contains pretrained models like ResNet-18, ResNet-50
                                                # transforms -> tools for resizing, normalizing images
from PIL import Image
import glob     # for listing files in drirectory patterns

#############################################

# choose GPU if available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##############################################

# load pretrained ResNet-50
# this model usually ends with classification but we just need the feature extractor
model = models.resnet50(pretrained = True)

# remove the final classification layer
# the model now outputs a 2048-dimensional feature vector
model = torch.nn.Sequential(*list(model.children())[:-1])

# move model to CPU/GPU
model.to(device).eval()

##############################################

# required preprocessing for ResNet-50
# this ensures the images look like the images ResNet was trained on
preprocess_images = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # convert to PyTorch Tensor
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),

])

##############################################

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
# make the directory to save the feature files
os.makedirs("models", exist_ok= True)

##############################################

# read images (training or testing)
# run the images throush ResNet-50
# save NPZ files containg: Feature vectors (2048-D), class labels
def extract_split(split_type):
    print(f"Extracting {split_type} features:")

    file_prefix = f"{split_type}_resnet50"

    dataset_dir = f"dataset_split/{split_type}"

    features = []   # feature vectors
    labels = []     # class labels
    paths = []      # image paths

    for cls_id, cls in enumerate(CLASSES):
        # get the class directory
        class_folder = os.path.join(dataset_dir, cls)
        # get all image paths
        image_paths = glob.glob(os.path.join(class_folder, "*"))

        # process each image
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                img = preprocess_images(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    # run image throush ResNet model, output shape -> (1, 2048, 1, 1)
                    # squeeze to make the shape -> (2048,)
                    feature = model(img).squeeze().cpu().numpy()
                
                # save the results
                features.append(feature)
                labels.append(cls_id)
                paths.append(path)
            except Exception as e:
                print(f"Skipping {path}, error: {e}")

    features = np.array(features)
    labels = np.array(labels)
    np.save(f"models/features_{file_prefix}.npy", features)
    np.save(f"models/labels_{file_prefix}.npy", labels)

    print(f"Saved features_{file_prefix}.npy shape = {features.shape}")
    print(f"Saved labels_{file_prefix}.npy shape = {labels.shape}")

##############################################

if __name__ == '__main__':
    extract_split("train")
    extract_split("test")
