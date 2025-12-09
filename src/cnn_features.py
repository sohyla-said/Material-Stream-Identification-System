# # src/cnn_features.py
# import os, torch, numpy as np
# from torchvision import models, transforms
# from PIL import Image
# import glob

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = models.resnet18(pretrained=True)
# model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final fc
# model.to(device).eval()

# preprocess = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485,0.456,0.406],
#                          std=[0.229,0.224,0.225])
# ])

# CLASSES = ['cardboard','glass','metal','paper','plastic','trash']
# DATASET_DIR = "dataset_split/train"

# features = []
# labels = []
# paths = []
# for class_id, cls in enumerate(CLASSES):
#     folder = os.path.join(DATASET_DIR, cls)
#     for p in glob.glob(os.path.join(folder, "*")):
#         try:
#             img = Image.open(p).convert('RGB')
#             x = preprocess(img).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 feat = model(x).squeeze().cpu().numpy()
#             features.append(feat)
#             labels.append(class_id)
#             paths.append(p)
#         except Exception as e:
#             print("skip", p, e)

# features = np.array(features)
# labels = np.array(labels)
# np.save("models/features_resnet18.npy", features)
# np.save("models/labels_resnet18.npy", labels)

import os, torch, numpy as np
from torchvision import models, transforms
from PIL import Image
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ResNet18 backbone (no FC)
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.to(device).eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
os.makedirs("models", exist_ok=True)

def extract_split(split_name, output_prefix):
    print(f"\nExtracting {split_name.upper()} features...")

    DATASET_DIR = f"dataset_split/{split_name}"
    features = []
    labels = []
    paths = []

    for class_id, cls in enumerate(CLASSES):
        folder = os.path.join(DATASET_DIR, cls)
        img_paths = glob.glob(os.path.join(folder, "*"))

        for p in img_paths:
            try:
                img = Image.open(p).convert('RGB')

                x = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model(x).squeeze().cpu().numpy()

                features.append(feat)
                labels.append(class_id)
                paths.append(p)

            except Exception as e:
                print("Skipping:", p, "error:", e)

    features = np.array(features)
    labels = np.array(labels)

    np.save(f"models/features_{output_prefix}.npy", features)
    np.save(f"models/labels_{output_prefix}.npy", labels)

    print(f"Saved: features_{output_prefix}.npy  shape={features.shape}")
    print(f"Saved: labels_{output_prefix}.npy shape={labels.shape}")


# RUN EXTRACTION
if __name__ == "__main__":
    extract_split("train", "train_resnet18")
    extract_split("test", "test_resnet18")
