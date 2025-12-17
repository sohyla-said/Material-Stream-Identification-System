import os
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import glob
import joblib
import json

scaler = joblib.load("models/deployment/scaler.joblib")

with open("models/deployment/svm_threshold.json") as f:
    svm_threshold = json.load(f)["threshold"]


def predict(dataFilePath, bestModelPath):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # required preprocessing for ResNet-50
    preprocess_images = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # convert to PyTorch Tensor
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    resnet_model = models.resnet50(pretrained = True)
    resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
    resnet_model.to(device).eval()

    # Load image
    image_paths = sorted(glob.glob(os.path.join(dataFilePath, "*")))

    # Load best model
    svm_model = joblib.load(bestModelPath)

     # Feature Extraction
    features = []
    for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                img = preprocess_images(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    # run image throush ResNet model
                    feature = resnet_model(img).squeeze().cpu().numpy()
                
                features.append(feature)
            except Exception as e:
                print(f"Skipping {path}, error: {e}")

    # Make predictions    
    predictions = []
    confidence = []
    for feature in features:
        # scale image
        feature_scaled = scaler.transform(feature.reshape(1, -1))

        # predict
        probs = svm_model.predict_proba(feature_scaled)
        confs = np.max(probs, axis=1)       # confidence
        preds = np.argmax(probs, axis=1)        # predicted class
        final_preds = np.where(confs >= svm_threshold, preds, 6)

        pred_id = int(final_preds[0])
        conf = float(confs[0])

        predictions.append(pred_id)
        confidence.append(conf)

    return predictions, confidence

