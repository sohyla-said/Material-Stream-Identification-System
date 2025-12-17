import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from PIL import Image
import torch
from torchvision import models, transforms      # torchvision.models -> contains pretrained models like ResNet-18, ResNet-50

import glob
import joblib
import json

# svm_model = joblib.load("models/deployment/svm_model.joblib")
scaler = joblib.load("models/deployment/scaler.joblib")

with open("models/deployment/svm_threshold.json") as f:
    svm_threshold = json.load(f)["threshold"]

# same svm predict with rejection logic as in Models notebook
def svm_predict_with_rejection(model, x, threshold, unknown_label=6):
    probs = model.predict_proba(x)
    max_probs = np.max(probs, axis=1)       # confidence
    preds = np.argmax(probs, axis=1)        # predicted class
    final_preds = np.where(max_probs >= threshold, preds, unknown_label)
    return final_preds, max_probs

def predict(dataFilePath, bestModelPath):



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess_images = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # convert to PyTorch Tensor
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    model = models.resnet50(pretrained = True)
    model = torch.nn.Sequential(*list(model.children())[:-1])

    # move model to CPU/GPU
    model.to(device).eval()

    image_paths = sorted(glob.glob(os.path.join(dataFilePath, "*")))

    svm_model = joblib.load(bestModelPath)

    features = []

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
            except Exception as e:
                print(f"Skipping {path}, error: {e}")

    
    predictions = []
    confidence = []

    for feature in features:
        feature_scaled = scaler.transform(feature.reshape(1, -1))

        preds, confs = svm_predict_with_rejection(svm_model, feature_scaled, svm_threshold, unknown_label=6)

        pred_id = int(preds[0])
        conf = float(confs[0])

        predictions.append(pred_id)
        confidence.append(conf)

    return predictions, confidence

