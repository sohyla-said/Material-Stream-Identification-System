import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import models, transforms
import glob
import joblib
import json

scaler = joblib.load("models/deployment/scaler.joblib")

with open("models/deployment/svm_threshold.json") as f:
    svm_threshold = json.load(f)["threshold"]

CLASS_NAMES =['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'unknown']


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
    image_names = []
    for path in image_paths:
            try:
                image_name = os.path.basename(path)
                image = Image.open(path).convert('RGB')
                img = preprocess_images(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    # run image throush ResNet model
                    feature = resnet_model(img).squeeze().cpu().numpy()
                
                features.append(feature)
                image_names.append(image_name)
            except Exception as e:
                print(f"Skipping {path}, error: {e}")

    # Prediction    
    results = []
    output_excel = "predictions.xlsx"
    for feature,image_name in zip(features, image_names):
        # scale image
        feature_scaled = scaler.transform(feature.reshape(1, -1))

        # predict
        probs = svm_model.predict_proba(feature_scaled)
        confs = np.max(probs, axis=1)       # confidence
        preds = np.argmax(probs, axis=1)        # predicted class
        final_preds = np.where(confs >= svm_threshold, preds, 6)

        pred_id = int(final_preds[0])
        conf = float(confs[0])
        pred_class = CLASS_NAMES[pred_id]

        # results.append([image_name, pred_class, conf])
        results.append([image_name, pred_class])


    # Save results to Excel, create if not exists
    # df = pd.DataFrame(results, columns=["imagename", "predicted_class", "Confidence level"])
    df = pd.DataFrame(results, columns=["imagename", "predicted_class"])
    if not os.path.exists(output_excel):
        df.to_excel(output_excel, index=False)
    else:
        # If file exists, append new predictions (without duplicating header)
        with pd.ExcelWriter(output_excel, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            # Read existing data to find where to start appending
            existing_df = pd.read_excel(output_excel)
            startrow = len(existing_df) + 1
            df.to_excel(writer, index=False, header=False, startrow=startrow)
    print(f"Predictions saved to {output_excel}")
    return df
if __name__ == "__main__":
    folder_path = "test_images/"  
    bestmodel_path = "models/deployment/svm_model.joblib"
    predict(folder_path, bestmodel_path)