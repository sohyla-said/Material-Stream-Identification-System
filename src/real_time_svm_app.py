import cv2
import numpy as np
import joblib
import json
import os
import sys
from PIL import Image
import torch
# import feature_extraction
sys.path.append(os.path.abspath("src"))
from feature_extraction import preprocess_images, model as feature_model, device

# Load SVM, Scaler, Threshold
svm_model = joblib.load("models/deployment/svm_model.joblib")
scaler = joblib.load("models/deployment/scaler.joblib")

with open("models/deployment/svm_threshold.json") as f:
    svm_threshold = json.load(f)["threshold"]

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
UNKNOWN_LABEL = 6
ALL_CLASSES = CLASSES + ["unknown"]

# same svm predict with rejection logic as in Models notebook
def svm_predict_with_rejection(model, x, threshold, unknown_label=6):
    probs = model.predict_proba(x)
    max_probs = np.max(probs, axis=1)       # confidence
    preds = np.argmax(probs, axis=1)        # predicted class
    final_preds = np.where(max_probs >= threshold, preds, unknown_label)

    return final_preds, max_probs

# Real-time Feature Extraction 
def extract_feature(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    img_tensor = preprocess_images(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = feature_model(img_tensor).squeeze().cpu().numpy()

    return feat.reshape(1, -1)


# Real-Time Application
def run_realtime_svm():

    # initialize web camera
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Could not open webcam.")
        return

    print("Running real-time SVM classification... Press 'q' to exit.")

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        feature = extract_feature(frame)

        feature_scaled = scaler.transform(feature)

        preds, confs = svm_predict_with_rejection(
            svm_model,
            feature_scaled,
            svm_threshold,
            unknown_label=UNKNOWN_LABEL
        )

        pred_id = int(preds[0])
        conf = float(confs[0])
        label = ALL_CLASSES[pred_id]

        # Display on screen 
        cv2.putText(frame, f"{label} ({conf:.2f})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if label != "unknown" else (0, 140, 255), 2)

        cv2.imshow("Real-Time SVM Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_svm()
