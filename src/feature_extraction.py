import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
import os

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
DIRECTORY = "dataset_augmented"

def load_image(path, size=(256, 192)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    return img

def extract_hog(img, pixels_per_cell = (16, 16), cells_per_block = (2, 2), orientations = 9):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feature = hog(gray_img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                      block_norm='L2-Hys', feature_vector=True)
    return hog_feature

def extract_color_hist(img, bins=(16, 16, 8)):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_img], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_texture(img, P=8, R=1):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_img, P, R, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def concatenate_features(path):
    img = load_image(path)
    hog_features = extract_hog(img)
    color_features = extract_color_hist(img)
    texture_features = extract_texture(img)
    features = np.concatenate([hog_features, color_features, texture_features])
    return features


def extract_features():
    X = []
    Y = []

    for class_id, class_name in enumerate(CLASSES):
        folder = f"{DIRECTORY}/{class_name}"

        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            features = concatenate_features(img_path)
            X.append(features)
            Y.append(class_id)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y

if __name__ == '__main__':
    X, Y = extract_features()
    print(X, Y)