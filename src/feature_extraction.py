# import cv2
# import numpy as np
# from skimage.feature import hog, local_binary_pattern
# import os



# CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# DIRECTORY = os.path.join(PROJECT_ROOT, "dataset_split")
# # print("Current working directory:", os.getcwd())
# # print("PROJECT_ROOT:", PROJECT_ROOT)
# # print("Dataset directory:", DIRECTORY)
# # folder = r"D:\Fourth_year\ML\Material-Stream-Identification-System\Material-Stream-Identification-System\dataset_augmented\cardboard"
# # print(os.path.exists(folder))

# def load_image(path, size=(256, 192)):
#     img = cv2.imread(path)
#     img = cv2.resize(img, size)
#     return img

# def extract_hog(img, pixels_per_cell = (16, 16), cells_per_block = (2, 2), orientations = 9):
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     hog_feature = hog(gray_img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
#                       block_norm='L2-Hys', feature_vector=True)
#     return hog_feature

# def extract_color_hist(img, bins=(16, 16, 8)):
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hist = cv2.calcHist([hsv_img], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
#     hist = cv2.normalize(hist, hist).flatten()
#     return hist

# def extract_texture(img, P=8, R=1):
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     lbp = local_binary_pattern(gray_img, P, R, method='uniform')
#     n_bins = P + 2
#     hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
#     hist = hist.astype("float")
#     hist /= (hist.sum() + 1e-7)
#     return hist

# def concatenate_features(path):
#     img = load_image(path)
#     hog_features = extract_hog(img)
#     color_features = extract_color_hist(img)
#     texture_features = extract_texture(img)
#     features = np.concatenate([hog_features, color_features, texture_features])
#     return features


# def extract_features(dataset_dir=None):
#     if dataset_dir is None:
#         dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dataset_augmented")
#         dataset_dir = os.path.abspath(dataset_dir)
    
#     X = []
#     Y = []

#     for class_id, class_name in enumerate(CLASSES):
#         folder = os.path.join(dataset_dir, class_name)

#         for filename in os.listdir(folder):
#             img_path = os.path.join(folder, filename)
#             features = concatenate_features(img_path)
#             X.append(features)
#             Y.append(class_id)

#     X = np.array(X)
#     Y = np.array(Y)
#     return X, Y

# if __name__ == '__main__':
#     X, Y = extract_features()
#     print(X, Y)


import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import skew
import os

CLASSES = ['cardboard','glass','metal','paper','plastic','trash']
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIRECTORY = os.path.join(PROJECT_ROOT, "dataset_split")

def load_image(path, size=(256,256)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    return img

# -------------------------
# Strong HOG
# -------------------------
def extract_hog(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return hog(
        gray,
        orientations=12,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm='L2-Hys',
        feature_vector=True
    )

# -------------------------
# Color Histogram (HSV)
# -------------------------
def extract_color_hist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,(8,8,4),[0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# -------------------------
# Improved LBP
# -------------------------
def extract_texture(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=16, R=2, method='uniform')
    hist, _ = np.histogram(lbp, bins=18, range=(0,18))
    hist = hist.astype(float) / (hist.sum()+1e-7)
    return hist

# -------------------------
# GLCM texture
# -------------------------
def extract_glcm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1,2], [0, np.pi/4], 256, symmetric=True, normed=True)
    feats = [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'dissimilarity').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'ASM').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean()
    ]
    return np.array(feats)

# -------------------------
# Color Moments (RGB)
# -------------------------
def extract_color_moments(img):
    chans = cv2.split(img)
    feats = []
    for ch in chans:
        flat = ch.reshape(-1)
        feats += [flat.mean(), flat.std(), skew(flat)]
    return np.array(feats)

# -------------------------
# Normalization Helpers
# -------------------------
def norm(v): return (v - v.mean()) / (v.std() + 1e-7)
def unit(v): return v / (np.linalg.norm(v) + 1e-7)

# -------------------------
# Combined Features
# -------------------------
def concatenate_features(path):
    img = load_image(path)

    hog_f   = norm(extract_hog(img))
    col_f   = unit(extract_color_hist(img))
    lbp_f   = unit(extract_texture(img))
    glcm_f  = norm(extract_glcm(img))
    cm_f    = norm(extract_color_moments(img))

    return np.concatenate([hog_f, col_f, lbp_f, glcm_f, cm_f])

# -------------------------
# Dataset Extraction
# -------------------------
def extract_features(split_name, dataset_dir=None):
    X, Y = [],  []
    if dataset_dir is None:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset_split", split_name)

    for class_id, cls in enumerate(CLASSES):
        folder = os.path.join(dataset_dir, cls)
        for f in os.listdir(folder):
            feats = concatenate_features(os.path.join(folder,f))
            X.append(feats)
            Y.append(class_id)
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    X_train, Y_train = extract_features(split_name="train")
    X_test, Y_test = extract_features(split_name="test")
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
