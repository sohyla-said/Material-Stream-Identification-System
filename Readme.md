# Material Stream Identification
The efficient and automated sorting of post-consumer waste is a critical bottleneck in achieving circular economy goals. This project aims to develop an Automated Material Stream Identification (MSI) System using fundamental Machine Learning (ML) techniques. It emphasizes mastery of the entire ML pipeline: Data Preprocessing, Feature Extraction, Classifier Training, and Performance Evaluation.

---
## 📌 Project Goal
The MSI system classifies waste images into the following six categories:

- cardboard  
- glass  
- metal  
- paper  
- plastic  
- trash 

## 📁 Project Structure

```
MSI_project/
│
├── dataset/                  # Original images, organized by class
├── corrupted_images/         # Corrupted images moved from the original dataset
├── dataset_augmented/        # Augmented images, organized by class
│
├── notebooks/                
│   ├── EDA.ipynb             # EDA and preprocessing notebook
│   
├── src/                      # Source code
│   ├── augmentation.py       # Data augmentation pipeline
│   ├── feature_extraction.py # Feature extraction (HOG, LBP, Color Histogram)
│
│
├── requirements.txt          # Python dependencies
├── Readme.md                 # Project documentation
├── requirements.txt          # Project depenedencies
└── .gitignore  
```

---

## 🚀 How to Run

1. **Place the dataset:**  
   Download the dataset zip file and unzip it or copy your waste image dataset and put the `dataset/` folder inside the project root directory.
   You can download the waste image dataset from the following link:  
   [Google Drive Dataset Folder](https://drive.google.com/drive/folders/1QE663fRIempGreUtxr4iLREmZIK4q8L7?usp=sharing)

   

2. **Intsall required dependencies**
   In the terminal execute:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the EDA notebook:**  
   Open the Exploratory Data Analysis (EDA) notebook (`notebooks/EDA.ipynb`) in Jupyter or VS Code and run all cells to explore and visualize the dataset.

4. **Run the augmentation script:**  
   In the terminal, execute:
   ```bash
   python src/augmentation.py
   ```

5. **Run the feature extraction script:**  
   In the terminal, execute:
   ```bash
   python src/feature_extraction.py
   ```

---

## 🎯 Data Augmentation Strategy
Apply data augmentation techniques to the provided dataset to artificially increase the training sample size by a minimum of 30%.
- The augmentations are designed to simulate real-world variations such as lighting, orientation, scale, and viewpoint changes.  
- These conditions commonly occur in recycling facilities and affect model robustness.

* The augmentation pipeline includes:

### ✔ RandomRotation(±20°)
Real objects may be rotated when photographed.
But too much rotation can distort features.
This helps the model generalize across different object orientations.

### ✔ RandomHorizontalFlip()
Many items can appear mirrored depending on how they fall or how the camera captures them.  
This helps model generalize under mirrored scenarios

### ✔ ColorJitter (brightness & contrast)
Lighting conditions vary greatly in real environments.  
This ensures the model remains stable under different brightness or shadow levels.

### ✔ RandomResizedCrop (scale 0.8 → 1.2)
This adds **zoom-in** and **zoom-out** effects while keeping output image size fixed at **384×512**.
Objects may appear closer or farther in real scenes.

- scale < 1.0 → zoom-out (more background)  
- scale > 1.0 → zoom-in (object fills the frame)

This simulates different distances between the camera and the waste object.

### ❗ Avoid:
- Vertical flip → unrealistic orientation for most waste items
- Extreme rotations (>45°) → distort the object
- Excessive color shifting → may change class appearance


---

# Feature Extraction – HOG + Color Histogram + LBP

This section documents the **feature extraction pipeline** used in the Material Stream Identification (MSI) project.  
The goal convert the raw 2D or 3D image data into a 1D numerical feature vector **(a fixed-length list of numbers)**. suitable for classical machine learning models such as **SVM** and **k-NN**.

## what are Feature Descriptors?
A feature descriptor is a mathematical representation of an image (or part of an image) designed to capture important, discriminative visual characteristics such as:
- Edges
- Texture
- Color patterns
- Shape
- Keypoints / structure
Good feature descriptors reduce noise, capture structure, and make the classes easier to separate.

This pipeline uses **three complementary feature descriptors**:

1. **HOG (Histogram of Oriented Gradients)** – shape and edge structure  
2. **Color Histogram (HSV space)** – color and material identity  
3. **LBP (Local Binary Patterns)** – texture patterns for material surfaces  

All three are concatenated into one unified 1D feature vector.

---

## 📌 Why These Feature Descriptors?

### ✔ HOG (Histogram of Oriented Gradients)
HOG captures object **shape**, **outline**, and **edge distribution**, 
✔ Good for outlining object shape
✔ Works very well for recyclable materials
✔ Robust against small rotations

which is useful for distinguishing:
- metal cans vs. plastic bottles  
- cardboard edges vs. paper sheets  
- glass items with sharp contours  

HOG works well for structured objects and is robust to lighting changes.

---

### ✔ Color Histogram (HSV)
Counts how often each color appears.
✔ Good when classes are color-distinguishable (metal, plastic, glass)
✔ Simple + small vector

- plastic items often have saturated colors  
- metal tends to have gray/silver tones  
- glass has specific hue ranges  
- paper and cardboard have beige/brown color patterns  

HSV is chosen instead of RGB because it separates intensity from color information, making it more stable when brightness varies.

---

### ✔ LBP (Local Binary Patterns)
compares each pixel to its neighbors → creates a texture code.
✔ Excellent for surfaces like cardboard, paper, plastic
✔ Robust to lighting
✔ Very fast

- cardboard has strong fiber texture  
- paper is smooth  
- metal is reflective  
- plastic is smoother than cardboard  
- trash is high-variance  

---

## 🔧 How the Pipeline Works

For every image:

1. **Load image** (resized to 256×192)
2. **Extract the following features**:
   - HOG vector  
   - HSV color histogram  
   - LBP texture histogram  
3. **Concatenate all features** into one long numeric vector
4. Store features (`X`) and class labels (`Y`)

```bash
final_feature_vector = [HOG || LBP || ColorHistogram]
```

These vectors are then used to train SVM/k-NN classifiers.

---