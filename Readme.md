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
├── dataset_split/            # Dataset splitted into train and test folders, Augmented train images, organized by class
├── models/                   # Extracted Features and class labels after performing feature extraction
│
├── notebooks/                
│   ├── EDA.ipynb             # EDA and preprocessing notebook
│   ├── Models.ipynb          # Models training notebook
│   
├── src/                      # Source code
│   ├── data_augmentation.py  # Data augmentation pipeline
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
   python src/data_augmentation.py
   ```

5. **Run the feature extraction script:**  
   In the terminal, execute:
   ```bash
   python src/feature_extraction.py
   ```
6. **Run the Models notebook:**  
   Open the Modles notebook (`notebooks/Models.ipynb`) in Jupyter or VS Code and run all cells to extract feature vectors and train SVM and KNN models.

---

## 🎯 Data Augmentation Strategy
Apply data augmentation techniques to the provided dataset to artificially increase the training sample size by a minimum of 30%.
- The augmentations are designed to simulate real-world variations such as lighting, orientation, scale, and viewpoint changes.  
- These conditions commonly occur in recycling facilities and affect model robustness.

* The augmentation pipeline includes:

### ✔ Resize((224, 224))
All images (original, augemented) must be converted into **fixed resolution** before feature extraction. SVMs and KNNs requires **fixed length vectors**, so we should standardize image dimensions.
Ensures consistent pixel shape for all images.
Why **224*224**?
- These dimensions are widely used in computer visionand balances detail vs computation.
- Smaller size may lose texture information needed to differentiate between materials.
- Larger size or keeping the same image size (512*384) increases feature-vector dimensionality unnecessarily, slowing down SVM/KNN.


### ✔ RandomRotation(±10°)
Randomly rotate the image by a degree between -10 and 10.
Real objects may be rotated when photographed.
But too much rotation can distort features.
This helps the model generalize across different object orientations.

### ✔ RandomHorizontalFlip()
Randomly flip an image left <->  right with 50% (default) probability
Many items can appear mirrored depending on how they fall or how the camera captures them.  
This helps model generalize under mirrored scenarios.

### ✔ ColorJitter (brightness & contrast)
Randomly increase or decrease brightness/contrast by up to +-20%.
Lighting conditions vary greatly in real environments.  
This ensures the model remains stable under different brightness or shadow levels.
why **0.2**?
This values creates noticeable but **not extreme** lighting changes.



### ❗ Avoid:
- Vertical flip → unrealistic orientation for most waste items
- Extreme rotations (>45°) → distort the object
- Excessive color shifting → may change class appearance


---

# Feature Extraction – CNN (ResNet-18)

This section documents the **feature extraction pipeline** used in the Material Stream Identification (MSI) project.  
The goal convert the raw 2D or 3D image data into a 1D numerical feature vector **(a fixed-length list of numbers)**. suitable for classical machine learning models such as **SVM** and **k-NN**.

## Why Feature Extraction using CNN?
* CNNs automatically learn rich and discriminative features
Traditional handcrafted features like:
- HOG
- LBP
- Color Histograms
work well for simple patterns, but they struggle with complex real-world waste images where:
- lighting varies
- object shapes deform
- textures overlap
- backgrounds differ

A Convolutional Neural Network (CNN) automatically learns multi-level feature representations:
- Early Layers: edges, corners	
- Middle Layers: textures, curves	
- Deep Layers: object parts, high-level patterns

This hierarchical learning makes CNN features far more powerful and robust than manual descriptors.

## Why ResNet-18?
ResNet-18 is widely used for feature extraction because:
✔ It is not too large (fast, lightweight, low GPU requirements)
✔ It is deep enough to capture meaningful semantic information
✔ It has residual connections, making training more stable
✔ It is pretrained on ImageNet, a massive dataset of 1.2M images

This pretraining allows ResNet-18 to learn:
- general edges
- materials
- object categories
- textures
- color representations

These general features transfer extremely well to the **MSI waste classification** problem, even with a relatively small dataset.

It provides fixed-length, compact, **512-dimensional** feature vectors, which is a requirement

---