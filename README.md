# Neural-Network
# Nepali Festival Image Recognition

A Python implementation using CNN and Transfer Learning (EfficientNetB0) to classify images of 5 Nepali festivals: Chhath, Dashain, Holi, Teej, and Tihar.

## Requirements
- Python 3.6+
- TensorFlow 2.x
- OpenCV
- scikit-learn
- matplotlib

## How It Works

1. **Dataset Preparation**:
   - Images are loaded from `Festival of Nepal/` directory
   - Each festival has its own subfolder with 250 images
   - Images resized to 150x150 pixels

2. **Model Architecture**:
   - Uses pre-trained EfficientNetB0 as base
   - Adds custom layers:
     ```python
     GlobalAveragePooling2D()
     Dense(256, activation='relu')
     Dropout(0.25)
     Dense(5, activation='softmax')
     ```

3. **Training**:
   - Image augmentation (rotation, flipping, zooming)
   - Early stopping to prevent overfitting
   - Learning rate reduction on plateau

4. **Evaluation**:
   - 82% test accuracy
   - Generates confusion matrix and classification report

## How to Run

1. Put your images in `Festival of Nepal/` with subfolders for each festival
2. Run the training:
   ```python
   python train.py
