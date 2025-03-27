
# Image Classification Using InceptionResNetV2

## Overview
This project implements an image classification model using the InceptionResNetV2 architecture with transfer learning on TensorFlow and Keras. The model classifies images into 7 distinct categories. The training includes data augmentation, fine-tuning, and comprehensive evaluation metrics such as accuracy, precision, and recall.

## Dataset
The dataset (`data_5.zip`) should be structured as follows after extraction:

```
data_extracted/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

## Requirements
- TensorFlow==2.18.0
- matplotlib==3.10.1
- random2==1.0.2

Install dependencies using:
```bash
pip install tensorflow==2.18.0 random2==1.0.2 matplotlib==3.10.1
```

## Model Architecture
- Base Model: InceptionResNetV2 pretrained on ImageNet (frozen initially)
- Added layers:
  - Global Average Pooling
  - Dense layers (32, 64, 128 neurons respectively with ReLU activation)
  - Dropout layer (0.3)
  - Output Dense layer (7 neurons with softmax activation)

## Data

The dataset used in this project is available for download from the following link:
[dataset](https://www.kaggle.com/datasets/garvpatidar/foodvision-101-subset/data)

## Data Augmentation
Implemented using Keras' `ImageDataGenerator`:
- Rotation: 20 degrees
- Shear: 0.2
- Zoom: 0.2
- Width & Height shifts: 0.2
- Horizontal flip: Enabled
- Rescaling: Applied (1./255)

## Training
- Epochs: 15 (with early stopping callback at 95% accuracy)
- Batch size: 32
- Optimizer: Adam (default learning rate)
- Loss: Categorical Crossentropy
- Metrics: Accuracy, Precision, Recall

## Usage
1. Ensure dataset is in the specified structure.
2. Run all code blocks sequentially in Google Colab or your local environment.
3. Training plots will visualize model performance metrics.
4. Model will automatically save as `model.h5`.

## Evaluation
The training script generates the following visualizations:
- Training vs Validation Accuracy
- Training vs Validation Loss
- Training vs Validation Precision
- Training vs Validation Recall

## Saving and Downloading the Model
The model is saved locally in the `.h5` format:
```python
model.save('model.h5')
```



