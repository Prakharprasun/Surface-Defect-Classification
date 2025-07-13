# Surface Defect Classification with EfficientNetB0

This repository contains a deep learning pipeline for classifying steel surface defects from the NEU Surface Defect Database (NEU-DET) into six categories using transfer learning with EfficientNetB0. The implementation prioritizes modularity, reproducibility, and robustness, incorporating data preprocessing, custom augmentations, dynamic fine-tuning, and inference utilities.

## Dataset

The NEU-DET dataset is organized into `train/` and `validation/` directories, each containing images for six defect classes: Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, and Scratches.

### Dataset Processing

The pipeline programmatically merges the `train/` and `validation/` directories into a unified `merged/` directory. A stratified 80/20 train-validation split is applied using `ImageDataGenerator` to ensure balanced class representation across splits.

## Model Architecture

| Feature              | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| Architecture         | EfficientNetB0, pretrained on ImageNet                                  |
| Fine-tuning          | Last ~150 layers unfrozen for task-specific adaptation                  |
| Input Size           | 300 × 300 pixels                                                       |
| Loss Function        | Categorical Crossentropy                                               |
| Optimizer            | Adam                                                                   |
| Metrics              | Accuracy, AUC, Precision, Recall                                        |
| Regularization       | Dropout (0.5, 0.3), EarlyStopping, ReduceLROnPlateau                   |
| Data Augmentation    | Horizontal flip, rotation, zoom, width/height shift                    |

## Dependencies

- `tensorflow`
- `numpy`
- `matplotlib`
- `scikit-learn`

The pipeline is designed for execution on Google Colab without additional configuration. Google Drive mounting is supported for dataset access:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Usage

1. Ensure the NEU-DET dataset is available in the specified directory structure.
2. Mount Google Drive in Colab if required.
3. Execute the `Surface Defects.ipynb` notebook sequentially.

### Workflow

- Merges `train/` and `validation/` into `merged/`.
- Applies a stratified 80/20 train-validation split.
- Trains the EfficientNetB0 model using transfer learning.
- Evaluates performance on the validation set.
- Provides a `predict_defect()` function for single-image inference.

## Inference

The `predict_defect()` function supports inference on individual test images:

```python
predict_defect('/path/to/test_image.jpg')
```

This outputs the input image, the predicted defect class, and confidence scores for all six classes.

## Performance

| Metric             | Training  | Validation |
|--------------------|-----------|------------|
| Loss               | 0.0298    | 0.0223     |
| Accuracy           | 0.9947    | 0.9945     |
| AUC                | 0.5972    | 1.0000     |
| Precision          | 0.9947    | 0.9945     |
| Recall             | 0.9896    | 0.9918     |

### Analysis

The model achieves high validation accuracy (99.45%) and a perfect validation AUC (1.0000), indicating strong generalization. Low training and validation losses reflect effective convergence. Class imbalance and domain shift between original splits were mitigated through dataset merging, class-weighted training, and data augmentations.

## Utilities

- **EarlyStopping**: Restores best weights if validation loss does not improve.
- **ReduceLROnPlateau**: Reduces learning rate upon validation loss stagnation.
- **Visual Predictions**: Provides streamlined visualization of model outputs.
- **Class Mapping**: Ensures consistent alignment of categorical labels.

## File Structure

| File/Folder            | Purpose                                  |
|------------------------|------------------------------------------|
| `Surface Defects.ipynb`| Main pipeline notebook                  |
| `train/`, `validation/`| Original dataset directories            |
| `merged/`              | Auto-generated merged dataset           |
| `predict_defect()`     | Utility for single-image inference      |

