# Lung Segmentation with DeepLabV3+

This project focuses on segmenting lung regions from chest X-ray images using the DeepLabV3+ semantic segmentation model. The model is trained and evaluated on two widely used medical datasets—Shenzhen and Montgomery—achieving high accuracy and robustness across variations in image sources.

---

## Problem Statement

Accurate lung segmentation from chest X-rays is essential for downstream tasks like pulmonary disease detection, anomaly analysis, and automated diagnostics. However, generalization across datasets with varying image quality and intensity distributions is challenging. This project addresses that challenge through preprocessing normalization and a deep learning approach using DeepLabV3+.

---

## Objective

- Train a deep neural network for precise lung segmentation from X-ray images.
- Ensure cross-dataset generalization between Shenzhen and Montgomery datasets.
- Maximize Intersection over Union (IoU) while maintaining alignment of images and masks.

---

## Datasets

- **Shenzhen Chest X-ray Set** (China)
- **Montgomery County Chest X-ray Set** (USA)

> Total images used: 704 (with corresponding binary lung masks)

---

## Preprocessing Pipeline

- Resize and center-crop images and masks to 512x512 resolution
- Histogram normalization to reduce contrast/brightness variation
- Binary mask alignment validation
- Normalization to [0, 1] range for input images

---

## Model Architecture

- **Model**: DeepLabV3+ (ResNet-50 backbone, pretrained on ImageNet)
- **Loss Function**: Binary Cross-Entropy + Dice Loss
- **Optimizer**: Adam
- **Input Size**: 512x512
- **Batch Size**: 8
- **Epochs**: 50

---

## Evaluation Metrics

| Metric              | Score  |
|---------------------|--------|
| Intersection over Union (IoU) | **92%**   |
| Dice Coefficient    | **94%**   |
| Pixel Accuracy      | **97%**   |

---

## Tools & Technologies

- Python 3.9+
- PyTorch
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

---

## Key Achievements

- Successfully trained a DeepLabV3+ model to segment lungs with **92% IoU**
- Handled domain shift between datasets through effective preprocessing
- Developed a reusable and modular segmentation pipeline

---

## Future Improvements

- Add post-processing (CRF, morphological operations) to improve boundary sharpness
- Experiment with lighter backbones (e.g., MobileNet) for edge deployment
- Extend model to multi-class segmentation for detecting anomalies

---


