
# **Integrated Model for Training and Testing**

This repository contains a PyTorch-based framework for training and testing an integrated model that uses multiple color spaces (RGB, HSV, YCbCr) and advanced attention mechanisms to enhance performance on a binary classification task.

## **Contents**
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Arguments](#arguments)
- [File Structure](#file-structure)

---

## **Features**
- Modular architecture with separate files for models, data loaders, and utilities.
- Support for training, validation, and testing workflows.
- Automatic calculation of evaluation metrics, including Accuracy, Precision, Recall, F1 Score, Equal Error Rate (EER), and BPCER at APCER thresholds.
- Customizable arguments for datasets, model paths, batch size, epochs, and learning rate.

---

## **Requirements**
Install the dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

Dependencies include:
- PyTorch
- Torchvision
- TIMM
- OpenCV
- Pillow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- tqdm

---

## **Setup**
Clone this repository and navigate to its directory:
```bash
git clone <repository_url>
cd <repository_name>
```

Ensure the datasets are prepared with the following structure:
```
/path/to/train
    /live
        img1.jpg
        img2.jpg
    /spoof
        img1.jpg
        img2.jpg
/path/to/val
    /live
        img1.jpg
    /spoof
        img1.jpg
/path/to/test
    /live
        img1.jpg
    /spoof
        img1.jpg
```

---

## **Usage**

### **Training**
Run the `train.py` script with the following command:
```bash
python train.py     --train_path /path/to/train     --val_path /path/to/val     --batch_size 32     --epochs 30     --lr 0.0001     --patience 5     --model_path best_model.pth
```

### **Testing**
Run the `test.py` script to evaluate the model:
```bash
python test.py     --test_path /path/to/test     --batch_size 32     --model_path best_model.pth
```

---

## **Arguments**

### **Training (`train.py`)**
| Argument       | Type    | Default        | Description                                   |
|----------------|---------|----------------|-----------------------------------------------|
| `--train_path` | string  | **Required**   | Path to the training dataset                 |
| `--val_path`   | string  | **Required**   | Path to the validation dataset               |
| `--batch_size` | integer | `32`           | Batch size for training and validation       |
| `--epochs`     | integer | `30`           | Number of training epochs                    |
| `--lr`         | float   | `0.0001`       | Learning rate for the optimizer              |
| `--patience`   | integer | `5`            | Early stopping patience                      |
| `--model_path` | string  | `best_model.pth` | Path to save the trained model weights      |

### **Testing (`test.py`)**
| Argument       | Type    | Default        | Description                                   |
|----------------|---------|----------------|-----------------------------------------------|
| `--test_path`  | string  | **Required**   | Path to the testing dataset                  |
| `--batch_size` | integer | `32`           | Batch size for the testing DataLoader        |
| `--model_path` | string  | **Required**   | Path to the saved model weights              |

---

## **File Structure**
```
├── models/
│   ├── reshape_and_conv.py
│   ├── residual_block.py
│   ├── custom_model.py
│   ├── channel_reducer.py
│   ├── base_model_factory.py
│   ├── integrated_model.py
│   └── integrated_model_wrapper.py
├── utils/
│   ├── data_loaders.py
│   └── color_transform.py
├── train.py
├── test.py
├── args.py
├── requirements.txt
├── README.md
```

---

## **Metrics**
The following metrics are calculated during testing:
1. **Accuracy**: Proportion of correct predictions.
2. **Precision**: True Positives / (True Positives + False Positives).
3. **Recall**: True Positives / (True Positives + False Negatives).
4. **F1 Score**: Harmonic mean of Precision and Recall.
5. **Equal Error Rate (EER)**: Where False Positive Rate = False Negative Rate.
6. **BPCER at APCER thresholds**: Bona fide Presentation Classification Error Rate at 5% and 10% Attack Presentation Classification Error Rate thresholds.

---

