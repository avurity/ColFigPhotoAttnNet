import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
import numpy as np

from models.integrated_model_wrapper import IntegratedModelWrapper
from models.base_model_factory import base_model_factory
from utils.data_loaders import get_data_loaders
from args import get_test_args  # Import the test arguments function


def compute_bpcer_at_apcer(fpr, tpr, thresholds, target_apcer):
    """
    Compute BPCER at a specific APCER threshold.
    
    Args:
        fpr (array): False Positive Rate (APCER) values.
        tpr (array): True Positive Rate values.
        thresholds (array): Decision thresholds corresponding to FPR and TPR.
        target_apcer (float): Target APCER threshold (e.g., 0.05 for 5%).
    
    Returns:
        bpcer (float): BPCER value at the specified APCER.
        threshold (float): Threshold at which the target APCER is achieved.
    """

    target_idx = np.nanargmin(np.abs(fpr - target_apcer))
    threshold_at_apcer = thresholds[target_idx]

    bpcer = 1 - tpr[target_idx]
    return bpcer, threshold_at_apcer


def test_model(args):
    device = torch.device(args.device)

    # Initialize model
    print(f"Loading model from {args.model_path}")
    model = IntegratedModelWrapper(base_model_factory).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()


    test_loader = get_data_loaders(
        None, None, args.test_path, args.batch_size
    )[-1]


    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())
            y_score.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())


    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    eer = np.min(np.mean([fpr, fnr], axis=0)) * 100

    apcer_5 = 0.05  # 5%
    apcer_10 = 0.10  # 10%

    bpcer_5, threshold_5 = compute_bpcer_at_apcer(fpr, tpr, thresholds, apcer_5)
    bpcer_10, threshold_10 = compute_bpcer_at_apcer(fpr, tpr, thresholds, apcer_10)

    # Print metrics
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"EER: {eer:.2f}%")
    print(f"BPCER at APCER=5%: {bpcer_5 * 100:.2f}% (Threshold: {threshold_5:.4f})")
    print(f"BPCER at APCER=10%: {bpcer_10 * 100:.2f}% (Threshold: {threshold_10:.4f})")


if __name__ == "__main__":
    args = get_test_args()  
    test_model(args)
