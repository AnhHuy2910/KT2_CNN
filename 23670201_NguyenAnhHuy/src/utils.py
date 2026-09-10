"""
utils.py
Các hàm tiện ích dùng chung: set seed, tính metrics, ghi log CSV, vẽ learning curves.

MSSV: 23670201
"""
import os
import csv
import json
import random
import numpy as np
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    top_k_accuracy_score,
)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42):
    """Đặt seed cho Python random, NumPy và PyTorch để đảm bảo khả năng tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Các dòng dưới đánh đổi một chút tốc độ để tăng tính tái lập.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# CSV history logger — history.csv phải có dữ liệu theo từng epoch
# --------------------------------------------------------------------------- #
class HistoryLogger:
    """Ghi log train/val loss & accuracy theo từng epoch ra file CSV.

    Cột tối thiểu theo đề bài (Phụ lục B):
    epoch, stage, train_loss, train_accuracy, val_loss, val_accuracy
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._file = open(csv_path, mode="w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["epoch", "stage", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr"]
        )
        self._file.flush()

    def log(self, epoch, stage, train_loss, train_acc, val_loss, val_acc, lr):
        self._writer.writerow([epoch, stage, train_loss, train_acc, val_loss, val_acc, lr])
        self._file.flush()

    def close(self):
        self._file.close()


# --------------------------------------------------------------------------- #
# Metrics tính trên test set (mục 8 của đề)
# --------------------------------------------------------------------------- #
def compute_test_metrics(y_true, y_pred, y_prob=None, num_classes=100):
    """
    y_true, y_pred: 1D array-like các nhãn (int)
    y_prob: 2D array (N, num_classes) xác suất softmax — cần cho Top-5 accuracy
    Trả về dict metrics: accuracy, macro_precision, macro_recall, macro_f1,
    top5_accuracy (nếu có y_prob), confusion_matrix (numpy array), S (chỉ số tổng hợp).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    top5 = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            top5 = top_k_accuracy_score(y_true, y_prob, k=5, labels=list(range(num_classes)))
        except Exception:
            top5 = None

    # S = (Accuracy + Macro F1) / 2, cả hai biểu diễn theo %
    S = (acc * 100 + macro_f1 * 100) / 2.0

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "top5_accuracy": top5,
        "confusion_matrix": cm,
        "S": S,
    }


def performance_score_student_cnn(S_percent: float) -> float:
    """Bảng điểm performance StudentCNN (mục 14.1), S theo %."""
    if S_percent >= 50:
        return 1.00
    if S_percent >= 45:
        return 0.90
    if S_percent >= 40:
        return 0.80
    if S_percent >= 35:
        return 0.70
    if S_percent >= 30:
        return 0.60
    if S_percent >= 25:
        return 0.50
    return 0.30


def performance_score_resnet18(S_percent: float) -> float:
    """Bảng điểm performance ResNet18 (mục 14.2), S theo %."""
    if S_percent >= 75:
        return 1.25
    if S_percent >= 70:
        return 1.15
    if S_percent >= 65:
        return 1.05
    if S_percent >= 60:
        return 0.95
    if S_percent >= 55:
        return 0.85
    if S_percent >= 50:
        return 0.75
    if S_percent >= 40:
        return 0.60
    return 0.40


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
