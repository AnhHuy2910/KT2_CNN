"""
dataset.py
Chuẩn bị CIFAR-100: tải dữ liệu, stratified split train/validation (45.000/5.000,
seed=42), Dataset/DataLoader cho StudentCNN (32x32) và ResNet18 (224x224).

Quy định đề bài:
  - 100 fine classes (KHÔNG dùng 20 coarse classes).
  - Train: 45.000 ảnh / Validation: 5.000 ảnh (stratified theo nhãn, seed=42)
    lấy từ official training set (50.000 ảnh).
  - Test: 10.000 ảnh official test set — chỉ dùng để đánh giá cuối cùng.
  - Cùng một split phải dùng cho cả StudentCNN và ResNet18.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision
import torchvision.transforms as T

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLIT_SEED = 42
VAL_SIZE = 5000  # trong tổng 50.000 ảnh official train


def get_stratified_split_indices(targets, val_size=VAL_SIZE, seed=SPLIT_SEED):
    """
    Trả về (train_idx, val_idx) stratified theo nhãn từ official training set.
    Dùng chung cho StudentCNN và ResNet18 -> gọi hàm này 1 lần, lưu index,
    rồi tái sử dụng lại (đảm bảo đúng yêu cầu "cùng một split").
    """
    targets = np.asarray(targets)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))
    return train_idx, val_idx


class TransformedSubset(Dataset):
    """Bọc một Subset của CIFAR100 (PIL Image, label) với transform tùy chỉnh."""

    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base_dataset[self.indices[i]]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def get_student_cnn_transforms():
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    return train_tf, eval_tf


def get_resnet18_transforms(image_size=224):
    """
    Resize ảnh CIFAR-100 (32x32) lên kích thước phù hợp ResNet18 pretrained.
    Train: resize lên (image_size + 32) rồi RandomCrop(image_size) + Flip — augmentation
           thực sự có ý nghĩa (khác với chỉ Resize thẳng 224x224, vốn không tạo ra crop khác nhau).
    Val/Test: resize thẳng về (image_size, image_size), không augmentation ngẫu nhiên.
    """
    resize_for_crop = image_size + 32
    train_tf = T.Compose([
        T.Resize((resize_for_crop, resize_for_crop)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


# --------------------------------------------------------------------------- #
# Dataset builders
# --------------------------------------------------------------------------- #
def load_official_cifar100(data_root="./data"):
    """Tải CIFAR-100 official train (50k) và test (10k) set, không transform (raw PIL)."""
    official_train = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=None
    )
    official_test = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=None
    )
    return official_train, official_test


def build_datasets(data_root="./data", model_type="student_cnn", image_size=224):
    """
    model_type: "student_cnn" hoặc "resnet18"
    Trả về train_ds, val_ds, test_ds (đã áp transform tương ứng) và train/val indices
    (để lưu lại / kiểm tra tính nhất quán split).
    """
    assert model_type in ("student_cnn", "resnet18")

    official_train, official_test = load_official_cifar100(data_root)
    targets = official_train.targets  # list 50000 nhãn fine (0..99)

    train_idx, val_idx = get_stratified_split_indices(targets, VAL_SIZE, SPLIT_SEED)

    if model_type == "student_cnn":
        train_tf, eval_tf = get_student_cnn_transforms()
    else:
        train_tf, eval_tf = get_resnet18_transforms(image_size)

    train_ds = TransformedSubset(official_train, train_idx, train_tf)
    val_ds = TransformedSubset(official_train, val_idx, eval_tf)
    # Test set: KHÔNG augmentation, không dùng để chọn model — chỉ đánh giá cuối cùng.
    test_ds = TransformedSubset(
        official_test, np.arange(len(official_test)), eval_tf
    )

    return train_ds, val_ds, test_ds, train_idx, val_idx


def build_dataloaders(data_root="./data", model_type="student_cnn",
                       batch_size=128, num_workers=4, image_size=224):
    train_ds, val_ds, test_ds, train_idx, val_idx = build_datasets(
        data_root, model_type, image_size
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def report_class_distribution(targets, num_classes=100):
    """Báo cáo phân bố số lượng mẫu mỗi lớp (yêu cầu mục 2 của đề)."""
    targets = np.asarray(targets)
    counts = np.bincount(targets, minlength=num_classes)
    return counts


if __name__ == "__main__":
    # Sanity check nhanh (cần mạng để tải CIFAR-100 lần đầu)
    train_loader, val_loader, test_loader = build_dataloaders(
        model_type="student_cnn", batch_size=64, num_workers=0
    )
    print("Số batch train:", len(train_loader))
    print("Số batch val:", len(val_loader))
    print("Số batch test:", len(test_loader))
    xb, yb = next(iter(train_loader))
    print("Batch shape:", xb.shape, "Labels shape:", yb.shape)
    print("Pixel value range:", xb.min().item(), xb.max().item())
