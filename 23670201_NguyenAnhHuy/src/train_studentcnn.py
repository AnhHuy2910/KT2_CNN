"""
train_studentcnn.py
Huấn luyện StudentCNN trên CIFAR-100 (100 fine classes).

Chạy:
    python src/train_studentcnn.py --epochs 100 --batch-size 128 --lr 1e-3

Yêu cầu đề bài đã đáp ứng:
  - >=100 epochs, CrossEntropyLoss, Adam/AdamW.
  - Log Train Loss/Acc, Val Loss/Acc mỗi epoch -> logs/studentcnn_history.csv
  - Best checkpoint theo Validation Accuracy cao nhất -> checkpoints/studentcnn_best.pth
  - Seed cố định (mặc định 42) cho Python/NumPy/PyTorch.
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import build_student_cnn, MSSV_CONFIG
from dataset import build_dataloaders, report_class_distribution, load_official_cifar100
from utils import set_seed, get_device, HistoryLogger, count_trainable_params, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=str, default="../checkpoints")
    p.add_argument("--log-dir", type=str, default="../logs")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint để resume")
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(mode=train)
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ---- Dữ liệu ----
    official_train, _ = load_official_cifar100(args.data_root)
    counts = report_class_distribution(official_train.targets)
    print("Phân bố mẫu mỗi lớp (official train, 100 lớp): "
          f"min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root=args.data_root,
        model_type="student_cnn",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} "
          f"| Test batches: {len(test_loader)}")

    # ---- Model ----
    model = build_student_cnn(num_classes=100, config=MSSV_CONFIG).to(device)
    total, trainable = count_trainable_params(model)
    print(f"StudentCNN — tổng tham số: {total:,} | trainable: {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_val_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("val_accuracy", 0.0)
        print(f"Resumed từ epoch {ckpt['epoch']}, best_val_acc={best_val_acc:.4f}")

    logger = HistoryLogger(os.path.join(args.log_dir, "studentcnn_history.csv"))
    best_ckpt_path = os.path.join(args.checkpoint_dir, "studentcnn_best.pth")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        logger.log(epoch, "train", train_loss, train_acc, val_loss, val_acc, lr_now)
        dt = time.time() - t0
        print(f"[Epoch {epoch:3d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={lr_now:.6f} | {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "config": MSSV_CONFIG,
            }, best_ckpt_path)
            print(f"  -> Lưu best checkpoint (val_acc={val_acc:.4f}) tại {best_ckpt_path}")

    logger.close()
    save_json({"best_val_accuracy": best_val_acc, "epochs_run": args.epochs},
              os.path.join(args.log_dir, "studentcnn_summary.json"))
    print(f"Hoàn tất huấn luyện StudentCNN. Best val_accuracy = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
