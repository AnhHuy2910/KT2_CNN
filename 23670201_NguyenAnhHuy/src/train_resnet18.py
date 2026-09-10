"""
train_resnet18.py
Transfer Learning ResNet18 (pretrained ImageNet) trên CIFAR-100 — 2 giai đoạn:

  Stage 1 (Feature extraction): 20 epochs, freeze toàn bộ backbone, train classifier.
  Stage 2 (Fine-tuning):        >=80 epochs, unfreeze theo F=(a+c) mod 3 = 0
                                 -> train layer4 + classifier,
                                    freeze conv1, bn1, layer1, layer2, layer3.

Tổng epochs >= 100. Log liên tục cho cả 2 stage vào cùng 1 file history CSV,
cột 'stage' phân biệt 'feature_extraction' / 'fine_tuning' (theo Phụ lục B).
Best checkpoint cuối cùng chọn theo Validation Accuracy cao nhất qua cả 2 stage.

Chạy:
    python src/train_resnet18.py --stage1-epochs 20 --stage2-epochs 80
"""
import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import build_resnet18, MSSV_CONFIG
from dataset import build_dataloaders
from utils import set_seed, get_device, HistoryLogger, count_trainable_params, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--stage1-epochs", type=int, default=20)
    p.add_argument("--stage2-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--stage1-lr", type=float, default=1e-3)
    p.add_argument("--stage2-lr-backbone", type=float, default=5e-5)
    p.add_argument("--stage2-lr-classifier", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=str, default="../checkpoints")
    p.add_argument("--log-dir", type=str, default="../logs")
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

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root=args.data_root,
        model_type="resnet18",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} "
          f"| Test batches: {len(test_loader)}")

    model = build_resnet18(num_classes=100, config=MSSV_CONFIG, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    logger = HistoryLogger(os.path.join(args.log_dir, "resnet18_history.csv"))
    best_ckpt_path = os.path.join(args.checkpoint_dir, "resnet18_best.pth")
    best_val_acc = 0.0
    global_epoch = 0

    # ------------------------------------------------------------------ #
    # Stage 1: Feature extraction — freeze backbone, train classifier only
    # ------------------------------------------------------------------ #
    print("\n=== Stage 1: Feature extraction (freeze backbone) ===")
    model.freeze_backbone()
    total, trainable = count_trainable_params(model)
    print(f"Tổng tham số: {total:,} | Trainable (Stage 1): {trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.stage1_lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.stage1_epochs)

    for epoch in range(1, args.stage1_epochs + 1):
        global_epoch += 1
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        logger.log(global_epoch, "feature_extraction", train_loss, train_acc, val_loss, val_acc, lr_now)
        print(f"[Stage1 Epoch {epoch:3d}/{args.stage1_epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {time.time()-t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": global_epoch, "stage": "feature_extraction",
                "model_state_dict": model.state_dict(), "val_accuracy": val_acc,
                "config": MSSV_CONFIG,
            }, best_ckpt_path)
            print(f"  -> Lưu best checkpoint (val_acc={val_acc:.4f})")

    # ------------------------------------------------------------------ #
    # Stage 2: Fine-tuning theo F = (a+c) mod 3 = 0 -> layer4 + classifier
    # ------------------------------------------------------------------ #
    print(f"\n=== Stage 2: Fine-tuning (F={MSSV_CONFIG['F']}) ===")
    trainable_layers = model.set_finetune_stage(MSSV_CONFIG["F"])
    print("Layers trainable ở Stage 2:", trainable_layers, "+ classifier")
    total2, trainable2 = count_trainable_params(model)
    print(f"Tổng tham số: {total2:,} | Trainable (Stage 2): {trainable2:,}")

    param_groups = model.get_param_groups(args.stage2_lr_backbone, args.stage2_lr_classifier)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.stage2_epochs)

    stage2_start_marker = global_epoch  # dùng để vẽ đường phân cách Stage1->Stage2 trên learning curve

    for epoch in range(1, args.stage2_epochs + 1):
        global_epoch += 1
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        lr_now = optimizer.param_groups[-1]["lr"]
        logger.log(global_epoch, "fine_tuning", train_loss, train_acc, val_loss, val_acc, lr_now)
        print(f"[Stage2 Epoch {epoch:3d}/{args.stage2_epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {time.time()-t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": global_epoch, "stage": "fine_tuning",
                "model_state_dict": model.state_dict(), "val_accuracy": val_acc,
                "config": MSSV_CONFIG,
            }, best_ckpt_path)
            print(f"  -> Lưu best checkpoint (val_acc={val_acc:.4f})")

    logger.close()
    save_json({
        "best_val_accuracy": best_val_acc,
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "stage2_start_global_epoch": stage2_start_marker + 1,
    }, os.path.join(args.log_dir, "resnet18_summary.json"))
    print(f"Hoàn tất huấn luyện ResNet18. Best val_accuracy = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
