# KIỂM TRA 2
**MSSV:** 23670201  
**Họ và tên:** Nguyễn Anh Huy

---

Bài kiểm tra này nhằm hiện thực và so sánh 2 models học sâu trên một tập dữ liệu ảnh phân loại:
- **ResNet18**: pretrained ResNet-18 model fine-tuned
- **StudentCNN**: tự xây dựng mô hình CNN

## Cấu trúc

```
23670201_NguyenAnhHuy/
├── data/               # Dataset directory
├── src/
│   ├── dataset.py      # Dataset loading and preprocessing
│   ├── model.py        # Model definitions (ResNet18 & StudentCNN)
│   ├── train_resnet18.py       # Training script for ResNet18
│   ├── train_studentcnn.py     # Training script for StudentCNN
│   └── utils.py        # Utility functions
├── checkpoints/        # Saved model weights
├── figures/            # Training plots and figures
├── logs/               # Training logs
└── README.md
```

## Cài đặt
```bash
pip install -r requirements.txt
```

## Training

**Train ResNet18:**
```bash
python src/train_resnet18.py --epochs 10 --batch-size 64 --lr 1e-3
```

**Train StudentCNN:**
```bash
python src/train_studentcnn.py --epochs 10 --batch-size 128 --lr 1e-3
```
