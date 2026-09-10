import torch
import torch.nn as nn
import torchvision.models as tv_models


# Cấu hình từ file config.json
MSSV_CONFIG= {'a': 0, 'b': 2, 'c': 0, 'd': 1, 
                'N': 3, 'C1': 64, 'kernel_size': 3,
                'activation': 'leaky_relu',
                'use_batchnorm': True,
                'pooling': 'max',
                'dropout_r': 0.3,
                'H': 128,           # Cho ResNet18
                'F': 0              # Fine-tuning
} 

def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Activation không hợp lệ: {name}")


def _get_pool(kind: str) -> nn.Module:
    kind = kind.lower()
    if kind == "max":
        return nn.MaxPool2d(kernel_size=2, stride=2)
    if kind == "avg":
        return nn.AvgPool2d(kernel_size=2, stride=2)
    raise ValueError(f"Pooling không hợp lệ: {kind}")

# Xây dựng 1 block
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 activation: str, use_bn: bool, pool_kind: str):
        super().__init__()
        padding = kernel_size // 2  # giữ nguyên kích thước không gian trước khi pool
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride=1, padding=padding, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(_get_activation(activation))
        layers.append(_get_pool(pool_kind))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# Xây dựng mô hình Student CNN theo cấu hình từ config.json
class StudentCNN(nn.Module):
    def __init__(self, num_classes: int = 100, config: dict = None,
                 weight_init: bool = True):
        super().__init__()
        cfg = config or MSSV_CONFIG
        N = cfg["N"]
        C1 = cfg["C1"]
        kernel_size = cfg["kernel_size"]
        activation = cfg["activation"]
        use_bn = cfg["use_batchnorm"]
        pool_kind = cfg["pooling"]
        dropout_r = cfg["dropout_r"]

        # Số kênh mỗi block: C_i = min(C1 * 2^(i-1), 256)
        channels = [min(C1 * (2 ** i), 256) for i in range(N)]
        self.channels = channels

        blocks = []
        in_ch = 3
        for out_ch in channels:
            blocks.append(
                ConvBlock(in_ch, out_ch, kernel_size, activation, use_bn, pool_kind)
            )
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        # Classifier bắt buộc: AdaptiveAvgPool2d(1) -> Flatten -> Dropout(r) -> Linear(CN, 100)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_r),
            nn.Linear(channels[-1], num_classes),
        )

        if weight_init:
            self._init_weights()

    def _init_weights(self):
        """Weight initialization hợp lý (Kaiming cho conv, thường cho linear)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ResNet18 Transfer Learning

class ResNet18Classifier(nn.Module):
    """
    ResNet18 pretrained ImageNet + classifier head tùy chỉnh theo MSSV.

        Linear(D, H) -> Activation -> Dropout(r) -> Linear(H, 100)

    D = số đặc trưng đầu ra của ResNet18 trước fc gốc (= 512 cho ResNet18).
    H = 128 * 2^(b mod 2)
    Activation, Dropout(r) dùng chung giá trị đã xác định cho StudentCNN.
    """

    def __init__(self, num_classes: int = 100, H: int = 128,
                 activation: str = "leaky_relu", dropout_r: float = 0.3,
                 pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        D = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()  # bỏ fc gốc, tự thêm classifier riêng
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Linear(D, H),
            _get_activation(activation),
            nn.Dropout(p=dropout_r),
            nn.Linear(H, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)

    
    # Freeze / unfreeze theo 2 giai đoạn huấn luyện
    def freeze_backbone(self):
        """Stage 1 (feature extraction): freeze toàn bộ backbone, chỉ train classifier."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def set_finetune_stage(self, F: int):
        """
        Stage 2 (fine-tuning) theo F = (a+c) mod 3:

            F=0: train layer4 + classifier;         freeze conv1,bn1,layer1-layer3
            F=1: train layer3+layer4 + classifier;  freeze conv1,bn1,layer1-layer2
            F=2: train layer2+layer3+layer4 + classifier; freeze conv1,bn1,layer1
        """
        # Mặc định freeze tất cả trước
        for p in self.backbone.parameters():
            p.requires_grad = False

        always_trainable = ["layer4"]
        if F == 1:
            always_trainable = ["layer3", "layer4"]
        elif F == 2:
            always_trainable = ["layer2", "layer3", "layer4"]
        elif F == 0:
            always_trainable = ["layer4"]
        else:
            raise ValueError(f"F không hợp lệ: {F}")

        for name in always_trainable:
            layer = getattr(self.backbone, name)
            for p in layer.parameters():
                p.requires_grad = True

        for p in self.classifier.parameters():
            p.requires_grad = True

        return always_trainable

    def get_param_groups(self, lr_backbone: float, lr_classifier: float):
        """Trả về param groups cho optimizer: LR nhỏ hơn cho backbone khi fine-tune."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        classifier_params = [p for p in self.classifier.parameters() if p.requires_grad]
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        groups.append({"params": classifier_params, "lr": lr_classifier})
        return groups


def build_student_cnn(num_classes=100, config=None):
    return StudentCNN(num_classes=num_classes, config=config)


def build_resnet18(num_classes=100, config=None, pretrained=True):
    cfg = config or MSSV_CONFIG
    return ResNet18Classifier(
        num_classes=num_classes,
        H=cfg["H"],
        activation=cfg["activation"],
        dropout_r=cfg["dropout_r"],
        pretrained=pretrained,
    )


# --------------------------------------------------------------------------- #
# Kiểm thử nhanh kiến trúc (sanity check) — chạy: python src/model.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from utils import count_trainable_params

    print("=== StudentCNN ===")
    model = build_student_cnn()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print("Output shape:", out.shape)
    total, trainable = count_trainable_params(model)
    print(f"Tổng tham số: {total:,} | Trainable: {trainable:,}")
    print(model)

    print("\n=== ResNet18 (Stage 1 - feature extraction) ===")
    r18 = build_resnet18(pretrained=False)  # pretrained=False chỉ để test kiến trúc offline
    r18.freeze_backbone()
    x2 = torch.randn(2, 3, 224, 224)
    out2 = r18(x2)
    print("Output shape:", out2.shape)
    total2, trainable2 = count_trainable_params(r18)
    print(f"Tổng tham số: {total2:,} | Trainable (Stage 1): {trainable2:,}")

    print("\n=== ResNet18 (Stage 2 - fine-tuning, F=0) ===")
    trainable_layers = r18.set_finetune_stage(MSSV_CONFIG["F"])
    print("Layers trainable ở Stage 2:", trainable_layers, "+ classifier")
    total3, trainable3 = count_trainable_params(r18)
    print(f"Tổng tham số: {total3:,} | Trainable (Stage 2): {trainable3:,}")
