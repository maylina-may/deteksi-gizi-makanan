"""
Training pipeline untuk deteksi makanan dengan YOLO
- Menggunakan YOLO11s (small) untuk akurasi lebih baik dari nano
- Image size 640x640 untuk detail lebih baik
- Augmentasi data untuk handle class imbalance
- Early stopping dan learning rate scheduling
"""

from ultralytics import YOLO
import torch

# ========== Konfigurasi ==========
DATA_YAML = "dataset/data.yaml"
MODEL_NAME = "yolo11s.pt"  # Small model - lebih akurat dari nano
EPOCHS = 150
IMGSZ = 640  # Lebih besar untuk detail tekstur
BATCH = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Image size: {IMGSZ}")
print(f"Epochs: {EPOCHS}")

# ========== Load Model ==========
model = YOLO(MODEL_NAME)

# ========== Training dengan Augmentasi & Class Weights ==========
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    
    # Class imbalance handling
    # Normalnya YOLO handle dengan focal loss, 
    # tapi kita kasih augmentasi lebih untuk kelas minoritas
    
    # Augmentasi agresif untuk meningkatkan generalisasi
    hsv_h=0.015,        # Hue augmentation
    hsv_s=0.7,          # Saturation augmentation
    hsv_v=0.4,          # Value augmentation
    degrees=15.0,       # Rotation ±15 derajat
    translate=0.2,      # Translation 20%
    scale=0.5,          # Scaling ±50%
    shear=5.0,          # Shear ±5 derajat
    perspective=0.0005, # Perspective transform
    flipud=0.3,         # Flip vertical 30%
    fliplr=0.5,         # Flip horizontal 50%
    mosaic=1.0,         # Mosaic augmentation
    mixup=0.2,          # Mixup augmentation 20%
    copy_paste=0.3,     # Copy-paste augmentation 30%
    erasing=0.4,        # Random erasing 40%
    auto_augment="randaugment",  # Auto augment
    
    # Training parameters
    lr0=0.001,          # Learning rate awal lebih kecil untuk fine-tuning
    lrf=0.01,           # Final learning rate factor
    momentum=0.937,     # SGD momentum
    weight_decay=0.0005, # Weight decay
    
    # Loss weights (default sudah cukup baik)
    box=7.5,            # Box loss weight
    cls=0.5,            # Class loss weight
    dfl=1.5,            # DFL loss weight
    
    # Early stopping
    patience=30,        # Stop jika tidak ada improvement 30 epoch
    save=True,
    save_period=10,     # Save checkpoint setiap 10 epoch
    plots=True,         # Generate plots
    
    # Validation
    val=True,
    split="val",
    
    # Cache for faster training
    cache=False,
    workers=8,
    
    # Optimizer
    optimizer="auto",
    warmup_epochs=5.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # NMS parameters
    close_mosaic=15,    # Disable mosaic in last 15 epochs
    cos_lr=True,        # Cosine learning rate scheduler
    
    # Other
    deterministic=True,
    seed=42,
    verbose=True,
)

print("=" * 50)
print("TRAINING COMPLETE!")
print(f"Best model saved to: {results.save_dir}/weights/best.pt")
print("=" * 50)

# ========== Validasi Model Terbaik ==========
print("\nMemvalidasi model terbaik...")
best_model = YOLO(f"{results.save_dir}/weights/best.pt")

metrics = best_model.val(
    data=DATA_YAML,
    split="val",
    imgsz=IMGSZ,
    batch=16,
    conf=0.001,
    iou=0.65,
    plots=True,
    save_json=True,
)

print(f"\nValidation Results:")
print(f"  mAP50: {metrics.box.map50:.4f}")
print(f"  mAP50-95: {metrics.box.map:.4f}")
print(f"  Precision: {metrics.box.mp:.4f}")
print(f"  Recall: {metrics.box.mr:.4f}")

# Per-class metrics
if hasattr(metrics.box, 'ap_class_index'):
    print(f"\nPer-class mAP50:")
    for i, idx in enumerate(metrics.box.ap_class_index):
        class_name = model.names[idx]
        ap50 = metrics.box.ap50[i]
        print(f"  {class_name}: {ap50:.4f}")

