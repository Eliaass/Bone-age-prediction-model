import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path("/storage/homefs/eb19p026/DL_Project")

CSV_PATH = PROJECT_DIR / "boneage-training-dataset.csv"
IMAGE_DIR = PROJECT_DIR / "cropped_overlayed_RSNA_dataset_1024x1024"

OUTPUT_DIR = PROJECT_DIR / "efficientnetv2-m_training"
PLOT_DIR = OUTPUT_DIR / "plots"

BEST_CHECKPOINT_PATH = OUTPUT_DIR / "efficientnetv2_m_best.pth"
LATEST_CHECKPOINT_PATH = OUTPUT_DIR / "efficientnetv2_m_latest.pth"

RESUME_TRAINING = False
RESUME_FROM_BEST_IF_NO_LATEST = False

MODEL_NAME = "tf_efficientnetv2_m.in21k"
NO_PRETRAINED = False

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024

VAL_SIZE = 0.15
SEED = 42

BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
NUM_WORKERS = 8

EPOCHS = 200
STEPS_PER_EPOCH = None
WARMUP_EPOCHS = 5
PATIENCE = 20

BACKBONE_LR = 2.6827251621760116e-05
HEAD_LR_MULTIPLIER = 5
HEAD_LR = 0.00013413625810880059

WEIGHT_DECAY = 2.66986667427446e-05
DROP_PATH = 0.1
HEAD_DROPOUT = 0.0
HIDDEN_DIM = 1024
SMOOTH_L1_BETA = 3.0

MAX_GRAD_NORM = 1.0
USE_AMP = True

AUGMENTATION_PRESET = "medium"

SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Image indexing / CSV filtering
# ============================================================

def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def build_image_index(image_dir):
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"IMAGE_DIR does not exist: {image_dir}")

    image_index = {}

    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        for path in image_dir.glob(f"*{ext}"):
            image_index[path.stem] = path

    if len(image_index) == 0:
        raise RuntimeError(f"No images found in: {image_dir}")

    print(f"Indexed {len(image_index)} images from: {image_dir}")

    return image_index


def filter_dataframe_to_existing_images(df, image_index, plot_dir):
    df = df.copy()
    df["id"] = df["id"].apply(normalize_id)

    exists_mask = df["id"].isin(image_index.keys())

    missing_df = df.loc[~exists_mask].copy()
    filtered_df = df.loc[exists_mask].copy()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if len(missing_df) > 0:
        missing_path = plot_dir / "missing_images.csv"
        missing_df.to_csv(missing_path, index=False)
        print(f"Warning: {len(missing_df)} CSV rows have no matching image.")
        print(f"Missing image report saved to: {missing_path}")

    print(f"Using {len(filtered_df)} samples after image filtering.")

    return filtered_df


# ============================================================
# Dataset / Model
# ============================================================

class BoneAgeDataset(Dataset):
    def __init__(self, dataframe, image_index, transform):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_index = image_index
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = normalize_id(row["id"])
        image_path = self.image_index[image_id]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        boneage = torch.tensor(float(row["boneage"]), dtype=torch.float32)

        male = torch.tensor(
            [float(str(row["male"]).lower() == "true")],
            dtype=torch.float32,
        )

        return {
            "image": image,
            "male": male,
            "target": boneage,
            "id": image_id,
        }


class BoneAgeEfficientNetV2(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=True,
        drop_path_rate=0.1,
        head_dropout=0.0,
        hidden_dim=1024,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=drop_path_rate,
        )

        feature_dim = self.backbone.num_features

        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image, male):
        features = self.backbone(image)
        features_with_sex = torch.cat([features, male], dim=1)
        prediction = self.regression_head(features_with_sex).squeeze(1)

        return prediction


# ============================================================
# Split / Metrics
# ============================================================

def create_stratified_split(df, val_size, seed):
    df = df.copy()

    df["age_bin"] = pd.qcut(
        df["boneage"],
        q=10,
        duplicates="drop",
        labels=False,
    )

    df["stratify_col"] = df["age_bin"].astype(str) + "_" + df["male"].astype(str)

    stratify_counts = df["stratify_col"].value_counts()
    use_stratify = stratify_counts.min() >= 2

    if use_stratify:
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=seed,
            stratify=df["stratify_col"],
        )
    else:
        print("Warning: Stratified split not possible. Falling back to random split.")

        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=seed,
        )

    train_df = train_df.drop(columns=["age_bin", "stratify_col"])
    val_df = val_df.drop(columns=["age_bin", "stratify_col"])

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    return train_df, val_df


def compute_mae(preds, targets):
    preds = np.asarray(preds)
    targets = np.asarray(targets)

    return float(np.mean(np.abs(preds - targets)))


def compute_rmse(preds, targets):
    preds = np.asarray(preds)
    targets = np.asarray(targets)

    return float(np.sqrt(np.mean((preds - targets) ** 2)))


# ============================================================
# Augmentation
# ============================================================

def build_train_transform():
    if AUGMENTATION_PRESET == "none":
        return transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    if AUGMENTATION_PRESET == "medium":
        return transforms.Compose([
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),

            transforms.RandomAffine(
                degrees=7,
                translate=(0.08, 0.08),
                scale=(0.90, 1.10),
                shear=3,
                fill=0,
            ),

            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    raise ValueError(f"Unknown AUGMENTATION_PRESET: {AUGMENTATION_PRESET}")


def build_val_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# Training / Validation
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp,
    grad_accum_steps,
    max_grad_norm,
    steps_per_epoch=None,
    loader_iter=None,
):
    model.train()

    all_preds = []
    all_targets = []

    optimizer.zero_grad(set_to_none=True)

    num_steps = steps_per_epoch if steps_per_epoch else len(loader)

    progress_bar = tqdm(range(num_steps), desc="Training", leave=False)

    for step_idx in progress_bar:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        except TypeError:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        images = batch["image"].to(device, non_blocking=True)
        males = batch["male"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            preds = model(images, males)
            loss = criterion(preds, targets) / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == num_steps:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

        progress_bar.set_postfix({
            "mae": f"{compute_mae(all_preds, all_targets):.2f}"
        })

    train_mae = compute_mae(all_preds, all_targets)

    return train_mae, loader_iter


@torch.no_grad()
def validate_one_epoch(model, loader, device, use_amp):
    model.eval()

    all_preds = []
    all_targets = []
    all_males = []
    all_ids = []

    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        males = batch["male"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            preds = model(images, males)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_males.extend(males.cpu().numpy().flatten())
        all_ids.extend(batch["id"])

    predictions_df = pd.DataFrame({
        "id": all_ids,
        "target": all_targets,
        "prediction": all_preds,
        "male": all_males,
        "abs_error": np.abs(np.asarray(all_preds) - np.asarray(all_targets)),
        "error": np.asarray(all_preds) - np.asarray(all_targets),
    })

    val_mae = compute_mae(all_preds, all_targets)
    val_rmse = compute_rmse(all_preds, all_targets)

    return val_mae, val_rmse, predictions_df


# ============================================================
# Checkpointing / Scheduling
# ============================================================

def build_scheduler(optimizer):
    def lr_lambda(epoch):
        if WARMUP_EPOCHS > 0 and epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(WARMUP_EPOCHS)

        cosine_epochs = max(1, EPOCHS - WARMUP_EPOCHS)
        progress = float(epoch - WARMUP_EPOCHS) / float(cosine_epochs)
        progress = min(max(progress, 0.0), 1.0)

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best_mae,
    patience_counter,
    history,
    config,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_mae": best_mae,
        "patience_counter": patience_counter,
        "history": history,
        "config": config,
    }

    torch.save(checkpoint, path)


def load_checkpoint_if_available(model, optimizer, scheduler, device):
    if not RESUME_TRAINING:
        return 1, float("inf"), 0, []

    checkpoint_path = None

    if Path(LATEST_CHECKPOINT_PATH).exists():
        checkpoint_path = LATEST_CHECKPOINT_PATH
    elif RESUME_FROM_BEST_IF_NO_LATEST and Path(BEST_CHECKPOINT_PATH).exists():
        checkpoint_path = BEST_CHECKPOINT_PATH

    if checkpoint_path is None:
        print("No checkpoint found. Starting from scratch.")
        return 1, float("inf"), 0, []

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint["epoch"]) + 1
        best_mae = float(checkpoint["best_mae"])
        patience_counter = int(checkpoint.get("patience_counter", 0))
        history = checkpoint.get("history", [])

        print(f"Resumed checkpoint from: {checkpoint_path}")
        print(f"Continuing from epoch {start_epoch}.")
        print(f"Best Val MAE so far: {best_mae:.2f}")

        return start_epoch, best_mae, patience_counter, history

    model.load_state_dict(checkpoint)

    print(f"Loaded old state_dict checkpoint from: {checkpoint_path}")
    print("Optimizer, scheduler, epoch and history could not be restored.")

    return 1, float("inf"), 0, []


# ============================================================
# Visualization
# ============================================================

def save_plots(history_df, predictions_df, plot_dir, prefix="latest"):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(plot_dir / f"{prefix}_history.csv", index=False)
    predictions_df.to_csv(plot_dir / f"{prefix}_validation_predictions.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    axes[0].plot(
        history_df["epoch"],
        history_df["train_mae"],
        label="Train MAE",
        marker="o",
        alpha=0.7,
    )
    axes[0].plot(
        history_df["epoch"],
        history_df["val_mae"],
        label="Val MAE",
        marker="o",
        alpha=0.7,
    )
    axes[0].set_title("Mean Absolute Error")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE (Months)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(
        predictions_df["target"],
        predictions_df["prediction"],
        alpha=0.4,
        s=10,
    )

    lims = [
        min(predictions_df["target"].min(), predictions_df["prediction"].min()),
        max(predictions_df["target"].max(), predictions_df["prediction"].max()),
    ]

    axes[1].plot(lims, lims, "r--", alpha=0.75, zorder=0)
    axes[1].set_title("Predicted vs. True Bone Age")
    axes[1].set_xlabel("True Age (Months)")
    axes[1].set_ylabel("Predicted Age (Months)")

    diff = predictions_df["prediction"] - predictions_df["target"]
    mean_val = (predictions_df["prediction"] + predictions_df["target"]) / 2

    bias = np.mean(diff)
    sd = np.std(diff)

    axes[2].scatter(mean_val, diff, alpha=0.4, s=10)
    axes[2].axhline(bias, color="red", linestyle="-", label=f"Bias: {bias:.2f}m")
    axes[2].axhline(
        bias + 1.96 * sd,
        color="gray",
        linestyle="--",
        label="95% LoA",
    )
    axes[2].axhline(
        bias - 1.96 * sd,
        color="gray",
        linestyle="--",
    )
    axes[2].set_title("Bland-Altman Error Analysis")
    axes[2].set_xlabel("Mean of Prediction and Target (Months)")
    axes[2].set_ylabel("Prediction - Target (Months)")
    axes[2].legend(fontsize="small")

    temp_df = predictions_df.copy()
    temp_df["age_group"] = (temp_df["target"] // 12).astype(int)

    age_stats = (
        temp_df
        .groupby("age_group")
        .agg(
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
            count=("id", "count"),
        )
        .reset_index()
    )

    axes[3].bar(
        age_stats["age_group"],
        age_stats["mae"],
        alpha=0.4,
        label="MAE",
        color="royalblue",
    )
    axes[3].step(
        age_stats["age_group"],
        age_stats["bias"],
        where="mid",
        label="Mean Bias",
        color="crimson",
        linestyle="--",
        linewidth=2,
        marker="D",
        markersize=4,
    )
    axes[3].axhline(
        0,
        color="black",
        linestyle="-",
        linewidth=1,
        alpha=0.5,
    )
    axes[3].set_title("Age Group Analysis")
    axes[3].set_xlabel("Age Group (Years)")
    axes[3].set_ylabel("Months")
    axes[3].legend(fontsize="small")

    plt.tight_layout()
    plt.savefig(plot_dir / f"{prefix}_dashboard.png", dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    import gc

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    gc.collect()

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"

    print("=" * 80)
    print("Bone Age Training - EfficientNetV2-M")
    print("=" * 80)
    print(f"Project directory: {PROJECT_DIR}")
    print(f"CSV path: {CSV_PATH}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model: {MODEL_NAME}")
    print(f"Image size: {IMAGE_HEIGHT}x{IMAGE_WIDTH}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRAD_ACCUM_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Backbone LR: {BACKBONE_LR}")
    print(f"Head LR: {HEAD_LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Drop path: {DROP_PATH}")
    print(f"Head dropout: {HEAD_DROPOUT}")
    print(f"Hidden dim: {HIDDEN_DIM}")
    print(f"Smooth L1 beta: {SMOOTH_L1_BETA}")
    print(f"Augmentation preset: {AUGMENTATION_PRESET}")
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")
    print("=" * 80)

    config = {
        "project_dir": str(PROJECT_DIR),
        "csv_path": str(CSV_PATH),
        "image_dir": str(IMAGE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "model_name": MODEL_NAME,
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "val_size": VAL_SIZE,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM_STEPS,
        "num_workers": NUM_WORKERS,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "warmup_epochs": WARMUP_EPOCHS,
        "patience": PATIENCE,
        "backbone_lr": BACKBONE_LR,
        "head_lr_multiplier": HEAD_LR_MULTIPLIER,
        "head_lr": HEAD_LR,
        "weight_decay": WEIGHT_DECAY,
        "drop_path": DROP_PATH,
        "head_dropout": HEAD_DROPOUT,
        "hidden_dim": HIDDEN_DIM,
        "smooth_l1_beta": SMOOTH_L1_BETA,
        "max_grad_norm": MAX_GRAD_NORM,
        "use_amp": USE_AMP,
        "augmentation_preset": AUGMENTATION_PRESET,
        "pretrained": not NO_PRETRAINED,
    }

    with open(OUTPUT_DIR / "training_config.json", "w") as f:
        json.dump(config, f, indent=4)

    image_index = build_image_index(IMAGE_DIR)

    df = pd.read_csv(CSV_PATH)
    df = filter_dataframe_to_existing_images(df, image_index, PLOT_DIR)

    train_df, val_df = create_stratified_split(df, VAL_SIZE, SEED)

    train_transform = build_train_transform()
    val_transform = build_val_transform()

    train_dataset = BoneAgeDataset(train_df, image_index, train_transform)
    val_dataset = BoneAgeDataset(val_df, image_index, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    model = BoneAgeEfficientNetV2(
        model_name=MODEL_NAME,
        pretrained=not NO_PRETRAINED,
        drop_path_rate=DROP_PATH,
        head_dropout=HEAD_DROPOUT,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": BACKBONE_LR,
            },
            {
                "params": model.regression_head.parameters(),
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = build_scheduler(optimizer)

    criterion = nn.SmoothL1Loss(beta=SMOOTH_L1_BETA)

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    start_epoch, best_mae, patience_counter, history = load_checkpoint_if_available(
        model,
        optimizer,
        scheduler,
        device,
    )

    loader_iter = None

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_mae, loader_iter = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            max_grad_norm=MAX_GRAD_NORM,
            steps_per_epoch=STEPS_PER_EPOCH,
            loader_iter=loader_iter,
        )

        val_mae, val_rmse, val_predictions_df = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            use_amp=use_amp,
        )

        scheduler.step()

        current_lrs = [group["lr"] for group in optimizer.param_groups]

        history.append({
            "epoch": epoch,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "backbone_lr": current_lrs[0],
            "head_lr": current_lrs[1],
        })

        print(
            f"Epoch {epoch} | "
            f"Train MAE: {train_mae:.2f} | "
            f"Val MAE: {val_mae:.2f} | "
            f"Val RMSE: {val_rmse:.2f} | "
            f"Backbone LR: {current_lrs[0]:.3e} | "
            f"Head LR: {current_lrs[1]:.3e}"
        )

        history_df = pd.DataFrame(history)

        if val_mae < best_mae:
            best_mae = val_mae
            patience_counter = 0

            save_checkpoint(
                path=BEST_CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_mae=best_mae,
                patience_counter=patience_counter,
                history=history,
                config=config,
            )

            save_plots(
                history_df=history_df,
                predictions_df=val_predictions_df,
                plot_dir=PLOT_DIR,
                prefix="best",
            )

            print(f"New best model saved. Best Val MAE: {best_mae:.2f}")

        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        save_checkpoint(
            path=LATEST_CHECKPOINT_PATH,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_mae=best_mae,
            patience_counter=patience_counter,
            history=history,
            config=config,
        )

        save_plots(
            history_df=history_df,
            predictions_df=val_predictions_df,
            plot_dir=PLOT_DIR,
            prefix="latest",
        )

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered. Best Val MAE: {best_mae:.2f}")
            break

    print("\nTraining finished.")
    print(f"Best validation MAE: {best_mae:.2f}")
    print(f"Best checkpoint saved to: {BEST_CHECKPOINT_PATH}")
    print(f"Latest checkpoint saved to: {LATEST_CHECKPOINT_PATH}")
    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()