import json
import math
import random
import re
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
IMAGE_DIR = PROJECT_DIR / "RSNA_gradCAMcropped_dataset_512x512"

BASE_CHECKPOINT_PATH = PROJECT_DIR / "convnextv2_tiny_training" / "convnextv2_tiny_best.pth"

OUTPUT_DIR = PROJECT_DIR / "convnextv2_tiny_gradcam_roi_finetuning"
PLOT_DIR = OUTPUT_DIR / "plots"

BEST_CHECKPOINT_PATH = OUTPUT_DIR / "convnextv2_tiny_roi_finetuned_best.pth"
LATEST_CHECKPOINT_PATH = OUTPUT_DIR / "convnextv2_tiny_roi_finetuned_latest.pth"

RESUME_TRAINING = False
RESUME_FROM_BEST_IF_NO_LATEST = False

MODEL_NAME = "convnextv2_tiny"
NO_PRETRAINED = True

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

VAL_SIZE = 0.15
SEED = 42

BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 1
NUM_WORKERS = 8

EPOCHS = 80
STEPS_PER_EPOCH = None
WARMUP_EPOCHS = 3
PATIENCE = 12

BASE_BACKBONE_LR = 2.890293271326065e-05
BASE_HEAD_LR = 0.0002890293271326065
FINETUNE_LR_MULTIPLIER = 0.1

BACKBONE_LR = BASE_BACKBONE_LR * FINETUNE_LR_MULTIPLIER
HEAD_LR = BASE_HEAD_LR * FINETUNE_LR_MULTIPLIER
HEAD_LR_MULTIPLIER = HEAD_LR / BACKBONE_LR

WEIGHT_DECAY = 2.3686574806450328e-05
DROP_PATH = 0.1
HEAD_DROPOUT = 0.0
HIDDEN_DIM = 2048
SMOOTH_L1_BETA = 7.0

MAX_GRAD_NORM = 1.0
USE_AMP = True

AUGMENTATION_PRESET = "light"
MAX_ROIS_PER_ORIGINAL = None  # kept for config compatibility; all crops are used in this pre-split version

TRAIN_IMAGE_DIR = IMAGE_DIR / "train"
VAL_IMAGE_DIR = IMAGE_DIR / "validation"

SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Image indexing / ROI dataframe construction
# ============================================================

def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def extract_original_id_from_stem(stem: str) -> str:
    if "_cropped_" in stem:
        return stem.split("_cropped_")[0]
    return re.findall(r"\d+", stem)[0] if re.findall(r"\d+", stem) else stem


def load_label_lookup(csv_path):
    label_df = pd.read_csv(csv_path).copy()
    label_df["id"] = label_df["id"].apply(normalize_id)
    label_df["boneage"] = pd.to_numeric(label_df["boneage"], errors="coerce")

    if label_df["boneage"].isna().any():
        bad_rows = label_df[label_df["boneage"].isna()]
        raise ValueError(f"Invalid boneage values found:\n{bad_rows}")

    return label_df.set_index("id")


def build_roi_dataframe_from_split_dir(split_dir, label_lookup, output_dir, split_name):
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not split_dir.exists():
        raise FileNotFoundError(f"{split_name} directory does not exist: {split_dir}")

    records = []
    unmatched_images = []

    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        for path in sorted(split_dir.glob(f"*{ext}")):
            image_id = path.stem
            original_id = normalize_id(extract_original_id_from_stem(image_id))

            if original_id not in label_lookup.index:
                unmatched_images.append(str(path))
                continue

            row = label_lookup.loc[original_id]
            records.append({
                "id": image_id,
                "original_id": original_id,
                "image_path": str(path),
                "boneage": float(row["boneage"]),
                "male": row["male"],
                "split": split_name,
            })

    if len(unmatched_images) > 0:
        unmatched_path = output_dir / f"unmatched_{split_name}_roi_images.csv"
        pd.DataFrame({"image_path": unmatched_images}).to_csv(unmatched_path, index=False)
        print(f"Warning: {len(unmatched_images)} {split_name} ROI images could not be matched to CSV IDs.")
        print(f"Unmatched {split_name} ROI report saved to: {unmatched_path}")

    if len(records) == 0:
        raise RuntimeError(f"No {split_name} ROI images could be matched to CSV IDs in: {split_dir}")

    roi_df = pd.DataFrame(records)

    roi_count_path = output_dir / f"{split_name}_roi_counts_per_original.csv"
    roi_df.groupby("original_id").size().reset_index(name="num_rois").to_csv(roi_count_path, index=False)

    print(f"Matched {len(roi_df)} {split_name} ROI images from: {split_dir}")
    print(f"Unique {split_name} original images represented: {roi_df['original_id'].nunique()}")
    print(f"{split_name.capitalize()} ROI count report saved to: {roi_count_path}")

    return roi_df


def check_no_split_leakage(train_df, val_df):
    train_ids = set(train_df["original_id"].astype(str))
    val_ids = set(val_df["original_id"].astype(str))
    overlap = sorted(train_ids.intersection(val_ids))

    if len(overlap) > 0:
        overlap_preview = overlap[:20]
        raise RuntimeError(
            f"Split leakage detected: {len(overlap)} original IDs occur in both train and validation. "
            f"Examples: {overlap_preview}"
        )

    print("Split leakage check passed: no original_id overlap between train and validation.")


# ============================================================
# Dataset / Model
# ============================================================

class BoneAgeROIDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
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
            "id": str(row["id"]),
            "original_id": str(row["original_id"]),
        }


class BoneAgeConvNeXtV2(nn.Module):
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

def create_original_level_split(roi_df, val_size, seed):
    original_df = (
        roi_df[["original_id", "boneage", "male"]]
        .drop_duplicates("original_id")
        .reset_index(drop=True)
    )

    original_df["age_bin"] = pd.qcut(
        original_df["boneage"],
        q=10,
        duplicates="drop",
        labels=False,
    )
    original_df["stratify_col"] = original_df["age_bin"].astype(str) + "_" + original_df["male"].astype(str)

    stratify_counts = original_df["stratify_col"].value_counts()
    use_stratify = stratify_counts.min() >= 2

    if use_stratify:
        train_originals, val_originals = train_test_split(
            original_df,
            test_size=val_size,
            random_state=seed,
            stratify=original_df["stratify_col"],
        )
    else:
        print("Warning: Stratified split not possible. Falling back to random split.")
        train_originals, val_originals = train_test_split(
            original_df,
            test_size=val_size,
            random_state=seed,
        )

    train_ids = set(train_originals["original_id"].astype(str))
    val_ids = set(val_originals["original_id"].astype(str))

    train_df = roi_df[roi_df["original_id"].astype(str).isin(train_ids)].copy()
    val_df = roi_df[roi_df["original_id"].astype(str).isin(val_ids)].copy()

    print(f"Train original images: {len(train_ids)}")
    print(f"Validation original images: {len(val_ids)}")
    print(f"Train ROI samples: {len(train_df)}")
    print(f"Validation ROI samples: {len(val_df)}")

    return train_df, val_df


def compute_mae(preds, targets):
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    return float(np.mean(np.abs(preds - targets)))


def compute_rmse(preds, targets):
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


def aggregate_predictions_by_original(predictions_df):
    agg_df = (
        predictions_df
        .groupby("original_id")
        .agg(
            target=("target", "first"),
            prediction=("prediction", "mean"),
            male=("male", "first"),
            num_rois=("id", "count"),
        )
        .reset_index()
        .rename(columns={"original_id": "id"})
    )

    agg_df["abs_error"] = np.abs(agg_df["prediction"] - agg_df["target"])
    agg_df["error"] = agg_df["prediction"] - agg_df["target"]
    return agg_df


# ============================================================
# Augmentation
# ============================================================

def build_train_transform():
    if AUGMENTATION_PRESET == "none":
        affine_transform = []
    elif AUGMENTATION_PRESET == "light":
        affine_transform = [
            transforms.RandomAffine(
                degrees=3,
                translate=(0.03, 0.03),
                scale=(0.97, 1.03),
                fill=0,
            )
        ]
    elif AUGMENTATION_PRESET == "medium":
        affine_transform = [
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=0,
            )
        ]
    elif AUGMENTATION_PRESET == "current":
        affine_transform = [
            transforms.RandomAffine(
                degrees=5,
                translate=(0.10, 0.10),
                scale=(0.90, 1.10),
                fill=0,
            )
        ]
    else:
        raise ValueError(f"Unknown AUGMENTATION_PRESET: {AUGMENTATION_PRESET}")

    return transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        *affine_transform,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


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
        except (StopIteration, TypeError):
            loader_iter = iter(loader)
            batch = next(loader_iter)

        images = batch["image"].to(device, non_blocking=True)
        males = batch["male"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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
        progress_bar.set_postfix({"mae": f"{compute_mae(all_preds, all_targets):.2f}"})

    train_mae = compute_mae(all_preds, all_targets)
    return train_mae, loader_iter


@torch.no_grad()
def validate_one_epoch(model, loader, device, use_amp):
    model.eval()

    all_preds = []
    all_targets = []
    all_males = []
    all_ids = []
    all_original_ids = []

    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        males = batch["male"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(images, males)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_males.extend(males.cpu().numpy().flatten())
        all_ids.extend(batch["id"])
        all_original_ids.extend(batch["original_id"])

    roi_predictions_df = pd.DataFrame({
        "id": all_ids,
        "original_id": all_original_ids,
        "target": all_targets,
        "prediction": all_preds,
        "male": all_males,
    })
    roi_predictions_df["abs_error"] = np.abs(roi_predictions_df["prediction"] - roi_predictions_df["target"])
    roi_predictions_df["error"] = roi_predictions_df["prediction"] - roi_predictions_df["target"]

    aggregated_predictions_df = aggregate_predictions_by_original(roi_predictions_df)

    val_mae = compute_mae(aggregated_predictions_df["prediction"], aggregated_predictions_df["target"])
    val_rmse = compute_rmse(aggregated_predictions_df["prediction"], aggregated_predictions_df["target"])
    roi_level_mae = compute_mae(roi_predictions_df["prediction"], roi_predictions_df["target"])

    return val_mae, val_rmse, aggregated_predictions_df, roi_predictions_df, roi_level_mae


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

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_mae, patience_counter, history, config):
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


def load_base_model_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        base_best_mae = checkpoint.get("best_mae", None)
        print(f"Loaded base model checkpoint from: {checkpoint_path}")
        if base_best_mae is not None:
            print(f"Base checkpoint best validation MAE: {float(base_best_mae):.2f}")
    else:
        model.load_state_dict(checkpoint, strict=True)
        print(f"Loaded base state_dict checkpoint from: {checkpoint_path}")


def load_finetune_checkpoint_if_available(model, optimizer, scheduler, device):
    if not RESUME_TRAINING:
        return 1, float("inf"), 0, []

    checkpoint_path = None
    if Path(LATEST_CHECKPOINT_PATH).exists():
        checkpoint_path = LATEST_CHECKPOINT_PATH
    elif RESUME_FROM_BEST_IF_NO_LATEST and Path(BEST_CHECKPOINT_PATH).exists():
        checkpoint_path = BEST_CHECKPOINT_PATH

    if checkpoint_path is None:
        print("No fine-tuning checkpoint found. Loading base checkpoint instead.")
        load_base_model_checkpoint(model, BASE_CHECKPOINT_PATH, device)
        return 1, float("inf"), 0, []

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_mae = float(checkpoint["best_mae"])
    patience_counter = int(checkpoint.get("patience_counter", 0))
    history = checkpoint.get("history", [])

    print(f"Resumed fine-tuning checkpoint from: {checkpoint_path}")
    print(f"Continuing from epoch {start_epoch}.")
    print(f"Best fine-tuning Val MAE so far: {best_mae:.2f}")

    return start_epoch, best_mae, patience_counter, history


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

    axes[0].plot(history_df["epoch"], history_df["train_mae"], label="Train MAE", marker="o", alpha=0.7)
    axes[0].plot(history_df["epoch"], history_df["val_mae"], label="Val MAE", marker="o", alpha=0.7)
    axes[0].set_title("Mean Absolute Error")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE (Months)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(predictions_df["target"], predictions_df["prediction"], alpha=0.4, s=10)
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
    axes[2].axhline(bias + 1.96 * sd, color="gray", linestyle="--", label="95% LoA")
    axes[2].axhline(bias - 1.96 * sd, color="gray", linestyle="--")
    axes[2].set_title("Bland-Altman Error Analysis")
    axes[2].set_xlabel("Mean of Prediction and Target (Months)")
    axes[2].set_ylabel("Prediction - Target (Months)")
    axes[2].legend(fontsize="small")

    temp_df = predictions_df.copy()
    temp_df["age_group"] = (temp_df["target"] // 12).astype(int)
    age_stats = (
        temp_df
        .groupby("age_group")
        .agg(mae=("abs_error", "mean"), bias=("error", "mean"), count=("id", "count"))
        .reset_index()
    )

    axes[3].bar(age_stats["age_group"], age_stats["mae"], alpha=0.4, label="MAE", color="royalblue")
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
    axes[3].axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
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
    print("Bone Age Fine-Tuning - ConvNeXtV2-Tiny on Grad-CAM ROI crops")
    print("=" * 80)
    print(f"Project directory: {PROJECT_DIR}")
    print(f"CSV path: {CSV_PATH}")
    print(f"ROI image directory: {IMAGE_DIR}")
    print(f"Train ROI directory: {TRAIN_IMAGE_DIR}")
    print(f"Validation ROI directory: {VAL_IMAGE_DIR}")
    print(f"Base checkpoint: {BASE_CHECKPOINT_PATH}")
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
    print(f"Max ROIs per original: {MAX_ROIS_PER_ORIGINAL}")
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")
    print("=" * 80)

    available_convnextv2_models = timm.list_models("*convnextv2*")
    if MODEL_NAME not in available_convnextv2_models:
        print(f"Model not found in current timm installation: {MODEL_NAME}")
        print("Available ConvNeXtV2 models:")
        for model_name in available_convnextv2_models:
            print(f"  - {model_name}")
        raise ValueError(f"Invalid ConvNeXtV2 model name: {MODEL_NAME}")

    config = {
        "project_dir": str(PROJECT_DIR),
        "csv_path": str(CSV_PATH),
        "image_dir": str(IMAGE_DIR),
        "train_image_dir": str(TRAIN_IMAGE_DIR),
        "val_image_dir": str(VAL_IMAGE_DIR),
        "split_source": "pre_split_train_validation_folders",
        "base_checkpoint_path": str(BASE_CHECKPOINT_PATH),
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
        "base_backbone_lr": BASE_BACKBONE_LR,
        "base_head_lr": BASE_HEAD_LR,
        "finetune_lr_multiplier": FINETUNE_LR_MULTIPLIER,
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
        "pretrained_for_model_init": not NO_PRETRAINED,
        "max_rois_per_original": MAX_ROIS_PER_ORIGINAL,
        "validation_metric_level": "original_id_aggregated_mean_prediction",
    }

    with open(OUTPUT_DIR / "finetuning_config.json", "w") as f:
        json.dump(config, f, indent=4)

    label_lookup = load_label_lookup(CSV_PATH)

    train_df = build_roi_dataframe_from_split_dir(
        split_dir=TRAIN_IMAGE_DIR,
        label_lookup=label_lookup,
        output_dir=PLOT_DIR,
        split_name="train",
    )

    val_df = build_roi_dataframe_from_split_dir(
        split_dir=VAL_IMAGE_DIR,
        label_lookup=label_lookup,
        output_dir=PLOT_DIR,
        split_name="validation",
    )

    check_no_split_leakage(train_df, val_df)

    print(f"Train original images: {train_df['original_id'].nunique()}")
    print(f"Validation original images: {val_df['original_id'].nunique()}")
    print(f"Train ROI samples: {len(train_df)}")
    print(f"Validation ROI samples: {len(val_df)}")

    train_df.to_csv(PLOT_DIR / "train_roi_samples.csv", index=False)
    val_df.to_csv(PLOT_DIR / "val_roi_samples.csv", index=False)

    train_transform = build_train_transform()
    val_transform = build_val_transform()

    train_dataset = BoneAgeROIDataset(train_df, train_transform)
    val_dataset = BoneAgeROIDataset(val_df, val_transform)

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

    model = BoneAgeConvNeXtV2(
        model_name=MODEL_NAME,
        pretrained=not NO_PRETRAINED,
        drop_path_rate=DROP_PATH,
        head_dropout=HEAD_DROPOUT,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
            {"params": model.regression_head.parameters(), "lr": HEAD_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = build_scheduler(optimizer)
    criterion = nn.SmoothL1Loss(beta=SMOOTH_L1_BETA)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    if RESUME_TRAINING:
        start_epoch, best_mae, patience_counter, history = load_finetune_checkpoint_if_available(
            model,
            optimizer,
            scheduler,
            device,
        )
    else:
        load_base_model_checkpoint(model, BASE_CHECKPOINT_PATH, device)
        start_epoch, best_mae, patience_counter, history = 1, float("inf"), 0, []

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

        val_mae, val_rmse, val_predictions_df, val_roi_predictions_df, val_roi_mae = validate_one_epoch(
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
            "val_roi_level_mae": val_roi_mae,
            "backbone_lr": current_lrs[0],
            "head_lr": current_lrs[1],
        })

        print(
            f"Epoch {epoch} | "
            f"Train ROI MAE: {train_mae:.2f} | "
            f"Val aggregated MAE: {val_mae:.2f} | "
            f"Val ROI MAE: {val_roi_mae:.2f} | "
            f"Val RMSE: {val_rmse:.2f} | "
            f"Backbone LR: {current_lrs[0]:.3e} | "
            f"Head LR: {current_lrs[1]:.3e}"
        )

        history_df = pd.DataFrame(history)
        val_roi_predictions_df.to_csv(PLOT_DIR / "latest_validation_roi_predictions.csv", index=False)

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
            val_roi_predictions_df.to_csv(PLOT_DIR / "best_validation_roi_predictions.csv", index=False)
            print(f"New best fine-tuned model saved. Best aggregated Val MAE: {best_mae:.2f}")
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
            print(f"Early stopping triggered. Best aggregated Val MAE: {best_mae:.2f}")
            break

    print("\nFine-tuning finished.")
    print(f"Best aggregated validation MAE: {best_mae:.2f}")
    print(f"Best checkpoint saved to: {BEST_CHECKPOINT_PATH}")
    print(f"Latest checkpoint saved to: {LATEST_CHECKPOINT_PATH}")
    print(f"Plots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
