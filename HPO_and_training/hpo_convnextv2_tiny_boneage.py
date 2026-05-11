# hpo_convnextv2_tiny_boneage.py

import argparse
import gc
import json
import math
import random
import signal
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ============================================================
# Defaults
# ============================================================

DEFAULT_PROJECT_DIR = "/storage/homefs/eb19p026/DL_Project"
DEFAULT_CSV_PATH = f"{DEFAULT_PROJECT_DIR}/boneage-training-dataset.csv"
DEFAULT_IMAGE_DIR = f"{DEFAULT_PROJECT_DIR}/cropped_overlayed_RSNA_dataset_1024x1024"
DEFAULT_HPO_DIR = f"{DEFAULT_PROJECT_DIR}/hpo_convnextv2_tiny_boneage"

DEFAULT_MODEL_NAME = "convnextv2_tiny"
SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

CURRENT_STUDY = None
CURRENT_ARGS = None


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def build_image_index(image_dir: str) -> dict:
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"IMAGE_DIR does not exist: {image_dir}")

    image_index = {}

    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        for path in image_dir.glob(f"*{ext}"):
            image_index[path.stem] = path

    if len(image_index) == 0:
        raise RuntimeError(f"No images found in: {image_dir}")

    return image_index


def filter_dataframe_to_existing_images(
    df: pd.DataFrame,
    image_index: dict,
    output_dir: str,
) -> pd.DataFrame:
    df = df.copy()

    df["id"] = df["id"].apply(normalize_id)
    df["boneage"] = pd.to_numeric(df["boneage"], errors="coerce")

    if df["boneage"].isna().any():
        bad_rows = df[df["boneage"].isna()]
        raise ValueError(f"Invalid boneage values found:\n{bad_rows}")

    exists_mask = df["id"].isin(image_index.keys())
    missing_df = df.loc[~exists_mask].copy()
    filtered_df = df.loc[exists_mask].copy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(missing_df) > 0:
        missing_path = output_dir / "missing_images.csv"
        missing_df.to_csv(missing_path, index=False)
        print(f"Warning: {len(missing_df)} rows missing images. Saved to: {missing_path}")

    if len(filtered_df) == 0:
        raise RuntimeError("No CSV rows match existing images.")

    return filtered_df


def create_stratified_split(df: pd.DataFrame, val_size: float, seed: int):
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
        print("Warning: Stratification fallback to random split.")
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=seed,
        )

    return (
        train_df.drop(columns=["age_bin", "stratify_col"]),
        val_df.drop(columns=["age_bin", "stratify_col"]),
    )


def compute_mae(preds, targets) -> float:
    return float(np.mean(np.abs(np.asarray(preds) - np.asarray(targets))))


def compute_rmse(preds, targets) -> float:
    return float(np.sqrt(np.mean((np.asarray(preds) - np.asarray(targets)) ** 2)))


# ============================================================
# Dataset / Model
# ============================================================

class BoneAgeDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_index: dict, transform):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_index = image_index
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.image_index[normalize_id(row["id"])]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        target = torch.tensor(float(row["boneage"]), dtype=torch.float32)

        male = torch.tensor(
            [float(str(row["male"]).lower() == "true")],
            dtype=torch.float32,
        )

        return {
            "image": image,
            "male": male,
            "target": target,
            "id": row["id"],
        }


class BoneAgeConvNeXtV2(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        drop_path_rate: float = 0.1,
        head_dropout: float = 0.2,
        hidden_dim: int = 1024,
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
        return self.regression_head(features_with_sex).squeeze(1)


# ============================================================
# Transforms / Optimizer / Scheduler
# ============================================================

def get_train_transform(image_size: int, augmentation_preset: str):
    affine_transform = []

    if augmentation_preset == "none":
        affine_transform = []

    elif augmentation_preset == "light":
        affine_transform = [
            transforms.RandomAffine(
                degrees=3,
                translate=(0.03, 0.03),
                scale=(0.97, 1.03),
                fill=0,
            )
        ]

    elif augmentation_preset == "medium":
        affine_transform = [
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=0,
            )
        ]

    elif augmentation_preset == "current":
        affine_transform = [
            transforms.RandomAffine(
                degrees=5,
                translate=(0.10, 0.10),
                scale=(0.90, 1.10),
                fill=0,
            )
        ]

    else:
        raise ValueError(f"Unknown augmentation_preset: {augmentation_preset}")

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        *affine_transform,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_val_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_optimizer(model, backbone_lr: float, head_lr: float, weight_decay: float):
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.regression_head.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer, total_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        cosine_epochs = max(1, total_epochs - warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(cosine_epochs)
        progress = min(max(progress, 0.0), 1.0)

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


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
    steps_per_epoch,
    loader_iter=None,
):
    model.train()

    all_preds = []
    all_targets = []

    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(range(steps_per_epoch), desc="Training", leave=False)

    for step_idx in progress_bar:
        try:
            batch = next(loader_iter)
        except (StopIteration, TypeError):
            loader_iter = iter(loader)
            batch = next(loader_iter)

        images = batch["image"].to(device, non_blocking=True)
        males = batch["male"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        if not torch.isfinite(images).all():
            raise ValueError(f"Non-finite image tensor found. IDs: {batch['id']}")

        if not torch.isfinite(targets).all():
            raise ValueError(f"Non-finite target found. IDs: {batch['id']}")

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(images, males)

            if not torch.isfinite(preds).all():
                raise ValueError(f"Non-finite predictions found. IDs: {batch['id']}")

            loss = criterion(preds, targets) / grad_accum_steps

            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss found. IDs: {batch['id']}")

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == steps_per_epoch:
            if scaler:
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

    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        males = batch["male"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(images, males)

        if not torch.isfinite(preds).all():
            raise ValueError(f"Non-finite validation predictions found. IDs: {batch['id']}")

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    val_mae = compute_mae(all_preds, all_targets)
    val_rmse = compute_rmse(all_preds, all_targets)

    return val_mae, val_rmse


# ============================================================
# Hyperparameter Optimization
# ============================================================

def suggest_params(trial):
    image_size = trial.suggest_categorical(
        "image_size",
        [512, 1024],
    )

    backbone_lr = trial.suggest_float(
        "backbone_lr",
        2e-6,
        3e-5,
        log=True,
    )

    head_lr_multiplier = trial.suggest_categorical(
        "head_lr_multiplier",
        [3, 5, 8, 10],
    )

    head_lr = backbone_lr * head_lr_multiplier

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-4,
        log=True,
    )

    drop_path = trial.suggest_categorical(
        "drop_path",
        [0.0, 0.05, 0.1, 0.15, 0.2],
    )

    head_dropout = trial.suggest_categorical(
        "head_dropout",
        [0.0, 0.1, 0.2, 0.3],
    )

    hidden_dim = trial.suggest_categorical(
        "hidden_dim",
        [512, 1024, 2048],
    )

    smooth_l1_beta = trial.suggest_categorical(
        "smooth_l1_beta",
        [1.0, 3.0, 5.0, 7.0, 10.0],
    )

    augmentation_preset = trial.suggest_categorical(
        "augmentation_preset",
        ["none", "light", "medium", "current"],
    )

    return {
        "image_size": image_size,
        "backbone_lr": backbone_lr,
        "head_lr_multiplier": head_lr_multiplier,
        "head_lr": head_lr,
        "weight_decay": weight_decay,
        "drop_path": drop_path,
        "head_dropout": head_dropout,
        "hidden_dim": hidden_dim,
        "smooth_l1_beta": smooth_l1_beta,
        "augmentation_preset": augmentation_preset,
    }


def objective(trial, train_df, val_df, image_index, device, args):
    params = suggest_params(trial)

    train_transform = get_train_transform(
        image_size=params["image_size"],
        augmentation_preset=params["augmentation_preset"],
    )

    val_transform = get_val_transform(
        image_size=params["image_size"],
    )

    train_loader = DataLoader(
        BoneAgeDataset(train_df, image_index, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        BoneAgeDataset(val_df, image_index, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    model = None
    optimizer = None
    scheduler = None
    criterion = None

    try:
        model = BoneAgeConvNeXtV2(
            model_name=args.model_name,
            pretrained=not args.no_pretrained,
            drop_path_rate=params["drop_path"],
            head_dropout=params["head_dropout"],
            hidden_dim=params["hidden_dim"],
        ).to(device)

        optimizer = build_optimizer(
            model=model,
            backbone_lr=params["backbone_lr"],
            head_lr=params["head_lr"],
            weight_decay=params["weight_decay"],
        )

        scheduler = build_scheduler(
            optimizer=optimizer,
            total_epochs=args.hpo_epochs,
            warmup_epochs=min(args.warmup_epochs, max(1, args.hpo_epochs // 4)),
        )

        criterion = nn.SmoothL1Loss(beta=params["smooth_l1_beta"])

        use_amp = args.use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        best_val_mae = float("inf")
        patience_counter = 0
        loader_iter = None

        for epoch in range(1, args.hpo_epochs + 1):
            train_mae, loader_iter = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
                grad_accum_steps=args.grad_accum_steps,
                max_grad_norm=args.max_grad_norm,
                steps_per_epoch=args.hpo_steps_per_epoch,
                loader_iter=loader_iter,
            )

            val_mae, val_rmse = validate_one_epoch(
                model=model,
                loader=val_loader,
                device=device,
                use_amp=use_amp,
            )

            scheduler.step()

            trial.report(val_mae, epoch)

            print(
                f"Trial {trial.number:03d} | "
                f"Epoch {epoch:03d}/{args.hpo_epochs} | "
                f"Img {params['image_size']} | "
                f"Train MAE: {train_mae:.2f} | "
                f"Val MAE: {val_mae:.2f} | "
                f"Val RMSE: {val_rmse:.2f} | "
                f"Best Trial MAE: {min(best_val_mae, val_mae):.2f}"
            )

            if not np.isfinite(val_mae):
                return float("inf")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
            else:
                patience_counter += 1

            if trial.should_prune():
                raise optuna.TrialPruned()

            if patience_counter >= args.hpo_patience:
                print(
                    f"Trial {trial.number:03d} stopped early. "
                    f"Best Val MAE: {best_val_mae:.2f}"
                )
                break

        return best_val_mae

    except RuntimeError as error:
        error_message = str(error).lower()

        if "out of memory" in error_message or "cuda" in error_message:
            print(f"Trial {trial.number:03d} failed because of CUDA/OOM.")
            torch.cuda.empty_cache()
            return float("inf")

        raise error

    finally:
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if scheduler is not None:
            del scheduler
        if criterion is not None:
            del criterion

        torch.cuda.empty_cache()
        gc.collect()


# ============================================================
# Saving
# ============================================================

def get_search_space():
    return {
        "image_size": [512, 1024],
        "backbone_lr": "log-uniform [2e-6, 3e-5]",
        "head_lr_multiplier": [3, 5, 8, 10],
        "head_lr": "derived as backbone_lr * head_lr_multiplier",
        "weight_decay": "log-uniform [1e-6, 1e-4]",
        "drop_path": [0.0, 0.05, 0.1, 0.15, 0.2],
        "head_dropout": [0.0, 0.1, 0.2, 0.3],
        "hidden_dim": [512, 1024, 2048],
        "smooth_l1_beta": [1.0, 3.0, 5.0, 7.0, 10.0],
        "augmentation_preset": ["none", "light", "medium", "current"],
    }


def get_augmentation_presets():
    return {
        "none": {
            "degrees": 0,
            "translate": 0.0,
            "scale": [1.0, 1.0],
        },
        "light": {
            "degrees": 3,
            "translate": 0.03,
            "scale": [0.97, 1.03],
        },
        "medium": {
            "degrees": 5,
            "translate": 0.05,
            "scale": [0.95, 1.05],
        },
        "current": {
            "degrees": 5,
            "translate": 0.10,
            "scale": [0.90, 1.10],
        },
    }


def build_result_dict(study, args, status: str):
    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    pruned_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.PRUNED
    ]

    failed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.FAIL
    ]

    running_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.RUNNING
    ]

    result = {
        "status": status,
        "completed_trials": len(completed_trials),
        "pruned_trials": len(pruned_trials),
        "failed_trials": len(failed_trials),
        "running_trials": len(running_trials),
        "total_trials_recorded": len(study.trials),
        "hpo_config": {
            "n_trials_target": args.n_trials,
            "hpo_epochs": args.hpo_epochs,
            "hpo_steps_per_epoch": args.hpo_steps_per_epoch,
            "hpo_patience": args.hpo_patience,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "num_workers": args.num_workers,
            "model_name": args.model_name,
            "val_size": args.val_size,
            "seed": args.seed,
            "use_amp": args.use_amp,
            "max_grad_norm": args.max_grad_norm,
            "warmup_epochs": args.warmup_epochs,
            "project_dir": args.project_dir,
            "csv_path": args.csv_path,
            "image_dir": args.image_dir,
            "storage": args.storage,
            "study_name": args.study_name,
        },
        "search_space": get_search_space(),
        "augmentation_presets": get_augmentation_presets(),
    }

    if len(completed_trials) > 0:
        best_params = dict(study.best_params)

        if "backbone_lr" in best_params and "head_lr_multiplier" in best_params:
            best_params["head_lr"] = (
                best_params["backbone_lr"] * best_params["head_lr_multiplier"]
            )

        result.update({
            "best_val_mae": float(study.best_value),
            "best_trial_number": int(study.best_trial.number),
            "best_params": best_params,
        })
    else:
        result.update({
            "best_val_mae": None,
            "best_trial_number": None,
            "best_params": None,
        })

    return result


def save_hpo_progress(study, hpo_dir, args, status: str = "partial"):
    hpo_dir = Path(hpo_dir)
    hpo_dir.mkdir(parents=True, exist_ok=True)

    if status == "completed":
        json_path = hpo_dir / "best_hpo_result.json"
        csv_path = hpo_dir / "optuna_trials.csv"
    else:
        json_path = hpo_dir / "best_hpo_result_partial.json"
        csv_path = hpo_dir / "optuna_trials_partial.csv"

    trials_df = study.trials_dataframe()
    trials_df.to_csv(csv_path, index=False)

    result = build_result_dict(study, args, status=status)

    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Saved HPO {status} result to: {json_path}")
    print(f"Saved HPO {status} trials to: {csv_path}")


def handle_signal(signum, frame):
    print(f"\nReceived signal {signum}. Saving current HPO progress before exit...")

    global CURRENT_STUDY
    global CURRENT_ARGS

    if CURRENT_STUDY is not None and CURRENT_ARGS is not None:
        try:
            save_hpo_progress(
                study=CURRENT_STUDY,
                hpo_dir=CURRENT_ARGS.hpo_dir,
                args=CURRENT_ARGS,
                status="interrupted",
            )
        except Exception as error:
            print(f"Failed to save interrupted HPO progress: {error}")

    sys.exit(128 + signum)


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_dir", type=str, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--hpo_dir", type=str, default=DEFAULT_HPO_DIR)

    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--no_pretrained", action="store_true")

    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--n_trials", type=int, default=24)
    parser.add_argument("--hpo_epochs", type=int, default=20)
    parser.add_argument("--hpo_steps_per_epoch", type=int, default=150)
    parser.add_argument("--hpo_patience", type=int, default=5)

    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument(
        "--study_name",
        type=str,
        default="hpo_convnextv2_tiny_boneage",
    )

    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URI. If omitted, SQLite DB is created inside hpo_dir.",
    )

    return parser.parse_args()


def run_hpo(args):
    global CURRENT_STUDY
    global CURRENT_ARGS

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    hpo_dir = Path(args.hpo_dir)
    hpo_dir.mkdir(parents=True, exist_ok=True)

    if args.storage is None:
        args.storage = f"sqlite:///{hpo_dir / 'optuna_study.db'}"

    set_seed(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("============================================================")
    print("HPO configuration")
    print("============================================================")
    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Using pretrained: {not args.no_pretrained}")
    print(f"Using AMP: {args.use_amp and device.type == 'cuda'}")
    print(f"Project dir: {args.project_dir}")
    print(f"CSV path: {args.csv_path}")
    print(f"Image dir: {args.image_dir}")
    print(f"HPO dir: {args.hpo_dir}")
    print(f"Study name: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Batch size: {args.batch_size}")
    print(f"Grad accum steps: {args.grad_accum_steps}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps}")
    print(f"Target trials: {args.n_trials}")
    print(f"HPO epochs per trial: {args.hpo_epochs}")
    print(f"HPO steps per epoch: {args.hpo_steps_per_epoch}")
    print("============================================================")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory GB: {round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)}")

    available_convnextv2_models = timm.list_models("*convnextv2*")

    if args.model_name not in available_convnextv2_models:
        print(f"Model not found in current timm installation: {args.model_name}")
        print("Available ConvNeXtV2 models:")
        for model_name in available_convnextv2_models:
            print(f"  - {model_name}")
        raise ValueError(f"Invalid ConvNeXtV2 model name: {args.model_name}")

    image_index = build_image_index(args.image_dir)

    df = filter_dataframe_to_existing_images(
        pd.read_csv(args.csv_path),
        image_index,
        args.hpo_dir,
    )

    train_df, val_df = create_stratified_split(
        df=df,
        val_size=args.val_size,
        seed=args.seed,
    )

    sampler = optuna.samplers.TPESampler(seed=args.seed)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    CURRENT_STUDY = study
    CURRENT_ARGS = args

    if len(study.trials) == 0:
        study.enqueue_trial({
            "image_size": 1024,
            "backbone_lr": 1e-5,
            "head_lr_multiplier": 5,
            "weight_decay": 1e-5,
            "drop_path": 0.1,
            "head_dropout": 0.1,
            "hidden_dim": 1024,
            "smooth_l1_beta": 5.0,
            "augmentation_preset": "medium",
        })

    recorded_trials = len(study.trials)
    remaining_trials = max(0, args.n_trials - recorded_trials)

    print(f"Existing trials in study: {recorded_trials}")
    print(f"Remaining trials to run: {remaining_trials}")

    save_hpo_progress(
        study=study,
        hpo_dir=args.hpo_dir,
        args=args,
        status="partial",
    )

    def save_after_trial(study_, trial_):
        save_hpo_progress(
            study=study_,
            hpo_dir=args.hpo_dir,
            args=args,
            status="partial",
        )

    try:
        if remaining_trials > 0:
            study.optimize(
                lambda trial: objective(
                    trial,
                    train_df,
                    val_df,
                    image_index,
                    device,
                    args,
                ),
                n_trials=remaining_trials,
                callbacks=[save_after_trial],
                gc_after_trial=True,
            )
        else:
            print("No remaining trials. Existing study already reached target n_trials.")

    finally:
        save_hpo_progress(
            study=study,
            hpo_dir=args.hpo_dir,
            args=args,
            status="partial",
        )

    save_hpo_progress(
        study=study,
        hpo_dir=args.hpo_dir,
        args=args,
        status="completed",
    )

    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    print("\nHPO finished.")

    if len(completed_trials) > 0:
        best_params = dict(study.best_params)
        best_params["head_lr"] = (
            best_params["backbone_lr"] * best_params["head_lr_multiplier"]
        )

        print(f"Best Val MAE: {study.best_value:.4f}")
        print(f"Best trial number: {study.best_trial.number}")
        print("Best params:")

        for key, value in best_params.items():
            print(f"  {key}: {value}")
    else:
        print("No completed trials found.")

    print(f"\nFinal result saved to: {hpo_dir / 'best_hpo_result.json'}")
    print(f"Partial result saved to: {hpo_dir / 'best_hpo_result_partial.json'}")
    print(f"Optuna SQLite storage: {hpo_dir / 'optuna_study.db'}")


if __name__ == "__main__":
    args = parse_args()
    run_hpo(args)