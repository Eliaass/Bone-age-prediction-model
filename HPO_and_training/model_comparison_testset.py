import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path("/storage/homefs/eb19p026/DL_Project")

TEST_CSV_PATH = PROJECT_DIR / "boneage-test-dataset.csv"

# Use the highest-resolution test folder.
# Each model will resize the images internally to its required input size.
TEST_IMAGE_DIR = PROJECT_DIR / "cropped_overlayed_RSNA_testset_1024x1024"

OUTPUT_DIR = PROJECT_DIR / "final_testset_model_comparison"

BATCH_SIZE = 32
NUM_WORKERS = 8
USE_AMP = True

# For testing, pretrained weights are not needed because the trained checkpoint is loaded.
NO_PRETRAINED = True

SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


MODELS_TO_EVALUATE = [
    {
        "name": "convnextv2_tiny_full_hand",
        "timm_model_name": "convnextv2_tiny",
        "checkpoint_path": PROJECT_DIR / "convnextv2_tiny_training" / "convnextv2_tiny_best.pth",
        "hidden_dim": 2048,
        "head_dropout": 0.0,
        "drop_path": 0.1,
        "image_height": 512,
        "image_width": 512,
    },
    {
        "name": "convnextv2_tiny_roi_finetuned",
        "timm_model_name": "convnextv2_tiny",
        "checkpoint_path": PROJECT_DIR / "convnextv2_tiny_gradcam_roi_finetuning" / "convnextv2_tiny_roi_finetuned_best.pth",
        "hidden_dim": 2048,
        "head_dropout": 0.0,
        "drop_path": 0.1,
        "image_height": 512,
        "image_width": 512,
    },
    {
        "name": "efficientnetv2_m",
        "timm_model_name": "tf_efficientnetv2_m",
        "checkpoint_path": PROJECT_DIR / "efficientnetv2-m_training" / "efficientnetv2_m_best.pth",
        "hidden_dim": 1024,
        "head_dropout": 0.0,
        "drop_path": 0.1,
        "image_height": 1024,
        "image_width": 1024,
    },
]


# ============================================================
# Utility
# ============================================================

def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return Path(str(value).strip()).stem


def build_image_index(image_dir):
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Test image directory does not exist: {image_dir}")

    image_index = {}

    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        for path in image_dir.glob(f"*{ext}"):
            image_index[path.stem] = path

    if len(image_index) == 0:
        raise RuntimeError(f"No test images found in: {image_dir}")

    print(f"Indexed {len(image_index)} test images from: {image_dir}")
    return image_index


def standardize_dataframe_columns(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    normalized_to_original = {
        str(col).strip().lower().replace(" ", "_").replace("-", "_"): col
        for col in df.columns
    }

    id_aliases = [
        "id",
        "image_id",
        "imageid",
        "case_id",
        "filename",
        "file_name",
        "image",
    ]

    target_aliases = [
        "boneage",
        "bone_age",
        "target",
        "age",
        "bone_age_months",
    ]

    male_aliases = [
        "male",
        "sex",
        "gender",
    ]

    def find_column(aliases):
        for alias in aliases:
            if alias in normalized_to_original:
                return normalized_to_original[alias]
        return None

    id_col = find_column(id_aliases)
    target_col = find_column(target_aliases)
    male_col = find_column(male_aliases)

    if id_col is None:
        raise ValueError(
            f"Could not find image ID column. Available columns: {df.columns.tolist()}"
        )

    if target_col is None:
        raise ValueError(
            f"Could not find bone age target column. Available columns: {df.columns.tolist()}"
        )

    if male_col is None:
        raise ValueError(
            f"Could not find male/sex column. Available columns: {df.columns.tolist()}"
        )

    df = df.rename(
        columns={
            id_col: "id",
            target_col: "boneage",
            male_col: "male",
        }
    )

    df["id"] = df["id"].apply(normalize_id)

    return df


def filter_test_dataframe_to_existing_images(df, image_index):
    df = standardize_dataframe_columns(df)

    exists_mask = df["id"].isin(image_index.keys())
    missing_df = df.loc[~exists_mask].copy()
    filtered_df = df.loc[exists_mask].copy()

    if len(missing_df) > 0:
        missing_path = OUTPUT_DIR / "test_missing_images.csv"
        missing_df.to_csv(missing_path, index=False)
        print(f"Warning: {len(missing_df)} test rows have no matching image.")
        print(f"Missing image report saved to: {missing_path}")

    if len(filtered_df) == 0:
        raise RuntimeError("No test samples remain after matching CSV IDs to image files.")

    print(f"Using {len(filtered_df)} test samples after image filtering.")
    return filtered_df


# ============================================================
# Dataset
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

        target = torch.tensor(float(row["boneage"]), dtype=torch.float32)

        male = torch.tensor(
            [float(str(row["male"]).lower() in ["true", "1", "male", "m"])],
            dtype=torch.float32,
        )

        return {
            "image": image,
            "male": male,
            "target": target,
            "id": image_id,
        }


def build_eval_transform(image_height, image_width):
    return transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# Model
# ============================================================

class BoneAgeTimmRegressor(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained=False,
        drop_path_rate=0.0,
        head_dropout=0.0,
        hidden_dim=2048,
    ):
        super().__init__()

        model_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
            "global_pool": "avg",
        }

        if drop_path_rate is not None:
            model_kwargs["drop_path_rate"] = drop_path_rate

        self.backbone = timm.create_model(
            model_name,
            **model_kwargs,
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


def check_timm_model_available(model_name):
    available = timm.list_models(model_name)

    if len(available) > 0:
        return

    all_models = timm.list_models("*efficientnetv2*") + timm.list_models("*convnextv2*")

    print(f"Requested timm model not found: {model_name}")
    print("Potentially relevant available models:")
    for name in all_models:
        print(" -", name)

    raise ValueError(f"Model not available in current timm installation: {model_name}")


def load_model_from_checkpoint(model_cfg, device):
    checkpoint_path = Path(model_cfg["checkpoint_path"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    timm_model_name = model_cfg["timm_model_name"]
    check_timm_model_available(timm_model_name)

    model = BoneAgeTimmRegressor(
        model_name=timm_model_name,
        pretrained=not NO_PRETRAINED,
        drop_path_rate=model_cfg.get("drop_path", 0.0),
        head_dropout=model_cfg.get("head_dropout", 0.0),
        hidden_dim=model_cfg.get("hidden_dim", 2048),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model_state_dict from: {checkpoint_path}")

        if "best_mae" in checkpoint:
            print(f"Checkpoint best validation MAE: {float(checkpoint['best_mae']):.3f} months")

        if "epoch" in checkpoint:
            print(f"Checkpoint epoch: {int(checkpoint['epoch'])}")

        if "config" in checkpoint:
            checkpoint_config = checkpoint["config"]
            print("Checkpoint config summary:")
            for key in ["model_name", "image_height", "image_width", "hidden_dim", "drop_path", "head_dropout"]:
                if key in checkpoint_config:
                    print(f"  {key}: {checkpoint_config[key]}")

    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded raw state_dict from: {checkpoint_path}")

    model.eval()
    return model


# ============================================================
# Evaluation
# ============================================================

def compute_metrics(preds, targets):
    preds = np.asarray(preds, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    errors = preds - targets
    abs_errors = np.abs(errors)

    return {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "median_ae": float(np.median(abs_errors)),
        "mean_error_bias": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "loa_lower": float(np.mean(errors) - 1.96 * np.std(errors)),
        "loa_upper": float(np.mean(errors) + 1.96 * np.std(errors)),
    }


@torch.no_grad()
def evaluate_model(model, loader, device, use_amp):
    model.eval()

    all_ids = []
    all_targets = []
    all_preds = []
    all_males = []

    for batch in tqdm(loader, desc="Testing", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        males = batch["male"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(images, males)

        all_ids.extend(batch["id"])
        all_targets.extend(targets.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_males.extend(males.detach().cpu().numpy().flatten())

    predictions_df = pd.DataFrame({
        "id": all_ids,
        "target": all_targets,
        "prediction": all_preds,
        "male": all_males,
    })

    predictions_df["error"] = predictions_df["prediction"] - predictions_df["target"]
    predictions_df["abs_error"] = np.abs(predictions_df["error"])

    metrics = compute_metrics(
        predictions_df["prediction"].to_numpy(),
        predictions_df["target"].to_numpy(),
    )

    return metrics, predictions_df


# ============================================================
# Main
# ============================================================

def main():
    import gc

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"

    print("=" * 80)
    print("Final Test-Set Model Comparison")
    print("=" * 80)
    print(f"Test CSV: {TEST_CSV_PATH}")
    print(f"Test image directory: {TEST_IMAGE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")
    print(f"NO_PRETRAINED: {NO_PRETRAINED}")
    print("=" * 80)

    test_image_index = build_image_index(TEST_IMAGE_DIR)

    test_df = pd.read_csv(TEST_CSV_PATH)
    test_df = filter_test_dataframe_to_existing_images(test_df, test_image_index)

    all_metrics = []

    for model_cfg in MODELS_TO_EVALUATE:
        model_name_for_report = model_cfg["name"]
        checkpoint_path = model_cfg["checkpoint_path"]

        image_height = model_cfg["image_height"]
        image_width = model_cfg["image_width"]

        print("\n" + "=" * 80)
        print(f"Evaluating model: {model_name_for_report}")
        print(f"timm model: {model_cfg['timm_model_name']}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Evaluation image size: {image_height}x{image_width}")
        print(f"Hidden dim: {model_cfg['hidden_dim']}")
        print(f"Drop path: {model_cfg['drop_path']}")
        print("=" * 80)

        transform = build_eval_transform(
            image_height=image_height,
            image_width=image_width,
        )

        test_dataset = BoneAgeDataset(
            dataframe=test_df,
            image_index=test_image_index,
            transform=transform,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )

        model = load_model_from_checkpoint(model_cfg, device)

        metrics, predictions_df = evaluate_model(
            model=model,
            loader=test_loader,
            device=device,
            use_amp=use_amp,
        )

        predictions_path = OUTPUT_DIR / f"{model_name_for_report}_test_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)

        row = {
            "model": model_name_for_report,
            "timm_model_name": model_cfg["timm_model_name"],
            "checkpoint_path": str(checkpoint_path),
            "test_csv_path": str(TEST_CSV_PATH),
            "test_image_dir": str(TEST_IMAGE_DIR),
            "evaluation_image_height": image_height,
            "evaluation_image_width": image_width,
            "hidden_dim": model_cfg["hidden_dim"],
            "drop_path": model_cfg["drop_path"],
            "head_dropout": model_cfg["head_dropout"],
            "num_test_samples": len(predictions_df),
            **metrics,
        }

        all_metrics.append(row)

        print(f"Test samples: {len(predictions_df)}")
        print(f"Test MAE: {metrics['mae']:.3f} months")
        print(f"Test RMSE: {metrics['rmse']:.3f} months")
        print(f"Test median AE: {metrics['median_ae']:.3f} months")
        print(f"Test mean error / bias: {metrics['mean_error_bias']:.3f} months")
        print(f"Predictions saved to: {predictions_path}")

        del model
        del test_loader
        del test_dataset

        torch.cuda.empty_cache()
        gc.collect()

    metrics_df = pd.DataFrame(all_metrics)

    metrics_csv_path = OUTPUT_DIR / "final_testset_model_metrics.csv"
    metrics_json_path = OUTPUT_DIR / "final_testset_model_metrics.json"

    metrics_df.to_csv(metrics_csv_path, index=False)

    with open(metrics_json_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("\n" + "=" * 80)
    print("Final test-set comparison")
    print("=" * 80)

    print(
        metrics_df[
            [
                "model",
                "evaluation_image_height",
                "evaluation_image_width",
                "num_test_samples",
                "mae",
                "rmse",
                "median_ae",
                "mean_error_bias",
            ]
        ].to_string(index=False)
    )

    print("\nReport summary")
    for row in all_metrics:
        print(
            f"{row['model']} "
            f"({row['evaluation_image_height']}x{row['evaluation_image_width']}): "
            f"MAE = {row['mae']:.3f} months, "
            f"RMSE = {row['rmse']:.3f} months, "
            f"Bias = {row['mean_error_bias']:.3f} months"
        )

    print("\nSaved outputs:")
    print(f"Metrics CSV: {metrics_csv_path}")
    print(f"Metrics JSON: {metrics_json_path}")


if __name__ == "__main__":
    main()