"""
Overlay Masks Script
Matches X-Ray images with tensor masks and creates masked images for training.

Structure:
- X-Ray images: archive/boneage-training-dataset/*.png (12,611 images)
- Tensor masks: Tensormask/*.png (14,208 masks)
- Only matched pairs are used for training

The script:
1. Finds all matching image-mask pairs by ID
2. Applies the mask directly to the X-Ray image
3. Saves masked images to overlayed_dataset/
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
XRAY_DIR = SCRIPT_DIR / "archive" / "boneage-training-dataset"
MASK_DIR = SCRIPT_DIR / "Tensormask"
OUTPUT_DIR = SCRIPT_DIR / "overlayed_RSNA_dataset"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_image_grayscale(path: Path) -> np.ndarray:
    """Load image as grayscale"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img


def apply_mask_to_image(xray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a tensor mask directly to an X-Ray image.
    
    Args:
        xray: Grayscale X-Ray image (H, W)
        mask: Grayscale mask image (H, W)
    
    Returns:
        Masked image (H, W) in grayscale
    """
    xray = np.clip(xray, 0, 255).astype(np.uint8)
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    
    # Convert any non-zero mask values to 1, keep background 0
    binary_mask = (mask > 0).astype(np.uint8)
    masked = xray * binary_mask
    return masked


def process_pair(img_id: int, xray_dir: Path, mask_dir: Path, output_dir: Path) -> tuple[bool, str]:
    """
    Process a single image-mask pair
    
    Returns:
        (success: bool, message: str)
    """
    xray_path = xray_dir / f"{img_id}.png"
    mask_path = mask_dir / f"{img_id}.png"
    output_path = output_dir / f"{img_id}.png"
    
    # Check if both files exist
    if not xray_path.exists():
        return False, f"X-Ray image not found: {img_id}"
    if not mask_path.exists():
        return False, f"Mask not found: {img_id}"
    
    try:
        # Load images
        xray = load_image_grayscale(xray_path)
        mask = load_image_grayscale(mask_path)
        
        # Apply mask
        masked = apply_mask_to_image(xray, mask)

        # Save masked image
        cv2.imwrite(str(output_path), masked)
        
        return True, f"Masked: {img_id}"
    
    except Exception as e:
        return False, f"Error processing {img_id}: {str(e)}"


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*70)
    print("Starting X-Ray to Mask Matching Pipeline")
    print("="*70)
    
    # Find all X-Ray images
    xray_files = sorted(list(XRAY_DIR.glob("*.png")))
    xray_ids = set([int(f.stem) for f in xray_files])
    
    # Find all mask files
    mask_files = sorted(list(MASK_DIR.glob("*.png")))
    mask_ids = set([int(f.stem) for f in mask_files])
    
    print(f"\nDataset statistics:")
    print(f"  X-Ray images found: {len(xray_ids)}")
    print(f"  Tensor masks found: {len(mask_ids)}")
    
    # Find matching pairs
    matching_ids = xray_ids & mask_ids  # Intersection
    only_xray = xray_ids - mask_ids
    only_mask = mask_ids - xray_ids
    
    print(f"\nMatching analysis:")
    print(f"  Matching pairs: {len(matching_ids)}")
    print(f"  X-Ray only (no mask): {len(only_xray)}")
    print(f"  Mask only (no X-Ray): {len(only_mask)}")
    
    if len(matching_ids) == 0:
        print("No matching image-mask pairs found!")
        return
    
    # Sort matching IDs
    matching_ids = sorted(list(matching_ids))
    
    # Process all matching pairs
    print(f"\nProcessing {len(matching_ids)} matching pairs...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    success_count = 0
    failure_count = 0
    failed_ids = []
    
    for img_id in tqdm(matching_ids, desc="Applying masks"):
        success, message = process_pair(img_id, XRAY_DIR, MASK_DIR, OUTPUT_DIR)
        
        if success:
            success_count += 1
        else:
            failure_count += 1
            failed_ids.append((img_id, message))
            if failure_count <= 5:
                print(f"Warning: {message}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Processing complete!")
    print(f"{'='*70}")
    print(f"Successfully masked: {success_count}")
    print(f"Failed: {failure_count}")
    
    if failed_ids:
        print(f"\nFirst failures (showing {min(5, len(failed_ids))}):")
        for img_id, msg in failed_ids[:5]:
            print(f"  - {msg}")

    print(f"\nSummary statistics:")
    print(f"  total_xray_images: {len(xray_ids)}")
    print(f"  total_masks: {len(mask_ids)}")
    print(f"  matching_pairs: {len(matching_ids)}")
    print(f"  successfully_masked: {success_count}")
    print(f"  failed: {failure_count}")

    print(f"\nSaved masked images to: {OUTPUT_DIR}")
    print("Ready for training!")


if __name__ == "__main__":
    main()