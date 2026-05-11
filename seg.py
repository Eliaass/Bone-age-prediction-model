import argparse 
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(_file_).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "images"
VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_inner_crop_bbox_two_pass(img: np.ndarray) -> tuple[int, int, int, int]:
    """
    Find the inner bounding box in two passes:
    1. Strip pure black scanner padding.
    2. Try to skip a thick gray bevel-like frame.
    """
    # Pass 1: remove black padding.
    mask_nonblack = (img > 5).astype(np.uint8)
    x1, y1, bw1, bh1 = cv2.boundingRect(mask_nonblack)
    
    if bw1 == 0 or bh1 == 0:
        return 0, 0, img.shape[1], img.shape[0]
        
    crop1 = img[y1:y1+bh1, x1:x1+bw1]
    
    # Pass 2: detect the inner area past the gray frame.
    h, w = crop1.shape
    th, _ = cv2.threshold(crop1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if th < 10:
        return x1, y1, x1 + bw1, y1 + bh1
        
    bg_mask = (crop1 < th).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bg_mask, connectivity=8)
    
    min_x, min_y = w, h
    max_x, max_y = 0, 0
    found_any = False
    margin = int(min(w, h) * 0.02)
    
    for j in range(1, n):
        area = stats[j, cv2.CC_STAT_AREA]
        # Ignore small noisy components.
        if area < w * h * 0.05: 
            continue
            
        bx = stats[j, cv2.CC_STAT_LEFT]
        by = stats[j, cv2.CC_STAT_TOP]
        bbw = stats[j, cv2.CC_STAT_WIDTH]
        bbh = stats[j, cv2.CC_STAT_HEIGHT]
        
        touches_border = (bx < margin) or (by < margin) or \
                         (bx + bbw > w - margin) or (by + bbh > h - margin)
        
        # Components away from the border are likely the background inside the frame.
        if not touches_border:
            min_x = min(min_x, bx)
            min_y = min(min_y, by)
            max_x = max(max_x, bx + bbw)
            max_y = max(max_y, by + bbh)
            found_any = True
            
    if found_any:
        pad = 5
        final_x1 = x1 + max(0, min_x - pad)
        final_y1 = y1 + max(0, min_y - pad)
        final_x2 = x1 + min(w, max_x + pad)
        final_y2 = y1 + min(h, max_y + pad)
        return final_x1, final_y1, final_x2, final_y2
    else:
        # Fall back to the first crop if no enclosed inner region was found.
        return x1, y1, x1 + bw1, y1 + bh1


def adaptive_multi_otsu(median_blur: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Split the histogram into three classes:
    1. Very dark pixels (borders or noise)
    2. Background
    3. Foreground (hand)
    Pick the threshold that best separates background from hand.
    """
    mask_valid = median_blur > 10
    if mask_valid.sum() == 0:
        return 0, np.zeros_like(median_blur)
    
    pixels = median_blur[mask_valid]
    hist, _ = np.histogram(pixels, bins=256, range=(0, 255))
    hist = hist.astype(float)
    total = hist.sum()
    sum_total = np.dot(np.arange(256), hist)
    
    maximum = 0.0
    th1, th2 = 0, 0
    w0, sum0 = 0.0, 0.0
    for i in range(254):
        w0 += hist[i]
        if w0 == 0: 
            continue
        sum0 += i * hist[i]
        m0 = sum0 / w0
        
        w1, sum1 = 0.0, 0.0
        for j in range(i + 1, 255):
            w1 += hist[j]
            if w1 == 0: 
                continue
            sum1 += j * hist[j]
            m1 = sum1 / w1
            
            w2 = total - w0 - w1
            if w2 <= 0: 
                break
            sum2 = sum_total - sum0 - sum1
            m2 = sum2 / w2
            
            between = w0 * (m0)*2 + w1 * (m1)2 + w2 * (m2)*2
            if between > maximum:
                maximum = between
                th1 = i
                th2 = j

    # Decide which threshold is the useful split.
    c0_size = hist[:th1 + 1].sum()
    # If the darkest class is small, treat it as border noise and use th2.
    if c0_size < total * 0.15: 
        chosen_th = th2
    else: 
        chosen_th = th1
        
    _, mask_high = cv2.threshold(median_blur, chosen_th, 255, cv2.THRESH_BINARY)
    return chosen_th, mask_high


def get_largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    largest_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (cc == largest_id).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes via contours instead of floodFill from (0, 0)."""
    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, -1)
    return (mask > 0).astype(np.uint8)


def save_gray(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(str(path))


def iter_image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )


def create_smooth_organic_mask(img: np.ndarray) -> np.ndarray:
    # 1. Tight inner crop with a two-pass box search.
    left, top, right, bottom = get_inner_crop_bbox_two_pass(img)
    top, bottom = max(0, top), min(img.shape[0], bottom)
    left, right = max(0, left), min(img.shape[1], right)
    
    crop = img[top:bottom, left:right].copy()
    if crop.size == 0:
        # Fall back to the full image if cropping fails.
        crop = img.copy()
        top, left = 0, 0
        bottom, right = img.shape

    # 2. Clear a thin outer band so scanner lines and edge junk do not
    # blur into a closed frame that confuses GrabCut.
    h_crop, w_crop = crop.shape
    margin_y = max(1, int(h_crop * 0.025))
    margin_x = max(1, int(w_crop * 0.025))
    crop[:margin_y, :] = 0
    crop[-margin_y:, :] = 0
    crop[:, :margin_x] = 0
    crop[:, -margin_x:] = 0

    # 3. Percentile clipping removes extreme outliers.
    p_low = np.percentile(crop, 3)
    p_high = np.percentile(crop, 97)
    clipped = np.clip(crop, p_low, p_high)

    # 4. Z-score normalization evens out contrast.
    mean_val = np.mean(clipped)
    std_val = np.std(clipped)
    
    if std_val == 0:
        img_norm = clipped.astype(np.uint8)
    else:
        z_scores = (clipped - mean_val) / (std_val + 1e-6)
        # Clamp z-scores from [-3, 3] into [0, 255].
        z_range = 3.0
        img_norm_float = (z_scores + z_range) / (2 * z_range) * 255.0
        img_norm = np.clip(img_norm_float, 0, 255).astype(np.uint8)

    # 5. Low-pass filtering smooths the normalized image.
    tiefpass = cv2.GaussianBlur(img_norm, (15, 15), 0)
    median_blur = cv2.medianBlur(tiefpass, 11)

    # 6. Adaptive multi-Otsu thresholding.
    th_high, mask_high = adaptive_multi_otsu(median_blur)

    sure_bg_mask = cv2.dilate(mask_high, np.ones((51, 51), np.uint8), iterations=2)
    sure_bg_pixels = median_blur[sure_bg_mask == 0]
    bg_max = np.percentile(sure_bg_pixels, 99) if sure_bg_pixels.size > 0 else 0
    th_low = max(bg_max + 1, th_high * 0.45)

    _, mask_low = cv2.threshold(median_blur, th_low, 255, cv2.THRESH_BINARY)

    current_mask = mask_high.copy()
    while True:
        dilated = cv2.dilate(current_mask, np.ones((3, 3), np.uint8))
        candidates = cv2.bitwise_and(mask_low, dilated)
        new_mask = cv2.bitwise_or(current_mask, candidates)
        if np.array_equal(current_mask, new_mask):
            break
        current_mask = new_mask

    # 7. Drop small border-touching blobs unless they are large enough to be the hand.
    h, w = current_mask.shape
    border_mask = np.zeros((h, w), dtype=np.uint8)
    border_mask[0:5, :] = 1
    border_mask[:, 0:5] = 1
    border_mask[:, -5:] = 1

    n, cc, _, _ = cv2.connectedComponentsWithStats(current_mask, connectivity=8)
    valid_mask = np.zeros_like(current_mask)
    for component_id in range(1, n):
        component = (cc == component_id).astype(np.uint8)
        # Keep components that stay off the border, or are large enough to be the hand.
        if not np.any(cv2.bitwise_and(component, border_mask)) or component.sum() > (h * w * 0.05):
            valid_mask = cv2.bitwise_or(valid_mask, component * 255)

    rough_mask = cv2.morphologyEx(
        valid_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    
    rough_mask = get_largest_component(rough_mask) * 255
    rough_mask = fill_holes(rough_mask) * 255

    snapped_mask = rough_mask.copy().astype(np.uint8)
    if np.any(rough_mask):
        img_color = cv2.cvtColor(median_blur, cv2.COLOR_GRAY2BGR)
        gc_mask = np.full(crop.shape, cv2.GC_BGD, dtype=np.uint8)

        dilated_rough = cv2.dilate(
            rough_mask, np.ones((11, 11), np.uint8), iterations=2
        )
        gc_mask[dilated_rough > 0] = cv2.GC_PR_BGD
        gc_mask[rough_mask > 0] = cv2.GC_PR_FGD

        sure_fg = cv2.erode(rough_mask, np.ones((7, 7), np.uint8), iterations=1)
        if np.any(sure_fg):
            gc_mask[sure_fg > 0] = cv2.GC_FGD

        has_background = np.any((gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD))
        has_foreground = np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD))
        if has_background and has_foreground:
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(
                    img_color,
                    gc_mask,
                    None,
                    bgd_model,
                    fgd_model,
                    3,
                    cv2.GC_INIT_WITH_MASK,
                )
                snapped_mask = np.where(
                    (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
                ).astype(np.uint8)
            except cv2.error:
                pass

    snapped_mask = get_largest_component(snapped_mask) * 255
    smooth_mask = cv2.GaussianBlur(snapped_mask, (31, 31), 0)
    _, smooth_mask = cv2.threshold(smooth_mask, 127, 255, cv2.THRESH_BINARY)

    # 8. Place the clean mask back onto the original canvas.
    full_mask = np.zeros_like(img, dtype=np.uint8)
    full_mask[top:bottom, left:right] = smooth_mask

    segmented = img.copy()
    segmented[full_mask == 0] = 0

    return segmented


def resolve_output_subdir(image_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative_path = image_path.relative_to(input_dir)
    parts = relative_path.parts[:-1]

    split_map = {
        "train": "training",
        "training": "training",
        "test": "testing",
        "testing": "testing",
    }
    split_index = None
    split_name = None

    for index, part in enumerate(parts):
        lower_part = part.lower()
        for key, mapped_name in split_map.items():
            if key in lower_part:
                split_index = index
                split_name = mapped_name

    if split_name is None:
        return output_dir / relative_path.parent

    remaining_parts = parts[split_index + 1 :]
    return output_dir / split_name / Path(*remaining_parts)


def process_image(image_path: Path, input_dir: Path, output_dir: Path) -> None:
    img = np.array(Image.open(image_path).convert("L"))
    segmented = create_smooth_organic_mask(img)
    image_output_dir = resolve_output_subdir(image_path, input_dir, output_dir)
    image_output_dir.mkdir(parents=True, exist_ok=True)
    save_gray(segmented, image_output_dir / image_path.name)


def build_output_dir(input_dir: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return input_dir.parent / f"{input_dir.name}_segmented"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segment all images in a folder tree and write only the segmented "
            "images into a separate output folder."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Root folder that contains the images to segment.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Destination root folder. Defaults to a sibling folder named "
            "'<input>_segmented', with training/testing subfolders when detected."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = build_output_dir(input_dir, args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")
    if output_dir == input_dir or output_dir.is_relative_to(input_dir):
        raise ValueError("Output folder must be outside the input folder.")

    image_files = iter_image_files(input_dir)
    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    failed = 0
    for image_path in image_files:
        try:
            process_image(image_path, input_dir, output_dir)
            processed += 1
            print(f"Processed: {image_path.relative_to(input_dir)}")
        except Exception as exc:
            failed += 1
            print(f"Failed: {image_path.relative_to(input_dir)} ({exc})")
        if processed == 3:
            break
    print(
        f"Done. Wrote {processed} image sets to {output_dir}"
        + (f" with {failed} failures." if failed else ".")
    )


if _name_ == "_main_":
    main()