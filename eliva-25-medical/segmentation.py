from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "test"
OUTPUT_DIR = SCRIPT_DIR / "output"
VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    largest_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (cc == largest_id).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    flood = (mask * 255).copy()
    h, w = mask.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 128)
    holes = (flood == 0).astype(np.uint8)
    return ((mask | holes) > 0).astype(np.uint8)


def save_gray(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(str(path))


def iter_image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )


def create_smooth_organic_mask(
    img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top, left, right, bottom = 10, 10, 10, 40
    crop = img[top : img.shape[0] - bottom, left : img.shape[1] - right]
    if crop.size == 0:
        raise ValueError("Image is too small for the configured crop margins.")

    p5 = np.percentile(crop, 5)
    p95 = np.percentile(crop, 95)
    img_norm = (
        crop
        if p95 <= p5
        else ((np.clip(crop, p5, p95) - p5) / (p95 - p5) * 255).astype(np.uint8)
    )

    gamma = 0.5
    img_gamma = np.array(255 * (img_norm / 255) ** gamma, dtype=np.uint8)
    median_blur = cv2.medianBlur(img_gamma, 11)

    th_high, mask_high = cv2.threshold(
        median_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

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

    h, w = current_mask.shape
    border_mask = np.zeros((h, w), dtype=np.uint8)
    border_mask[0:5, :] = 1
    border_mask[:, 0:5] = 1
    border_mask[:, -5:] = 1

    n, cc, _, _ = cv2.connectedComponentsWithStats(current_mask, connectivity=8)
    valid_mask = np.zeros_like(current_mask)
    for component_id in range(1, n):
        component = (cc == component_id).astype(np.uint8)
        if not np.any(cv2.bitwise_and(component, border_mask)):
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

        has_background = np.any(
            (gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD)
        )
        has_foreground = np.any(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        )
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
                snapped_mask = rough_mask.copy().astype(np.uint8)

    snapped_mask = get_largest_component(snapped_mask) * 255
    smooth_mask = cv2.GaussianBlur(snapped_mask, (31, 31), 0)
    _, smooth_mask = cv2.threshold(smooth_mask, 127, 255, cv2.THRESH_BINARY)

    full_mask = np.zeros_like(img, dtype=np.uint8)
    full_mask[top : img.shape[0] - bottom, left : img.shape[1] - right] = smooth_mask

    segmented = img.copy()
    segmented[full_mask == 0] = 0
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(
        full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 100, 100), 4)

    return full_mask, segmented, overlay


def process_image(image_path: Path, output_dir: Path) -> None:
    img = np.array(Image.open(image_path).convert("L"))
    mask, segmented, overlay = create_smooth_organic_mask(img)
    stem = image_path.stem

    save_gray(mask, output_dir / f"{stem}_mask.png")
    save_gray(segmented, output_dir / f"{stem}_segmented.png")
    Image.fromarray(overlay).save(str(output_dir / f"{stem}_overlay.png"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_files = iter_image_files(INPUT_DIR)
    if not image_files:
        raise FileNotFoundError(f"No images found in {INPUT_DIR}")

    for image_path in image_files:
        process_image(image_path, OUTPUT_DIR)
        print(f"Processed: {image_path.name}")

    print(f"Done. Wrote {len(image_files)} image sets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
