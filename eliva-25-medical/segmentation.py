from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.segmentation import morphological_chan_vese


# ============================================================
# INPUT / OUTPUT
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_IMAGE = SCRIPT_DIR / "test" / "1.png"
OUT_DIR = SCRIPT_DIR / "try_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def largest_component(mask: np.ndarray) -> np.ndarray:
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


def clean_binary_mask(mask: np.ndarray, close_kernel=(5, 5), open_kernel=(3, 3), close_iter=2, open_iter=1) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel),
        iterations=close_iter,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel),
        iterations=open_iter,
    )
    return mask


def select_best_contour_from_mask(ridge_mask: np.ndarray, target_mask: np.ndarray, min_area_ratio: float):
    contours, _ = cv2.findContours(ridge_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    ys, xs = np.where(target_mask > 0)
    if xs.size == 0:
        return None

    cx = float(xs.mean())
    cy = float(ys.mean())
    base_area = float(target_mask.sum())

    best_score = -1.0
    best_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * base_area:
            continue
        if cv2.pointPolygonTest(cnt, (cx, cy), False) < 0:
            continue

        tmp = np.zeros_like(target_mask, dtype=np.uint8)
        cv2.drawContours(tmp, [cnt], -1, 1, thickness=-1)
        tmp = fill_holes(tmp)

        overlap = float((tmp & target_mask).sum()) / float(target_mask.sum() + 1e-6)
        score = overlap * 2.0 + np.sqrt(area / (base_area + 1e-6))

        if score > best_score:
            best_score = score
            best_contour = cnt

    return best_contour


# ============================================================
# STEP 1
# CREATE base_mask = uploaded_1_mask_best_clahe.png
# ============================================================
def create_best_clahe_base_mask(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Same setup as before for image 1
    top, left, right, bottom = 10, 10, 10, 40
    crop = img[top : img.shape[0] - bottom, left : img.shape[1] - right]

    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(crop)

    alpha = 0.20
    blend = cv2.addWeighted(
        crop.astype(np.float32), 1.0 - alpha,
        clahe_img.astype(np.float32), alpha,
        0.0,
    )
    blend = np.clip(blend, 0, 255).astype(np.uint8)

    blur = cv2.GaussianBlur(blend, (0, 0), sigmaX=3, sigmaY=3)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask_crop = (th > 0).astype(np.uint8)
    mask_crop = largest_component(mask_crop)
    mask_crop = fill_holes(mask_crop)
    mask_crop = cv2.morphologyEx(
        mask_crop,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    mask_crop = cv2.morphologyEx(
        mask_crop,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    mask_crop = largest_component(mask_crop)
    mask_crop = fill_holes(mask_crop)

    full_mask = np.zeros_like(img, dtype=np.uint8)
    full_mask[top : img.shape[0] - bottom, left : img.shape[1] - right] = mask_crop * 255

    segmented = img.copy()
    segmented[full_mask == 0] = 0

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours((full_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    return full_mask, segmented, overlay


# ============================================================
# STEP 2
# PLAYABLE background-corrected + CLAHE + Gaussian + ridge-thr
# ============================================================
def run_ridge_pipeline(img: np.ndarray, base_mask: np.ndarray):
    base_bin = (base_mask > 0).astype(np.uint8)

    # Parameters to play with
    top, left, right, bottom = 10, 10, 10, 40

    bg_sigma = 35
    clahe_clip = 1.0
    clahe_grid = (8, 8)
    gauss_sigma = 2.0

    outer_kernel = (41, 41)
    inner_kernel = (23, 23)

    ridge_percentile = 72

    close_kernel = (7, 7)
    close_iter = 2

    open_kernel = (3, 3)
    open_iter = 1

    dilate_kernel = (3, 3)
    dilate_iter = 1

    min_area_ratio = 0.35

    crop = img[top : img.shape[0] - bottom, left : img.shape[1] - right]
    mask_crop = base_bin[top : img.shape[0] - bottom, left : img.shape[1] - right]

    # Background corrected + CLAHE + Gaussian
    bg = cv2.GaussianBlur(crop, (0, 0), sigmaX=bg_sigma, sigmaY=bg_sigma)
    darkness = cv2.subtract(bg, crop)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    darkness_eq = clahe.apply(darkness)

    dark_blur = cv2.GaussianBlur(
        darkness_eq,
        (0, 0),
        sigmaX=gauss_sigma,
        sigmaY=gauss_sigma,
    )

    # Ring around current contour
    outer = cv2.dilate(
        mask_crop,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, outer_kernel),
        iterations=1,
    )
    inner = cv2.erode(
        mask_crop,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, inner_kernel),
        iterations=1,
    )
    ring = ((outer > 0) & (inner == 0)).astype(np.uint8)

    # Ridge threshold inside ring
    vals = dark_blur[ring > 0]
    thr = float(np.percentile(vals, ridge_percentile))

    ridge = np.zeros_like(dark_blur, dtype=np.uint8)
    ridge[(dark_blur >= thr) & (ring > 0)] = 1
    ridge = clean_binary_mask(ridge, close_kernel=close_kernel, open_kernel=open_kernel, close_iter=close_iter, open_iter=open_iter)
    ridge = cv2.dilate(
        ridge,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilate_kernel),
        iterations=dilate_iter,
    )

    # Pick contour that best matches the hand
    picked_contour = select_best_contour_from_mask(ridge, mask_crop, min_area_ratio)

    refined_crop = mask_crop.copy()
    if picked_contour is not None:
        refined_crop = np.zeros_like(mask_crop, dtype=np.uint8)
        cv2.drawContours(refined_crop, [picked_contour], -1, 1, thickness=-1)
        refined_crop = fill_holes(refined_crop)

    refined_crop = largest_component(refined_crop)
    refined_crop = fill_holes(refined_crop)
    refined_crop = cv2.morphologyEx(
        refined_crop,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    refined_crop = largest_component(refined_crop)
    refined_crop = fill_holes(refined_crop)

    new_mask = np.zeros_like(img, dtype=np.uint8)
    new_mask[top : img.shape[0] - bottom, left : img.shape[1] - right] = refined_crop * 255

    segmented = img.copy()
    segmented[new_mask == 0] = 0

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    old_contours, _ = cv2.findContours(base_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_contours, _ = cv2.findContours((new_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, old_contours, -1, (180, 180, 180), 1)
    cv2.drawContours(overlay, new_contours, -1, (255, 255, 255), 2)

    ring_vis = (ring * 255).astype(np.uint8)
    ridge_vis = (ridge * 255).astype(np.uint8)

    return {
        "dark_blur": dark_blur,
        "ring_vis": ring_vis,
        "ridge_vis": ridge_vis,
        "new_mask": new_mask,
        "segmented": segmented,
        "overlay": overlay,
        "thr": thr,
        "picked_contour_found": picked_contour is not None,
    }


def run_levelset_refinement(img: np.ndarray, base_mask: np.ndarray, iterations: int = 10):
    image_float = img_as_float(img)
    init_level_set = (base_mask > 0).astype(np.uint8)
    levelset = morphological_chan_vese(
        image_float,
        num_iter=iterations,
        init_level_set=init_level_set,
        smoothing=4,
        lambda1=4,
        lambda2=10.0,
    )
    refined = levelset.astype(np.uint8)
    refined = largest_component(refined)
    refined = fill_holes(refined)
    refined = clean_binary_mask(refined, close_kernel=(5, 5), open_kernel=(5, 5), close_iter=2, open_iter=1)
    refined = largest_component(refined)
    refined = fill_holes(refined)

    new_mask = np.zeros_like(img, dtype=np.uint8)
    new_mask[refined > 0] = 255

    segmented = img.copy()
    segmented[new_mask == 0] = 0

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    return {
        "levelset_mask": new_mask,
        "segmented": segmented,
        "overlay": overlay,
        "refined_binary": refined,
    }

# ============================================================
# MAIN
# ============================================================
img = np.array(Image.open(INPUT_IMAGE).convert("L"))

# Create the base mask first
base_mask, base_segmented, base_overlay = create_best_clahe_base_mask(img)

# Save it exactly like before
save_gray(base_mask, OUT_DIR / "uploaded_1_mask_best_clahe.png")
save_gray(base_segmented, OUT_DIR / "uploaded_1_segmented_best_clahe.png")
Image.fromarray(base_overlay).save(str(OUT_DIR / "uploaded_1_overlay_best_clahe.png"))

# Run the ridge pipeline using the generated base mask
result = run_ridge_pipeline(img, base_mask)

save_gray(result["new_mask"], OUT_DIR / "mask_play.png")
save_gray(result["segmented"], OUT_DIR / "segmented_play.png")
Image.fromarray(result["overlay"]).save(str(OUT_DIR / "overlay_play.png"))
save_gray(result["ring_vis"], OUT_DIR / "ring_play.png")
save_gray(result["ridge_vis"], OUT_DIR / "ridge_play.png")

# Run level-set refinement using the base mask
levelset_result = run_levelset_refinement(img, base_mask, iterations=200)

Image.fromarray(levelset_result["overlay"]).save(str(OUT_DIR / "overlay_levelset.png"))

# Show everything
fig = plt.figure(figsize=(12, 18))

ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(img, cmap="gray")
ax1.set_title("Original")
ax1.axis("off")

ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(result["dark_blur"], cmap="gray")
ax2.set_title("Background corrected + CLAHE + Gaussian")
ax2.axis("off")

ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(result["ridge_vis"], cmap="gray")
ax3.set_title(f"Ridge, thr={result['thr']:.1f}")
ax3.axis("off")

ax4 = fig.add_subplot(2, 3, 4)
ax4.imshow(base_mask, cmap="gray")
ax4.set_title("Base mask")
ax4.axis("off")

ax5 = fig.add_subplot(2, 3, 5)
ax5.imshow(result["new_mask"], cmap="gray")
ax5.set_title("New mask")
ax5.axis("off")

ax6 = fig.add_subplot(2, 3, 6)
ax6.imshow(result["overlay"])
ax6.set_title("Gray = base, white = new")
ax6.axis("off")

fig.tight_layout()
plt.show()

fig2 = plt.figure(figsize=(6, 6))
ax1 = fig2.add_subplot(1, 1, 1)
ax1.imshow(levelset_result["overlay"])
ax1.set_title("Level-set overlay")
ax1.axis("off")

fig2.tight_layout()
plt.show()

print("Saved:")
print(OUT_DIR / "uploaded_1_mask_best_clahe.png")
print(OUT_DIR / "uploaded_1_segmented_best_clahe.png")
print(OUT_DIR / "uploaded_1_overlay_best_clahe.png")
print(OUT_DIR / "mask_play.png")
print(OUT_DIR / "segmented_play.png")
print(OUT_DIR / "overlay_play.png")
print(OUT_DIR / "ring_play.png")
print(OUT_DIR / "ridge_play.png")
print(OUT_DIR / "overlay_levelset.png")
print("thr =", result["thr"])
print("picked_contour_found =", result["picked_contour_found"])