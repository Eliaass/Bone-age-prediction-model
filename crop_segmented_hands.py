import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "overlayed_RSNA_testset"
OUTPUT_DIR = "cropped_overlayed_RSNA_testset_1024x1024"

OUTPUT_SIZE = 1024         # Final output image size (square)
THRESHOLD = 5              # Pixel values > threshold are considered foreground
MARGIN = 0.01              # 1% margin around the hand bounding box
SAVE_FORMAT = None         # None = keep original extension, or use "png"/"jpg"


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def find_foreground_bbox(image: Image.Image, threshold: int = 5):
    """
    Find bounding box of non-background pixels in a masked X-ray image.

    Assumption:
    - background is black or near-black
    - hand region is brighter than the background

    Returns:
        (x_min, y_min, x_max, y_max) or None if no foreground is found.
    """

    gray = image.convert("L")
    arr = np.array(gray)

    foreground = arr > threshold

    if not foreground.any():
        return None

    ys, xs = np.where(foreground)

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return x_min, y_min, x_max, y_max


def add_margin_to_bbox(bbox, image_width: int, image_height: int, margin: float = 0.10):
    """
    Expand bounding box by a relative margin.
    """

    x_min, y_min, x_max, y_max = bbox

    box_width = x_max - x_min + 1
    box_height = y_max - y_min + 1

    margin_x = int(box_width * margin)
    margin_y = int(box_height * margin)

    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(image_width - 1, x_max + margin_x)
    y_max = min(image_height - 1, y_max + margin_y)

    return x_min, y_min, x_max, y_max


def pad_to_square(image: Image.Image, fill=0):
    """
    Pad image to a square canvas while preserving aspect ratio.
    """

    width, height = image.size

    if width == height:
        return image

    max_side = max(width, height)

    pad_left = (max_side - width) // 2
    pad_right = max_side - width - pad_left
    pad_top = (max_side - height) // 2
    pad_bottom = max_side - height - pad_top

    return ImageOps.expand(
        image,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=fill
    )


def crop_pad_resize_image(
    image: Image.Image,
    output_size: int = 512,
    threshold: int = 5,
    margin: float = 0.10
):
    """
    Full preprocessing pipeline:
    1. Find hand bounding box
    2. Add margin
    3. Crop image
    4. Pad to square
    5. Resize to target size
    """

    bbox = find_foreground_bbox(image, threshold=threshold)

    if bbox is None:
        # Fallback: keep original image if no foreground was found
        cropped = image
    else:
        bbox = add_margin_to_bbox(
            bbox=bbox,
            image_width=image.width,
            image_height=image.height,
            margin=margin
        )

        x_min, y_min, x_max, y_max = bbox
        cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1))

    padded = pad_to_square(cropped, fill=0)

    resized = padded.resize(
        (output_size, output_size),
        resample=Image.BILINEAR
    )

    return resized


def is_image_file(path: Path):
    return path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in input_dir.iterdir() if p.is_file() and is_image_file(p)]

    print(f"Found {len(image_paths)} images in:")
    print(input_dir)
    print()
    print(f"Saving cropped images to:")
    print(output_dir)
    print()

    failed_images = []

    for image_path in tqdm(image_paths, desc="Cropping images"):
        try:
            image = Image.open(image_path)

            # Keep original mode if possible.
            # If your images are grayscale, they remain grayscale.
            # If they are RGB, they remain RGB.
            processed = crop_pad_resize_image(
                image=image,
                output_size=OUTPUT_SIZE,
                threshold=THRESHOLD,
                margin=MARGIN
            )

            if SAVE_FORMAT is None:
                output_path = output_dir / image_path.name
                processed.save(output_path)
            else:
                output_path = output_dir / f"{image_path.stem}.{SAVE_FORMAT.lower()}"
                processed.save(output_path, format=SAVE_FORMAT.upper())

        except Exception as e:
            failed_images.append((str(image_path), str(e)))

    print()
    print("Done.")
    print(f"Successfully processed: {len(image_paths) - len(failed_images)}")
    print(f"Failed: {len(failed_images)}")

    if failed_images:
        print()
        print("Failed images:")
        for path, error in failed_images[:20]:
            print(f"{path} -> {error}")

        if len(failed_images) > 20:
            print(f"... and {len(failed_images) - 20} more.")


if __name__ == "__main__":
    main()