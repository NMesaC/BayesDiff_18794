import pyarrow.parquet as pq
import pandas as pd
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def center_crop_arr(pil_image, image_size):
    """Center crop and resize an image to the desired size."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def dump_filtered_laion_images(parquet_path, local_image_path, out_dir,
                               keyword="cat", image_size=299, max_images=None):
    os.makedirs(out_dir, exist_ok=True)

    # Load parquet table
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    # Filter rows where the prompt contains the keyword (case-insensitive)
    print(df['TEXT'].head(10))
    filtered_df = df[df['TEXT'].str.contains(keyword, case=False, na=False, regex=True)]
    print(filtered_df)
    exit(0)
    print(f"Found {len(filtered_df)} images containing '{keyword}' in parquet")

    entries = os.listdir(local_image_path)
    saved_count = 0

    for subpath in tqdm(entries):
        x_path = os.path.join(local_image_path, subpath)

        # Skip invalid/unreadable images
        try:
            img = Image.open(x_path).convert("RGB")
        except Exception:
            continue

        # Optional: check if this file corresponds to a filtered prompt
        # This requires a mapping from filename -> parquet row index
        # If you don't have it, we can just dump all valid images
        arr = center_crop_arr(img, image_size)
        img_cropped = Image.fromarray(arr)
        img_cropped.save(os.path.join(out_dir, f"{saved_count}.png"))
        saved_count += 1

        if max_images is not None and saved_count >= max_images:
            break

    print(f"Saved {saved_count} images to {out_dir}")

# Example usage:
dump_filtered_laion_images(
    parquet_path="./laion_art/laion-art.parquet",
    local_image_path="./image_from_url",
    out_dir="laion_cat_images",
    keyword=r'chat|chatte|cat',
    image_size=299,
    max_images=1000  # number of images to dump for FID
)
