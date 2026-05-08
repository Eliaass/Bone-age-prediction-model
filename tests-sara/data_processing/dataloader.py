
from os import path, listdir

import pandas as pd
from sklearn.model_selection import train_test_split

from PIL import Image
import numpy as np


def get_BAA_DFS(seed, datapath = None, image_folder = "images", apply_segmentation = False):

    # 1) get RSNA dataset: thru kagglehub if not found in folders
    df_path = path.join(datapath, "boneage-training-dataset.csv")
    img_path = path.join(datapath, image_folder)

    
    if datapath is None or not path.exists(img_path):

        print(f"download the rsna dataset at: {img_path}")
        return
        """
        datapath = kagglehub.dataset_download("kmader/rsna-bone-age")
        print(f"Path to dataset files: {datapath}")

        img_path = path.join(datapath, "boneage-training-dataset", "boneage-training-dataset")
        df_path = path.join(datapath, "boneage-training-dataset.csv")
        """

    # 2) convert to a DF    
    df = pd.read_csv(df_path, delimiter=",")

    # 3) convert feature to float
    df["male"] = df["male"].astype(float)

    # 4) add image and mask path to df: if it leads to no file, get rid of row
    if apply_segmentation:
        df["img_path"] = df["id"].apply(lambda id: path.join(img_path, f"{id}_segmented.png"))
        df["mask_path"] = df["id"].apply(lambda id: path.join(img_path, f"{id}_mask.png"))

        df = df[ df["mask_path"].map(has_hand_mask) ]

    else:
        df["img_path"] = df["id"].apply(lambda id: path.join(img_path, f"{id}.png"))
        df["mask_path"] = None

    # 4.5) if img is mostly blank, remove
    print("finished checking")

    # 5) train/val split
    train_df, val_df = train_test_split(df, test_size = 0.2, random_state = seed)
    print(f"\nFull df size len: {df.shape[0]}\ntrain/val sizes: {train_df.shape[0]}/{val_df.shape[0]}")

    return train_df, val_df

def has_hand_mask(mask_p):
    if not path.exists (mask_p):
        return False
    
    mask = np.array(Image.open(mask_p).convert("L"))
    wcount = np.count_nonzero(mask)
    wratio = wcount / (mask.shape[0] * mask.shape[1])
    return wratio > 0.075 and wratio < 0.5