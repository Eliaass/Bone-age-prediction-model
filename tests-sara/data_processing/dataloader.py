import kagglehub
from os import path, listdir

import pandas as pd
from sklearn.model_selection import train_test_split


def get_BAA_DFS(datapath = None, imgs_segmented = False):

    # 1) get RSNA dataset: thru kagglehub if not found in folders
    df_path = path.join(datapath, "boneage-training-dataset.csv")
    img_path = path.join(datapath, "images")
    if imgs_segmented:
        img_path = path.join(datapath, "segmented")
    
    if datapath is None or not path.exists(img_path):
        datapath = kagglehub.dataset_download("kmader/rsna-bone-age")
        print(f"Path to dataset files: {datapath}")

        img_path = path.join(datapath, "boneage-training-dataset", "boneage-training-dataset")
        df_path = path.join(datapath, "boneage-training-dataset.csv")

    # 2) convert to a DF    
    df = pd.read_csv(df_path, delimiter=",")

    # 3) convert feature to float
    df["male"].replace([True, False], [1.0, 0.0], inplace=True)

    # 4) add image path to df
    df["path"] = df["id"].apply(lambda id: path.join(img_path, f"{id}.png"))

    # 5) train/val split
    train_df, val_df = train_test_split(df, test_size = 0.2, random_state = 42)
    print(f"\nFull df size len: {df.shape[0]}\ntrain/val sizes: {train_df.shape[0]}/{val_df.shape[0]}")

    return train_df, val_df