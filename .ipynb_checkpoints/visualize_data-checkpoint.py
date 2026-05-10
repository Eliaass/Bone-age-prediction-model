"""Data visualization"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR / "tests-sara" / "data" / "boneage-training-dataset.csv"
TRAIN_IMG_DIR = BASE_DIR / "tests-sara" / "data" / "boneage-training-dataset"


def load_data(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def basic_stats(df: pd.DataFrame) -> None:
    print("=== BASIC STATS ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nDuplicate IDs:")
    print(df["id"].duplicated().sum())
    print("\nAge describe:")
    print(df["boneage"].describe())
    print("\nGender counts:")
    print(df["male"].value_counts())


def plot_age_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    sns.histplot(df["boneage"], bins=40, kde=True)
    plt.title("Bone Age Distribution")
    plt.xlabel("Bone age (months)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plot_age_distr.png")
    plt.show()


def plot_age_bins(df: pd.DataFrame) -> None:
    bins = [0, 60, 120, 180, 240, 300]
    labels = ["0-5y", "5-10y", "10-15y", "15-20y", "20-25y"]
    plot_df = df.copy()
    plot_df["age_bin"] = pd.cut(plot_df["boneage"], bins=bins, labels=labels, include_lowest=True)

    plt.figure(figsize=(10, 5))
    sns.countplot(data=plot_df, x="age_bin", order=labels)
    plt.title("Age Bins")
    plt.xlabel("Age bin")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plot_age_bins.png")
    plt.show()
    


def plot_age_by_gender(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["male"] = plot_df["male"].map({True: "Male", False: "Female"})

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=plot_df, x="male", y="boneage")
    plt.title("Bone Age by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Bone age (months)")
    plt.tight_layout()
    plt.savefig("plot_gender.png")
    plt.show()
    


def main() -> None:
    df = load_data(TRAIN_CSV)
    basic_stats(df)
    plot_age_distribution(df)
    plot_age_bins(df)
    plot_age_by_gender(df)


if __name__ == "__main__":
    main()