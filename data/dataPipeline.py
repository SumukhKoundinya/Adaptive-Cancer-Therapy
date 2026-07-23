import sys
import os

# FIX PATH ISSUE (CRITICAL on Windows)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_dataset
from data.preprocess import clean_dataset


FEATURES = [
    "age_at_pd1",
    "tmb",
    "kps",
    "mutation_count",
    "response_encoded"
]


def run_pipeline():

    df = load_dataset("data/gbm_columbia_2019_clinical_data.tsv")

    df_clean = clean_dataset(df)

    print("CLEAN SHAPE:", df_clean.shape)
    print(df_clean.head())

    X = df_clean[[c for c in FEATURES if c in df_clean.columns]]
    y = df_clean["os_months"]

    print("\nFEATURES USED:", X.columns.tolist())
    print("\nTARGET READY:", y.head())

    # -------------------------
    # SAVE OUTPUT (ADD THIS)
    # -------------------------
    output_path = "data/processed_dataset.csv"
    df_clean.to_csv(output_path, index=False)

    print(f"\n✅ Saved processed dataset to: {output_path}")


if __name__ == "__main__":
    run_pipeline()