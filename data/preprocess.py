import numpy as np
import pandas as pd
from data.encoders import encode_response, encode_yesno


def clean_dataset(df):

    df = df.replace(["NA", "na", "N/A", ""], np.nan)

    # -------------------------
    # SAFE COLUMN HANDLING
    # -------------------------

    def safe_numeric(col):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Age FIX (your real column name is different)
    if "age_at_pd1_therapy" in df.columns:
        df["age_at_pd1"] = pd.to_numeric(df["age_at_pd1_therapy"], errors="coerce")

    safe_numeric("tmb_(nonsynonymous)")
    df["tmb"] = df.get("tmb_(nonsynonymous)")

    safe_numeric("kps")

    # Mutation count FIX (column exists sometimes as "mutation_count")
    if "mutation_count" in df.columns:
        df["mutation_count"] = pd.to_numeric(df["mutation_count"], errors="coerce")

    # -------------------------
    # TARGETS
    # -------------------------
    # Response label — accept multiple possible column names from different preprocessing steps
    if "response_to_pd1_inhibitor_(rano)" in df.columns:
        df["response_encoded"] = df["response_to_pd1_inhibitor_(rano)"].apply(encode_response)
    elif "best_response_to_pd1_inhibitor_(rano)" in df.columns:
        # some exported datasets use 'best_response...' header
        df["response_encoded"] = df["best_response_to_pd1_inhibitor_(rano)"].apply(encode_response)

    if "overall_survival_from_pd1i_(months)" in df.columns:
        df["os_months"] = pd.to_numeric(df["overall_survival_from_pd1i_(months)"], errors="coerce")

    if "progress_free_survival_(months)" in df.columns:
        df["pfs_months"] = pd.to_numeric(df["progress_free_survival_(months)"], errors="coerce")

    # -------------------------
    # BINARY FEATURES
    # -------------------------
    binary_cols = [
        "bev_failure_before_pd1_inhibitor",
        "liver_disfunction",
        "treatment_ongoing"
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].apply(encode_yesno)

    # -------------------------
    # DROP IRRELEVANT IDS
    # -------------------------
    drop_cols = [
        "study_id", "patient_id", "sample_id",
        "cancer_type_detailed", "tumor_location"
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # -------------------------
    # CLEAN FINAL
    # -------------------------
    df = df.dropna(subset=["os_months"])

    return df