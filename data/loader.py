import pandas as pd

def load_dataset(path: str):
    # TSV FIX (your dataset is tab-separated)
    df = pd.read_csv(path, sep="\t")

    # normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df