import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame):
        X = self._build_features(df)
        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True

        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def transform(self, df: pd.DataFrame):
        if not self.fitted:
            raise Exception("FeatureModel not fitted yet.")
        X = self._build_features(df)
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def _build_features(self, df):
        features = [
            "age_at_pd_1_therapy",
            "kps",
            "mutation_count",
            "tmb",
            "number_of_pd1_inhibitor_injections",
            "number_of_prior_recurrences"
        ]

        X = df[features].copy()
        return X.fillna(0)
