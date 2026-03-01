import pandas as pd
import numpy as np

df = pd.read_csv("cumulative.csv")

# only keep rows we care about, drop candidates
df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]

# drop leakage cols - these are NASAs own output labels, cant use them for prediction
leakage_cols = [
    "koi_score", "koi_pdisposition",
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"
]
df = df.drop(columns=[c for c in leakage_cols if c in df.columns])

# drop id/metadata cols, not useful for ML
meta_cols = ["kepid", "kepoi_name", "kepler_name", "koi_tce_delivname", "rowid"]
df = df.drop(columns=[c for c in meta_cols if c in df.columns])

# these two are 100% NaN in the dataset
df = df.drop(columns=["koi_teq_err1", "koi_teq_err2"], errors='ignore')

# encode target col
df["target"] = (df["koi_disposition"] == "CONFIRMED").astype(int)
df = df.drop(columns=["koi_disposition"])

# only keep numeric
df = df.select_dtypes(include="number")

# fill NaN with median
df = df.fillna(df.median())

df.to_csv("kepler_clean.csv", index=False)

print(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")
print(f"\nClass distribution:\n{df['target'].value_counts().to_string()}")
print(f"\nNaN remaining: {df.isna().sum().sum()}")
print(f"\nColumns ({len(df.columns)}):")
for col in df.columns:
    print(f"  {col}")
