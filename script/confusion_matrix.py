import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings('ignore')

print("loading data...")
df = pd.read_csv('kepler_clean.csv')

# feature engineering - same as pipeline
df['depth_duration_ratio'] = df['koi_depth'] / df['koi_duration'].replace(0, np.nan)
df['prad_srad_ratio'] = df['koi_prad'] / df['koi_srad'].replace(0, np.nan)
df['snr_depth_ratio'] = df['koi_model_snr'] / df['koi_depth'].replace(0, np.nan)
engineered_cols = ['depth_duration_ratio', 'prad_srad_ratio', 'snr_depth_ratio']
for col in engineered_cols:
    df[col] = df[col].fillna(df[col].median())

all_cols = [c for c in df.columns if c != 'target']
uncertainty_cols = [c for c in all_cols if 'err1' in c or 'err2' in c]
base_cols = [c for c in all_cols if c not in uncertainty_cols and c not in engineered_cols]
final_features = base_cols + uncertainty_cols + engineered_cols

X = df[final_features]
y = df['target']

# 80/20 split, stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print("training XGBoost...")
model = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42, verbosity=0)
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['False Positive', 'Confirmed'])
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title("Confusion Matrix - XGBoost (Best Model)", fontsize=14)
plt.grid(False)
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

print("saved confusion_matrix.png")

tn, fp, fn, tp = cm.ravel()
print(f"\nTP (correctly confirmed):  {tp}")
print(f"TN (correctly rejected):   {tn}")
print(f"FP (incorrectly confirmed): {fp}")
print(f"FN (missed planets):        {fn}")
