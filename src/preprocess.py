# SDSS — Step 3: Preprocessing & Feature Engineering
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("sdss_clean.csv")

# Drop non-predictive columns
drop_cols = ['alpha', 'delta', 'ra', 'dec', 'run_ID', 'rerun_ID', 'cam_col',
             'field_ID', 'spec_obj_ID', 'obj_ID', 'plate', 'MJD', 'fiber_ID']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Engineer colour index features
df['u_g'] = df['u'] - df['g']
df['g_r'] = df['g'] - df['r']
df['r_i'] = df['r'] - df['i']
df['i_z'] = df['i'] - df['z']

print("Features:", list(df.columns))

# Encode target
le = LabelEncoder()
df['label'] = le.fit_transform(df['class'])
print("Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Split
X = df.drop(columns=['class', 'label'])
y = df['label']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# Save
data = {
    "X_train": X_train_sc, "X_val": X_val_sc, "X_test": X_test_sc,
    "y_train": y_train,    "y_val": y_val,     "y_test": y_test,
    "feature_names": list(X.columns),
    "class_names": list(le.classes_),
    "scaler": scaler, "label_encoder": le,
}
print("Features saved:", list(X.columns))
with open("sdss_ready.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ Saved sdss_ready.pkl")