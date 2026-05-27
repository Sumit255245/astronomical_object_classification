# SDSS — Step 1: Load & Inspect
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("star_classification.csv")
print(f"Shape: {df.shape}")
print(df['class'].value_counts())
print(df.isnull().sum())

# Drop metadata columns
drop_cols = ['obj_ID','run_ID','rerun_ID','cam_col','field_ID','spec_obj_ID']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Encode label
le = LabelEncoder()
df['label'] = le.fit_transform(df['class'])
print("Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Save
df.to_csv("sdss_clean.csv", index=False)
print("✅ Saved sdss_clean.csv")