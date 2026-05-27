# SDSS — Step 5: Hyperparameter Tuning (Random Forest)
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

with open("sdss_ready.pkl", "rb") as f:
    d = pickle.load(f)

X_train, y_train = d["X_train"], d["y_train"]
X_val,   y_val   = d["X_val"],   d["y_val"]
CLASS_NAMES      = d["class_names"]

# Baseline
baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
baseline.fit(X_train, y_train)
base_acc = accuracy_score(y_val, baseline.predict(X_val))
print(f"Baseline accuracy: {base_acc:.4f}")

# Search space
param_dist = {
    "n_estimators"      : [100, 200, 300, 500],
    "max_depth"         : [None, 10, 20, 30],
    "min_samples_split" : [2, 5, 10],
    "min_samples_leaf"  : [1, 2, 4],
    "max_features"      : ["sqrt", "log2", 0.5],
    "bootstrap"         : [True, False],
}

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=5, cv=3, scoring="accuracy",
    random_state=42, n_jobs=-1, verbose=2,
)

print("\nRunning search (takes 5–10 mins)...")
search.fit(X_train, y_train)

best = search.best_estimator_
tuned_acc = accuracy_score(y_val, best.predict(X_val))

print(f"\nBest params  : {search.best_params_}")
print(f"Tuned val acc: {tuned_acc:.4f}  (improvement: +{tuned_acc - base_acc:.4f})")
print(classification_report(y_val, best.predict(X_val), target_names=CLASS_NAMES))

with open("sdss_best_model.pkl", "wb") as f:
    pickle.dump({"model": best, "params": search.best_params_,
                 "val_accuracy": tuned_acc}, f)

print("✅ Saved sdss_best_model.pkl")