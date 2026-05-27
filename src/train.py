# SDSS — Step 4: Train & Compare 8 ML Models
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

with open("sdss_ready.pkl", "rb") as f:
    d = pickle.load(f)

X_train, y_train = d["X_train"], d["y_train"]
X_val,   y_val   = d["X_val"],   d["y_val"]
CLASS_NAMES      = d["class_names"]

models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes"         : GaussianNB(),
    "Decision Tree"       : DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=5),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost"             : XGBClassifier(n_estimators=100, random_state=42,
                                          eval_metric="mlogloss", verbosity=0),
}

results = []
trained = {}

print(f"{'Model':<22} {'Val Acc':>8} {'CV Mean':>8} {'CV Std':>8}")
print("─" * 52)

for name, model in models.items():
    model.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"{name:<22} {val_acc:>8.4f} {cv.mean():>8.4f} {cv.std():>8.4f}")
    results.append({"Model": name, "Val Accuracy": val_acc,
                    "CV Mean": cv.mean(), "CV Std": cv.std()})
    trained[name] = model

results_df = pd.DataFrame(results).sort_values("Val Accuracy", ascending=False)
best_name  = results_df.iloc[0]["Model"]
best_model = trained[best_name]

print(f"\n✅ Best model: {best_name}")
print(classification_report(y_val, best_model.predict(X_val), target_names=CLASS_NAMES))

# Plot
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#2ecc71" if m == best_name else "#4C72B0" for m in results_df["Model"]]
bars = ax.barh(results_df["Model"], results_df["Val Accuracy"],
               color=colors, edgecolor="white")
ax.errorbar(results_df["CV Mean"], results_df["Model"],
            xerr=results_df["CV Std"], fmt="none",
            color="gray", capsize=4, linewidth=1.2)
ax.set_xlim(0.75, 1.01)
ax.set_title("Model Comparison", fontweight="bold")
for bar, val in zip(bars, results_df["Val Accuracy"]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("step4_model_comparison.png", dpi=150)
plt.show()

with open("sdss_models.pkl", "wb") as f:
    pickle.dump({"trained": trained, "results": results_df, "best_name": best_name}, f)

print("✅ Saved sdss_models.pkl")
print("Next: Step 5 — Tuning", best_name)