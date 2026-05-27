# SDSS — Step 6: Final Evaluation
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import joblib

sns.set_theme(style="whitegrid")

with open("sdss_ready.pkl", "rb") as f:
    d = pickle.load(f)
with open("sdss_best_model.pkl", "rb") as f:
    b = pickle.load(f)

X_test, y_test  = d["X_test"],  d["y_test"]
CLASS_NAMES     = d["class_names"]
FEATURE_NAMES   = d["feature_names"]
model           = b["model"]

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)
acc     = accuracy_score(y_test, y_pred)

print(f"Final Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# 1. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, cbar_kws={"label": "%"})
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix (%)", fontweight="bold")
plt.tight_layout()
plt.savefig("step6_confusion_matrix.png", dpi=150)
plt.show()

# 2. ROC curves
y_bin = label_binarize(y_test, classes=[0, 1, 2])
COLORS = ['#4C72B0', '#DD8452', '#55A868']
fig, ax = plt.subplots(figsize=(7, 5))
for i, (cls, color) in enumerate(zip(CLASS_NAMES, COLORS)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{cls} (AUC={auc(fpr,tpr):.4f})")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curves")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("step6_roc_curves.png", dpi=150)
plt.show()

# 3. Feature importance
importances = model.feature_importances_
idx = np.argsort(importances)
colors = ['#2ecc71' if FEATURE_NAMES[i]=='redshift' else '#4C72B0' for i in idx]
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([FEATURE_NAMES[i] for i in idx], importances[idx],
        color=colors, edgecolor='white')
ax.set_title("Feature Importance", fontweight="bold")
plt.tight_layout()
plt.savefig("step6_feature_importance.png", dpi=150)
plt.show()

# Save final model
joblib.dump({"model": model, "scaler": d["scaler"],
             "label_encoder": d["label_encoder"],
             "feature_names": FEATURE_NAMES}, "sdss_final_model.joblib")

print("✅ Saved sdss_final_model.joblib")
print("── PROJECT COMPLETE ──")