# SDSS — Step 2: EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("sdss_clean.csv")
sns.set_theme(style="whitegrid")

COLORS = {'GALAXY': '#4C72B0', 'QSO': '#DD8452', 'STAR': '#55A868'}
ORDER  = ['GALAXY', 'QSO', 'STAR']

# 1. Class distribution
counts = df['class'].value_counts().reindex(ORDER)
counts.plot(kind='bar', color=list(COLORS.values()), edgecolor='white', rot=0)
plt.title("Class Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("eda_1_classes.png", dpi=150)
plt.show()

# 2. Redshift by class
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for cls, color in COLORS.items():
    axes[0].hist(df[df['class']==cls]['redshift'], bins=80,
                 alpha=0.5, label=cls, color=color)
    sns.kdeplot(df[df['class']==cls]['redshift'],
                ax=axes[1], label=cls, color=color, fill=True, alpha=0.25)
axes[0].set(title="Redshift Histogram", xlabel="Redshift", xlim=(-0.1, 6))
axes[1].set(title="Redshift KDE",       xlabel="Redshift", xlim=(-0.05, 3))
for ax in axes: ax.legend()
plt.tight_layout()
plt.savefig("eda_2_redshift.png", dpi=150)
plt.show()

# 3. Spectral bands boxplot
fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=False)
for ax, band in zip(axes, ['u','g','r','i','z']):
    sns.boxplot(data=df, x='class', y=band, order=ORDER,
                palette=COLORS, ax=ax, linewidth=0.8,
                flierprops=dict(marker='o', markersize=1, alpha=0.3))
    ax.set(title=f"Band {band}", xlabel="")
plt.suptitle("Photometric Bands by Class", y=1.02, fontweight='bold')
plt.tight_layout()
plt.savefig("eda_3_bands.png", dpi=150)
plt.show()

# 4. Correlation heatmap
cols = ['u','g','r','i','z','redshift']
corr = df[cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5,
            square=True, vmin=-1, vmax=1)
plt.title("Feature Correlation", fontweight='bold')
plt.tight_layout()
plt.savefig("eda_4_corr.png", dpi=150)
plt.show()

# 5. Pairplot (3k sample)
sample = df[['u','g','r','redshift','class']].sample(3000, random_state=42)
sns.pairplot(sample, hue='class', hue_order=ORDER, palette=COLORS,
             diag_kind='kde', plot_kws=dict(alpha=0.3, s=10), corner=True)
plt.suptitle("Pairplot (3k sample)", y=1.01, fontweight='bold')
plt.savefig("eda_5_pairplot.png", dpi=130, bbox_inches='tight')
plt.show()

print("Done — 5 plots saved.")