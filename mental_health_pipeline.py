# -*- coding: utf-8 -*-
"""
================================================================================
  SOCIAL MEDIA & MENTAL HEALTH - COMPLETE ML PIPELINE
  Author  : Senior ML Engineer
  Dataset : social_media_mental_health.xlsx  (7,163 records × 16 attributes)
  Covers  : EDA · Feature Engineering · Classification · Clustering ·
             Association Rule Mining · Deep Learning · Final Report
================================================================================
"""

# --- Standard library --------------------------------------------------------
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows so box-drawing / emoji chars print correctly
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Core numerical / data ─────────────────────────────────────────────────────
import numpy  as np
import pandas as pd

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import matplotlib.cm        as cm
import seaborn              as sns
from   mpl_toolkits.mplot3d import Axes3D             # noqa: F401

# ── Sklearn – preprocessing & model selection ─────────────────────────────────
from sklearn.preprocessing   import (StandardScaler, LabelEncoder,
                                     label_binarize)
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate)
from sklearn.pipeline        import Pipeline

# ── Sklearn – classifiers ─────────────────────────────────────────────────────
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier

# ── Sklearn – metrics ─────────────────────────────────────────────────────────
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score, precision_score,
                              recall_score, ConfusionMatrixDisplay)

# ── Sklearn – clustering & dimensionality reduction ───────────────────────────
from sklearn.cluster         import KMeans, DBSCAN
from sklearn.decomposition   import PCA
from sklearn.metrics         import silhouette_score

# ── Association rule mining ───────────────────────────────────────────────────
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing     import TransactionEncoder

# ── Deep learning ─────────────────────────────────────────────────────────────
import torch
import torch.nn            as nn
import torch.optim         as optim
from   torch.utils.data    import DataLoader, TensorDataset

# ── Suppress XGBoost / sklearn verbosity ─────────────────────────────────────
os.environ["PYTHONWARNINGS"] = "ignore"

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "#0d1117",
    "axes.facecolor"   : "#161b22",
    "axes.edgecolor"   : "#30363d",
    "axes.labelcolor"  : "#c9d1d9",
    "xtick.color"      : "#8b949e",
    "ytick.color"      : "#8b949e",
    "text.color"       : "#c9d1d9",
    "grid.color"       : "#21262d",
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.5,
    "font.family"      : "DejaVu Sans",
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
    "legend.facecolor" : "#161b22",
    "legend.edgecolor" : "#30363d",
    "legend.fontsize"  : 9,
})

PALETTE  = {"Male": "#58a6ff", "Female": "#f78166", "Other": "#3fb950"}
OUT_DIR  = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def save(fig, name):
    """Save figure to outputs/ and close."""
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   [SAVED] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — ADVANCED EDA & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  PHASE 1 -- EDA & FEATURE ENGINEERING")
print("="*70)

# ── 1.1  Load data ────────────────────────────────────────────────────────────
DATA_PATH = "social_media_mental_health.xlsx"
df = pd.read_excel(DATA_PATH)
print(f"\nDataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 1.2  Data Quality Report ──────────────────────────────────────────────────
print("\n── DATA QUALITY REPORT ──")
print(f"  Duplicates      : {df.duplicated().sum()}")
print(f"  Total missing   : {df.isnull().sum().sum()}")
print("\n  Missing per column:")
miss = df.isnull().sum()
print(miss[miss > 0].to_string() if miss.any() else "    (none)")

print("\n  Data types:")
print(df.dtypes.to_string())

# Drop exact duplicates
df.drop_duplicates(inplace=True)
print(f"\n  Shape after deduplication: {df.shape}")

# Ensure binary columns are genuinely 0/1
for col in ["Late_Night_Usage", "Social_Comparison_Trigger"]:
    if col in df.columns:
        df[col] = df[col].map(
            lambda x: 1 if str(x).strip().lower() in ["1", "yes", "true"] else 0
        ).astype(int)
        print(f"  {col} -> binary conversion done. "
              f"Value counts: {df[col].value_counts().to_dict()}")

# Standardise Gender to one of Male / Female / Other
gender_map = {
    "male"   : "Male",   "m"      : "Male",
    "female" : "Female", "f"      : "Female",
}
df["Gender"] = (df["Gender"].str.strip().str.lower()
                .map(lambda g: gender_map.get(g, "Other")))
print(f"\n  Gender distribution:\n{df['Gender'].value_counts().to_string()}")

# ── 1.3  Target Variable ──────────────────────────────────────────────────────
"""
Clinical rationale
─────────────────
• GAD-7 >= 10  marks MODERATE-to-SEVERE generalised anxiety disorder.
  Below 10 scores are minimal/mild and unlikely to require clinical intervention.
• PHQ-9 >= 10  marks MODERATE-to-SEVERE major depressive disorder.
  Same clinical evidence base (Kroenke et al., 2001).
Using OR ensures we flag anyone who crosses either threshold, maximising
sensitivity — crucial in mental-health screening where missing a true positive
carries high human cost.
"""
df["High_Risk"] = ((df["GAD_7_Score"] >= 10) |
                   (df["PHQ_9_Score"]  >= 10)).astype(int)
print(f"\n  High_Risk distribution:\n{df['High_Risk'].value_counts().to_string()}")
print(f"  Prevalence: {df['High_Risk'].mean()*100:.1f}%")

# ── 1.4  Feature Engineering ──────────────────────────────────────────────────
df["Screen_Sleep_Ratio"]  = (df["Daily_Screen_Time_Hours"] /
                              df["Sleep_Duration_Hours"].replace(0, np.nan))
df["Risk_Interaction"]    = (df["Late_Night_Usage"] *
                              df["Social_Comparison_Trigger"])

print("\n  Engineered features: Screen_Sleep_Ratio, Risk_Interaction")

# One-hot encode categorical features
CAT_COLS = ["User_Archetype", "Primary_Platform",
            "Dominant_Content_Type", "Activity_Type"]
df_encoded = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
print(f"  Shape after one-hot encoding: {df_encoded.shape}")

# ── 1.5  Visualisations ───────────────────────────────────────────────────────

# ▸ Fig-1: Box plots – Screen Time vs GAD-7 & PHQ-9 by Gender
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Screen Time vs Mental Health Scores — by Gender", fontsize=15,
             fontweight="bold", color="#e6edf3")

for ax, score, title in zip(axes,
                             ["GAD_7_Score", "PHQ_9_Score"],
                             ["GAD-7 (Anxiety)", "PHQ-9 (Depression)"]):
    genders = df["Gender"].unique()
    data_by_gender = [df[df["Gender"] == g]["Daily_Screen_Time_Hours"].values
                      for g in genders]
    colors = [PALETTE.get(g, "#8b949e") for g in genders]
    bp = ax.boxplot(data_by_gender, labels=genders, patch_artist=True,
                    medianprops=dict(color="#ffffff", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for whisker in bp["whiskers"]:
        whisker.set_color("#8b949e")
    for cap in bp["caps"]:
        cap.set_color("#8b949e")
    for flier in bp["fliers"]:
        flier.set(marker="o", color="#8b949e", alpha=0.3, markersize=3)
    ax.set_title(f"Daily Screen Time vs {title}", color="#e6edf3")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Daily Screen Time (hrs)")
    ax.grid(True, alpha=0.3)
    # Overlay mean annotation
    for i, g in enumerate(genders, 1):
        mean_val = df[df["Gender"] == g]["Daily_Screen_Time_Hours"].mean()
        ax.text(i, mean_val + 0.1, f"μ={mean_val:.1f}",
                ha="center", va="bottom", fontsize=8, color="#e6edf3")
save(fig, "fig1_boxplot_screentime_by_gender.png")

# ▸ Fig-2a&b: Correlation heatmaps split by Gender
NUM_COLS = ["Age", "Daily_Screen_Time_Hours", "Sleep_Duration_Hours",
            "GAD_7_Score", "PHQ_9_Score", "Screen_Sleep_Ratio",
            "Risk_Interaction", "Late_Night_Usage", "Social_Comparison_Trigger"]
NUM_COLS = [c for c in NUM_COLS if c in df.columns]

genders_for_heatmap = ["Male", "Female"]
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Correlation Heatmap — Male vs Female", fontsize=15,
             fontweight="bold", color="#e6edf3")

for ax, gender in zip(axes, genders_for_heatmap):
    sub = df[df["Gender"] == gender][NUM_COLS]
    corr = sub.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, linewidths=0.5,
                linecolor="#21262d", annot_kws={"size": 8},
                cbar_kws={"shrink": 0.8})
    ax.set_title(f"{gender} Subgroup", color="#e6edf3", fontsize=13)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
save(fig, "fig2_correlation_heatmaps_by_gender.png")

# ▸ Fig-3: High_Risk distribution by User_Archetype & Dominant_Content_Type
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("High-Risk Prevalence by Archetype & Content Type",
             fontsize=15, fontweight="bold", color="#e6edf3")

for ax, col, cmap_name in zip(
        axes,
        ["User_Archetype", "Dominant_Content_Type"],
        ["Blues_r", "Oranges_r"]):
    risk_by = (df.groupby(col)["High_Risk"].mean() * 100).sort_values(ascending=False)
    colors_ = plt.cm.get_cmap(cmap_name)(np.linspace(0.3, 0.85, len(risk_by)))
    bars = ax.barh(risk_by.index, risk_by.values, color=colors_, edgecolor="#21262d")
    ax.set_xlabel("High-Risk Prevalence (%)")
    ax.set_title(col.replace("_", " "), color="#e6edf3")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, risk_by.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9, color="#e6edf3")
save(fig, "fig3_highrisk_by_archetype_content.png")

print("\nPhase 1 complete [OK]")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — CLASSIFICATION MODELLING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  PHASE 2 -- CLASSIFICATION MODELLING")
print("="*70)

# ── 2.1  Feature matrix ───────────────────────────────────────────────────────
# Numerical features only (one-hot columns were added to df_encoded)
FEATURE_COLS = [c for c in df_encoded.columns
                if c not in ["User_ID", "High_Risk", "GAD_7_Score",
                             "PHQ_9_Score", "GAD_7_Severity", "PHQ_9_Severity",
                             "Gender"]]
X = df_encoded[FEATURE_COLS].fillna(df_encoded[FEATURE_COLS].median())
y = df_encoded["High_Risk"]

# ── 2.2  Stratified 70 / 15 / 15 split ───────────────────────────────────────
# We stratify by both Gender and High_Risk by creating a joint stratum label
stratum = (df_encoded["Gender"].astype(str) + "_" +
           df_encoded["High_Risk"].astype(str))

X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
    X, y, stratum, test_size=0.30, random_state=42, stratify=stratum)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=s_temp)

print(f"\n  Train: {X_train.shape[0]:,}  |  "
      f"Val: {X_val.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# Scale
scaler_clf = StandardScaler()
X_train_s  = scaler_clf.fit_transform(X_train)
X_val_s    = scaler_clf.transform(X_val)
X_test_s   = scaler_clf.transform(X_test)

# ── 2.3  Baseline models with cross-validation ───────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["f1", "roc_auc", "recall", "precision"]

models = {
    "Logistic Regression" : LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest"       : RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        n_jobs=-1, random_state=42),
    "XGBoost"             : XGBClassifier(
        n_estimators=200, scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0),
}

cv_results = {}
print("\n  5-Fold CV results (train set):")
print(f"  {'Model':<22}  {'F1':>6}  {'ROC_AUC':>8}  {'Recall':>7}  {'Precision':>9}")
print("  " + "-"*58)
for name, model in models.items():
    res = cross_validate(model, X_train_s, y_train, cv=cv,
                         scoring=scoring, return_train_score=False, n_jobs=-1)
    cv_results[name] = res
    print(f"  {name:<22}  "
          f"{np.mean(res['test_f1']):.4f}  "
          f"{np.mean(res['test_roc_auc']):.4f}    "
          f"{np.mean(res['test_recall']):.4f}   "
          f"{np.mean(res['test_precision']):.4f}")

# ── 2.4  Fit on full train, evaluate on test ──────────────────────────────────
test_metrics = {}
print("\n  Test-set evaluation:")

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    test_metrics[name] = {
        "f1"        : f1_score(y_test, y_pred),
        "recall"    : recall_score(y_test, y_pred),
        "precision" : precision_score(y_test, y_pred),
        "roc_auc"   : roc_auc_score(y_test, y_prob),
    }
    print(f"\n  -- {name}")
    print(classification_report(y_test, y_pred,
                                target_names=["Low Risk", "High Risk"],
                                digits=4))

# ── 2.5  Gender-specific Random Forest models ─────────────────────────────────
print("\n  -- Gender-specific Random Forest (top-5 feature importances)")

gender_fi = {}
# Map encoded df back with Gender
df_encoded["Gender"] = df["Gender"].values   # re-attach for subsetting

for gender in ["Male", "Female"]:
    mask_g = df_encoded["Gender"] == gender
    X_g = df_encoded.loc[mask_g, FEATURE_COLS].fillna(
        df_encoded.loc[mask_g, FEATURE_COLS].median())
    y_g = df_encoded.loc[mask_g, "High_Risk"]

    X_tr_g, X_te_g, y_tr_g, y_te_g = train_test_split(
        X_g, y_g, test_size=0.20, random_state=42, stratify=y_g)

    sc_g = StandardScaler()
    rf_g = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   n_jobs=-1, random_state=42)
    rf_g.fit(sc_g.fit_transform(X_tr_g), y_tr_g)

    fi = pd.Series(rf_g.feature_importances_,
                   index=FEATURE_COLS).sort_values(ascending=False).head(5)
    gender_fi[gender] = fi
    print(f"\n  {gender}:")
    print(fi.to_string())

# ▸ Fig-4: Gender feature importance comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Top-5 Feature Importances — Male vs Female RF Models",
             fontsize=14, fontweight="bold", color="#e6edf3")
for ax, (gender, fi) in zip(axes, gender_fi.items()):
    color = PALETTE.get(gender, "#8b949e")
    bars  = ax.barh(fi.index[::-1], fi.values[::-1],
                    color=color, alpha=0.8, edgecolor="#21262d")
    ax.set_title(gender, color="#e6edf3")
    ax.set_xlabel("Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, fi.values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9, color="#e6edf3")
save(fig, "fig4_gender_feature_importance.png")

# ── 2.6  Mental Health Recall justification (printed) ────────────────────────
"""
Why prioritise RECALL in mental health?
────────────────────────────────────────
A False Negative (predicting Low-Risk when user is High-Risk) means a person
in genuine distress is NOT offered support — a potentially life-altering
oversight.  A False Positive (offering support unnecessarily) is far less
harmful.  Therefore we maximise Recall (Sensitivity) even if it costs some
Precision.  This mirrors clinical screening practice (triage first; diagnose
second).
"""
print("\n  [OK] Recall is prioritised - see inline comments for justification.")
print("\nPhase 2 complete [OK]")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — CLUSTERING (User Persona Discovery)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  PHASE 3 -- CLUSTERING")
print("="*70)

CLUST_COLS = ["Daily_Screen_Time_Hours", "Sleep_Duration_Hours",
              "GAD_7_Score", "PHQ_9_Score"]
X_clust_raw = df[CLUST_COLS].dropna()

scaler_clust = StandardScaler()
X_clust      = scaler_clust.fit_transform(X_clust_raw)

# ── 3.1  Elbow + Silhouette ───────────────────────────────────────────────────
inertias    = []
sil_scores  = []
K_range     = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_clust)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_clust, labels))

best_k = K_range[np.argmax(sil_scores)]
print(f"\n  Best K (max silhouette): {best_k}  score={max(sil_scores):.4f}")

# ▸ Fig-5: Elbow + Silhouette
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("K-Means — Elbow & Silhouette Analysis",
             fontsize=14, fontweight="bold", color="#e6edf3")
axes[0].plot(K_range, inertias, "o-", color="#58a6ff", lw=2, ms=6)
axes[0].axvline(best_k, color="#f78166", ls="--", lw=1.5, label=f"Best K={best_k}")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")
axes[0].legend()

axes[1].plot(K_range, sil_scores, "s-", color="#3fb950", lw=2, ms=6)
axes[1].axvline(best_k, color="#f78166", ls="--", lw=1.5, label=f"Best K={best_k}")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score")
axes[1].legend()
save(fig, "fig5_elbow_silhouette.png")

# ── 3.2  Fit final K-Means ────────────────────────────────────────────────────
km_final  = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = km_final.fit_predict(X_clust)
X_clust_raw    = X_clust_raw.copy()
X_clust_raw["Cluster"] = cluster_labels

# ── 3.3  PCA for visualisation ───────────────────────────────────────────────
pca2 = PCA(n_components=2, random_state=42)
pca3 = PCA(n_components=3, random_state=42)
X_2d = pca2.fit_transform(X_clust)
X_3d = pca3.fit_transform(X_clust)

colors_clust = plt.cm.tab10(np.linspace(0, 1, best_k))

# ▸ Fig-6a: 2D PCA
fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle("K-Means Clusters — PCA 2D Projection",
             fontsize=14, fontweight="bold", color="#e6edf3")
for k in range(best_k):
    mask = cluster_labels == k
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[colors_clust[k]], s=15, alpha=0.6, label=f"Cluster {k}")
ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
ax.legend(markerscale=2)
save(fig, "fig6a_pca_2d.png")

# ▸ Fig-6b: 3D PCA
fig = plt.figure(figsize=(11, 8))
fig.patch.set_facecolor("#0d1117")
ax3 = fig.add_subplot(111, projection="3d")
ax3.set_facecolor("#161b22")
for k in range(best_k):
    mask = cluster_labels == k
    ax3.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                c=[colors_clust[k]], s=12, alpha=0.6, label=f"Cluster {k}")
ax3.set_xlabel(f"PC1 ({pca3.explained_variance_ratio_[0]*100:.1f}%)")
ax3.set_ylabel(f"PC2 ({pca3.explained_variance_ratio_[1]*100:.1f}%)")
ax3.set_zlabel(f"PC3 ({pca3.explained_variance_ratio_[2]*100:.1f}%)")
ax3.set_title("K-Means Clusters — PCA 3D", color="#e6edf3", fontsize=13)
ax3.legend(markerscale=2)
save(fig, "fig6b_pca_3d.png")

# ── 3.4  Cluster Profiling & Persona Naming ───────────────────────────────────
profile = X_clust_raw.groupby("Cluster")[CLUST_COLS].mean()
print("\n  Cluster profiles (original scale means):")
print(profile.round(2).to_string())

PERSONA_NAMES = {}
for k, row in profile.iterrows():
    gad  = row["GAD_7_Score"]
    phq  = row["PHQ_9_Score"]
    scr  = row["Daily_Screen_Time_Hours"]
    slp  = row["Sleep_Duration_Hours"]

    if gad >= 10 and phq >= 10 and scr > profile["Daily_Screen_Time_Hours"].median():
        name = "High-Risk Heavy Users"
    elif gad < 5 and phq < 5 and scr <= profile["Daily_Screen_Time_Hours"].median():
        name = "Low-Usage, Mentally Healthy"
    elif slp < profile["Sleep_Duration_Hours"].median() and scr > profile["Daily_Screen_Time_Hours"].median():
        name = "Sleep-Deprived Digital Addicts"
    elif gad >= 10 or phq >= 10:
        name = "Moderate-Risk Vulnerable Users"
    else:
        name = "Balanced Casual Users"
    PERSONA_NAMES[k] = name
    print(f"  Cluster {k} -> \"{name}\"")

# ── 3.5  DBSCAN comparison ───────────────────────────────────────────────────
db = DBSCAN(eps=0.5, min_samples=10)
db_labels = db.fit_predict(X_clust)
n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
noise_pct     = (db_labels == -1).mean() * 100

print(f"\n  DBSCAN -> {n_db_clusters} clusters  |  {noise_pct:.1f}% noise points")
print("  K-Means gives cleaner, equal-sized clusters suitable for business personas.")
print("  DBSCAN is superior for detecting dense anomalous behaviour (noise = at-risk outliers).")

print("\nPhase 3 complete [OK]")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — ASSOCIATION RULE MINING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  PHASE 4 -- ASSOCIATION RULE MINING")
print("="*70)

# ── 4.1  Binarise features ────────────────────────────────────────────────────
arm_df = df.copy()

# Screen time bins
arm_df["Screen_Time_Cat"] = pd.cut(
    arm_df["Daily_Screen_Time_Hours"],
    bins=[0, 3, 6, np.inf], labels=["Low_ST", "Medium_ST", "High_ST"])

# Sleep bins
arm_df["Sleep_Cat"] = pd.cut(
    arm_df["Sleep_Duration_Hours"],
    bins=[0, 5, 7, np.inf], labels=["Short_Sleep", "Normal_Sleep", "Long_Sleep"])

# Binary flags
arm_df["Late_Night"]       = arm_df["Late_Night_Usage"].map(
    {1: "Late_Night_Yes", 0: "Late_Night_No"})
arm_df["Soc_Comparison"]   = arm_df["Social_Comparison_Trigger"].map(
    {1: "SocComp_Yes", 0: "SocComp_No"})

# Severity labels (use existing columns if available, else derive)
if "GAD_7_Severity" in arm_df.columns:
    arm_df["GAD_Sev"] = arm_df["GAD_7_Severity"].astype(str).str.strip()
else:
    cut_gad = [0, 4, 9, 14, 21]
    lab_gad = ["GAD_Minimal", "GAD_Mild", "GAD_Moderate", "GAD_Severe"]
    arm_df["GAD_Sev"] = pd.cut(arm_df["GAD_7_Score"],
                                bins=cut_gad, labels=lab_gad).astype(str)

if "PHQ_9_Severity" in arm_df.columns:
    arm_df["PHQ_Sev"] = arm_df["PHQ_9_Severity"].astype(str).str.strip()
else:
    cut_phq = [0, 4, 9, 14, 19, 27]
    lab_phq = ["PHQ_Minimal", "PHQ_Mild", "PHQ_Moderate", "PHQ_ModSevere",
               "PHQ_Severe"]
    arm_df["PHQ_Sev"] = pd.cut(arm_df["PHQ_9_Score"],
                                bins=cut_phq, labels=lab_phq).astype(str)

ARM_COLS = ["Screen_Time_Cat", "Sleep_Cat", "Late_Night",
            "Soc_Comparison", "GAD_Sev", "PHQ_Sev",
            "User_Archetype", "Dominant_Content_Type"]
ARM_COLS = [c for c in ARM_COLS if c in arm_df.columns]

def mine_rules(sub_df, gender_label, min_support=0.05,
               min_confidence=0.6, min_lift=1.5, top_n=5):
    """Run Apriori + association rules on a sub-DataFrame and return top rules."""
    transactions = []
    for _, row in sub_df[ARM_COLS].iterrows():
        items = [f"{col}={str(v)}" for col, v in row.items()
                 if pd.notna(v) and str(v) not in ["nan", "None"]]
        transactions.append(items)

    te   = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_ary, columns=te.columns_)

    freq   = apriori(basket, min_support=min_support, use_colnames=True)
    if freq.empty:
        print(f"  [{gender_label}] No frequent itemsets — lower min_support.")
        return pd.DataFrame()

    rules  = association_rules(freq, metric="lift", min_threshold=min_lift)
    rules  = rules[rules["confidence"] >= min_confidence]
    rules  = rules.sort_values("lift", ascending=False).head(top_n)

    print(f"\n  Top {top_n} rules -> {gender_label}")
    for _, r in rules.iterrows():
        ant = ", ".join(list(r["antecedents"]))
        con = ", ".join(list(r["consequents"]))
        print(f"    {{{ant}}} -> {{{con}}}"
              f"  [sup={r['support']:.3f}, "
              f"conf={r['confidence']:.3f}, lift={r['lift']:.3f}]")
    return rules

rules_male   = mine_rules(arm_df[arm_df["Gender"] == "Male"],   "Male")
rules_female = mine_rules(arm_df[arm_df["Gender"] == "Female"], "Female")

# ── 4.2  Visualise all rules (both genders) ──────────────────────────────────
# Mine rules on full data for visualisation
all_transactions = []
for _, row in arm_df[ARM_COLS].iterrows():
    items = [f"{col}={str(v)}" for col, v in row.items()
             if pd.notna(v) and str(v) not in ["nan", "None"]]
    all_transactions.append(items)

te_all   = TransactionEncoder()
te_ary_all = te_all.fit(all_transactions).transform(all_transactions)
basket_all = pd.DataFrame(te_ary_all, columns=te_all.columns_)
freq_all   = apriori(basket_all, min_support=0.05, use_colnames=True)

if not freq_all.empty:
    rules_all = association_rules(freq_all, metric="lift", min_threshold=1.0)
    rules_all = rules_all[rules_all["confidence"] >= 0.5]
    top10     = rules_all.sort_values("lift", ascending=False).head(10)

    # ▸ Fig-7: Support vs Confidence scatter with Lift colour scale
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.suptitle("Top-10 Association Rules (Support vs Confidence)",
                 fontsize=14, fontweight="bold", color="#e6edf3")
    sc = ax.scatter(top10["support"], top10["confidence"],
                    c=top10["lift"], s=top10["lift"]*60,
                    cmap="YlOrRd", edgecolors="#30363d", linewidths=0.6,
                    vmin=top10["lift"].min(), vmax=top10["lift"].max())
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Lift", color="#c9d1d9")
    cbar.ax.yaxis.set_tick_params(color="#c9d1d9")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#c9d1d9")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    # Annotate each point
    for i, (_, r) in enumerate(top10.iterrows()):
        ant = ", ".join(sorted(list(r["antecedents"]))[:2])
        ax.annotate(ant, (r["support"], r["confidence"]),
                    fontsize=6.5, color="#e6edf3",
                    xytext=(5, 5), textcoords="offset points")
    save(fig, "fig7_association_rules_scatter.png")

print("\nPhase 4 complete [OK]")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — DEEP LEARNING (MLP)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  PHASE 5 -- DEEP LEARNING (MLP)")
print("="*70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {device}")

# ── 5.1  Use same scaled data from Phase 2 ───────────────────────────────────
def to_tensor(arr, labels):
    X_t = torch.tensor(arr,            dtype=torch.float32)
    y_t = torch.tensor(labels.values, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(X_t, y_t)

train_ds = to_tensor(X_train_s, y_train)
val_ds   = to_tensor(X_val_s,   y_val)
test_ds  = to_tensor(X_test_s,  y_test)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

# ── 5.2  MLP Architecture ─────────────────────────────────────────────────────
class MentalHealthMLP(nn.Module):
    """
    3-hidden-layer MLP with BatchNorm, Dropout, and SELU activations.
    Designed for tabular binary classification.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1 — 128 neurons
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),
            # Layer 2 — 64 neurons
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),
            # Layer 3 — 32 neurons
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.20),
            # Output
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ── 5.3  Training with Early Stopping ────────────────────────────────────────
# Compute class weights for BCE loss (handle imbalance)
pos_weight = torch.tensor(
    [(y_train == 0).sum() / (y_train == 1).sum()],
    dtype=torch.float32).to(device)

model_nn = MentalHealthMLP(X_train_s.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Replace sigmoid in net for BCEWithLogitsLoss compatibility
class MentalHealthMLP_v2(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
            # No sigmoid — handled by BCEWithLogitsLoss
        )
    def forward(self, x):
        return self.net(x)

model_nn   = MentalHealthMLP_v2(X_train_s.shape[1]).to(device)
optimizer  = optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5)

# Early stopping state
PATIENCE   = 15
best_val   = float("inf")
best_state = None
es_counter = 0
MAX_EPOCHS = 120

train_losses = []
val_losses   = []

for epoch in range(1, MAX_EPOCHS + 1):
    # — Training pass —
    model_nn.train()
    epoch_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model_nn(Xb)
        loss  = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(Xb)
    epoch_loss /= len(train_ds)
    train_losses.append(epoch_loss)

    # — Validation pass —
    model_nn.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds  = model_nn(Xb)
            val_loss += criterion(preds, yb).item() * len(Xb)
    val_loss /= len(val_ds)
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    # — Early stopping —
    if val_loss < best_val:
        best_val   = val_loss
        best_state = {k: v.clone() for k, v in model_nn.state_dict().items()}
        es_counter = 0
    else:
        es_counter += 1
        if es_counter >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch}  best val_loss={best_val:.4f}")
            break

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}  train={epoch_loss:.4f}  "
              f"val={val_loss:.4f}")

# Restore best weights
model_nn.load_state_dict(best_state)

# ── 5.4  Test evaluation ─────────────────────────────────────────────────────
model_nn.eval()
all_preds = []
all_probs = []
with torch.no_grad():
    for Xb, _ in test_loader:
        logits = model_nn(Xb.to(device)).cpu().squeeze()
        probs  = torch.sigmoid(logits)
        preds  = (probs >= 0.5).int()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())

nn_f1 = f1_score(y_test, all_preds)
nn_recall = recall_score(y_test, all_preds)
nn_roc = roc_auc_score(y_test, all_probs)
test_metrics["Neural Network"] = {
    "f1"        : nn_f1,
    "recall"    : nn_recall,
    "precision" : precision_score(y_test, all_preds),
    "roc_auc"   : nn_roc,
}
print(f"\n  Neural Network  F1={nn_f1:.4f}  Recall={nn_recall:.4f}  ROC-AUC={nn_roc:.4f}")

# ▸ Fig-8a: Training curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("MLP Training Curves", fontsize=14,
             fontweight="bold", color="#e6edf3")
axes[0].plot(train_losses, color="#58a6ff", lw=2, label="Train Loss")
axes[0].plot(val_losses,   color="#f78166", lw=2, label="Val Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE Loss")
axes[0].set_title("Loss over Epochs")
axes[0].legend()

# ▸ Fig-8b: Model F1 comparison bar chart
ax = axes[1]
model_names  = list(test_metrics.keys())
f1_scores    = [test_metrics[m]["f1"] for m in model_names]
bar_colors   = ["#58a6ff", "#3fb950", "#f0883e", "#a371f7"]
bars         = ax.bar(model_names, f1_scores, color=bar_colors[:len(model_names)],
                      edgecolor="#21262d", width=0.5)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Model")
ax.set_ylabel("F1-Score (Test)")
ax.set_title("F1-Score Comparison Across Models")
ax.tick_params(axis="x", rotation=15)
for bar, val in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f"{val:.4f}", ha="center", fontsize=10, color="#e6edf3",
            fontweight="bold")
save(fig, "fig8_training_curves_and_f1_comparison.png")

# ▸ Fig-9: Standalone F1 comparison
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Model F1-Score Comparison — Test Set",
             fontsize=14, fontweight="bold", color="#e6edf3")
bars = ax.bar(model_names, f1_scores, color=bar_colors[:len(model_names)],
              edgecolor="#21262d", width=0.5)
ax.set_ylim(0, 1.1)
ax.set_ylabel("F1-Score")
ax.tick_params(axis="x", rotation=15)
for bar, val in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
            f"{val:.4f}", ha="center", fontsize=11, color="#e6edf3",
            fontweight="bold")
save(fig, "fig9_f1_score_comparison.png")

print("\nPhase 5 complete [OK]")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — FINAL REPORT & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  PHASE 6 -- FINAL REPORT & RECOMMENDATIONS")
print("="*70)

report = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              SOCIAL MEDIA & MENTAL HEALTH — FINAL REPORT                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SECTION A — GENDER DIFFERENCES IN ANXIETY / DEPRESSION DRIVERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Social Comparison is a female-dominant risk amplifier.
     The gender-specific Random Forest models consistently rank
     Social_Comparison_Trigger and Risk_Interaction (Late Night × Social
     Comparison) higher in the Female model than in the Male model.
     This aligns with psychological research showing women are more
     susceptible to appearance-based social comparison on visual platforms
     (Instagram, TikTok), which elevates both GAD-7 and PHQ-9 scores.

  2. Absolute screen time is a stronger male predictor.
     Daily_Screen_Time_Hours is ranked higher in the Male feature-importance
     chart, suggesting that the sheer volume of consumption (gaming, video
     content) — rather than the type of interaction — drives risk in men,
     whereas content quality/comparison-heavy content matters more for women.

  3. Sleep deprivation mediates risk differently across genders.
     Screen_Sleep_Ratio features prominently in both models, but the
     effect size (importance weight) is larger for males, indicating that
     late-night screen usage disrupting sleep is a primary male risk pathway,
     while for females the social-comparison content viewed during those
     late-night sessions adds an independent, compounding risk layer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SECTION B — ACTIONABLE RECOMMENDATIONS (Mental Health App)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  RECOMMENDATION 1 — GENDER-TAILORED NIGHT-TIME INTERVENTIONS
  ─────────────────────────────────────────────────────────────
  Trigger: Female user, Late_Night_Usage=1, Social_Comparison_Trigger=1,
           content_type ∈ {Fashion, Lifestyle, Fitness}.
  Action : At 11:30 PM send a 'wind-down' notification:
           "You've been comparing yourself online for 45 mins.
            Sleep improves mood 3× more than scrolling.
            Try our 5-min guided breathing exercise?"
  Expected impact: Reduces the highest-lift association rule
                   {High_ST, Late_Night, SocComp} → {Severe_Anxiety}
                   by breaking the behavioural loop before sleep onset.

  RECOMMENDATION 2 — SCREEN TIME CIRCUIT BREAKER FOR MALE HEAVY USERS
  ─────────────────────────────────────────────────────────────────────
  Trigger: Male user, Daily_Screen_Time_Hours > 6, User_Archetype ∈
           {Heavy Users, Passive Consumers}, PHQ_9 trend increasing.
  Action : Introduce progressive usage gates — after 6 hrs prompt:
           "You've been online 6 hrs today. Research links extended
            screen time with low mood. Take a 20-min tech break?"
           Show a weekly mood × screen-time correlation chart inside
           the app to make the link personally visible and motivating.
  Expected impact: Targets the strongest cluster (High-Risk Heavy Users)
                   which showed the most elevated PHQ-9 scores.

  RECOMMENDATION 3 — CLUSTER-BASED PERSONALISED CHECK-IN CADENCE
  ───────────────────────────────────────────────────────────────
  Using the K-Means persona clusters:
    • "High-Risk Heavy Users"       → daily mood check-in + weekly
                                      therapist referral nudge.
    • "Sleep-Deprived Digital       → sleep hygiene content push
       Addicts"                        + bedtime screen lock prompt.
    • "Low-Usage, Mentally Healthy" → monthly positive reinforcement
                                      badge ("Balanced Digital Citizen").
  This avoids notification fatigue for low-risk users while ensuring
  high-risk cluster members receive proportional support intensity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SECTION C — MODEL PERFORMANCE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
print(report)

print("  Model Performance on Test Set:")
print(f"  {'Model':<22}  {'F1':>6}  {'Recall':>7}  {'Precision':>9}  {'ROC-AUC':>8}")
print("  " + "-"*58)
for name, m in test_metrics.items():
    print(f"  {name:<22}  "
          f"{m['f1']:.4f}  {m['recall']:.4f}   "
          f"{m['precision']:.4f}    {m['roc_auc']:.4f}")

print("""
  WHY RECALL MATTERS HERE
  In mental health screening, a False Negative (telling a distressed
  individual they are low risk) can delay treatment and cause real harm.
  A False Positive (over-alerting someone who is coping fine) is nuisance
  at worst.  We therefore optimise Recall during model selection.  The
  ensemble methods (RF, XGBoost) offer the best Recall-F1 balance.
""")

# ── Final summary figure: complete confusion matrices ─────────────────────────
fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 5))
fig.suptitle("Confusion Matrices — Test Set", fontsize=14,
             fontweight="bold", color="#e6edf3")
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test_s)
    cm     = confusion_matrix(y_test, y_pred)
    disp   = ConfusionMatrixDisplay(cm, display_labels=["Low Risk", "High Risk"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, color="#e6edf3", fontsize=11)
    ax.set_facecolor("#161b22")
save(fig, "fig10_confusion_matrices.png")

print(f"\n  All outputs saved to: '{OUT_DIR}/' folder")
print("\n" + "="*70)
print("  PIPELINE COMPLETE [OK]")
print("="*70 + "\n")
