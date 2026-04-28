#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate all manuscript figures from pipeline outputs.
Run after main.py to produce publication-quality plots.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import shap

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12})

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------
OUT_DIR = "results"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load required data
species_trends = pd.read_csv(os.path.join(OUT_DIR, "species_trends.csv"))
ml_df = pd.read_csv(os.path.join(OUT_DIR, "ml_dataset.csv"))
weather_path = "weather_summary.csv"
if not os.path.exists(weather_path):
    raise FileNotFoundError("weather_summary.csv not found – needed for hotspot map coordinates.")
weather = pd.read_csv(weather_path)
loc_coords = weather[['LOCALITY_ID', 'LATITUDE', 'LONGITUDE']].drop_duplicates()

# Predictors list (must match those used in main.py)
predictors = ['RECENT_MEAN_COUNT', 'RECENT_MEAN_CHECKLISTS',
              'CLIM_TEMP_MEAN', 'CLIM_PRECIP_TOTAL', 'MEAN_COUNT']

# Ensure required columns exist
missing = [c for c in predictors if c not in ml_df.columns]
if missing:
    raise RuntimeError(f"Missing predictor columns in ml_dataset.csv: {missing}")

# Decline threshold
threshold = np.percentile(species_trends['SLOPE_TS'].dropna(), 5)

# ------------------------------------------------------------------------
# Figure 1: Histogram of Theil-Sen slopes with decline threshold
# ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(species_trends['SLOPE_TS'].dropna(), bins=30, kde=True, ax=ax)
ax.axvline(threshold, color='red', linestyle='--', linewidth=1.5,
           label=f'Decline threshold ({threshold:.3f})')
ax.set_title("Distribution of Theil–Sen slopes (resident species, 2018–2024)")
ax.set_xlabel("Slope (log(count+1) per year)")
ax.set_ylabel("Number of species")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_slope_histogram.png"), dpi=300)
plt.close()

# ------------------------------------------------------------------------
# Figure 3: slope vs climate variables
# ------------------------------------------------------------------------
# Ensure slope column is present (ml_df from main.py includes SLOPE_TS)
if 'SLOPE_TS' not in ml_df.columns:
    ml_df = ml_df.merge(species_trends[['SPECIES_ID', 'SLOPE_TS']], on='SPECIES_ID', how='left')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(data=ml_df, x='CLIM_PRECIP_TOTAL', y='SLOPE_TS',
                hue='RAPID_DECLINE', alpha=0.7, ax=axes[0])
axes[0].axhline(0, color='grey', linestyle=':')
axes[0].set_title("Slope vs Total Precipitation")
axes[0].set_xlabel("Total Precipitation (recent window)")
axes[0].set_ylabel("Slope (Theil–Sen)")

sns.scatterplot(data=ml_df, x='CLIM_TEMP_MEAN', y='SLOPE_TS',
                hue='RAPID_DECLINE', alpha=0.7, ax=axes[1])
axes[1].axhline(0, color='grey', linestyle=':')
axes[1].set_title("Slope vs Mean Temperature")
axes[1].set_xlabel("Mean Temperature (recent window)")
axes[1].set_ylabel("Slope (Theil–Sen)")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_slope_vs_climate.png"), dpi=300)
plt.close()

# ------------------------------------------------------------------------
# Figure 4: Correlation heatmap
# ------------------------------------------------------------------------
corr_cols = ['SLOPE_TS', 'CLIM_TEMP_MEAN', 'CLIM_PRECIP_TOTAL', 'RECENT_MEAN_COUNT']
corr_cols = [c for c in corr_cols if c in ml_df.columns]
corr = ml_df[corr_cols].corr()
plt.figure(figsize=(5, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Correlation Matrix (resident species)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_correlation_heatmap.png"), dpi=300)
plt.close()

# ------------------------------------------------------------------------
# Figure 5: Decline hotspot map
# ------------------------------------------------------------------------
resident_agg_path = os.path.join(OUT_DIR, "resident_agg.csv")
if not os.path.exists(resident_agg_path):
    raise FileNotFoundError(
        "resident_agg.csv not found. Make sure main.py saved it "
        "(line 'resident_agg.to_csv(...)' after ecological filtering)."
    )

resident_agg = pd.read_csv(resident_agg_path, dtype={'LOCALITY_ID': str})
# List of declining species
declining_species = species_trends.loc[species_trends['RAPID_DECLINE'], 'SPECIES_ID'].tolist()

decl_agg = resident_agg[resident_agg['SPECIES_ID'].isin(declining_species)]
if decl_agg.empty:
    print("No declining species found in resident_agg – hotspot map skipped.")
else:
    hotspot = decl_agg.groupby('LOCALITY_ID', as_index=False).agg(
        n_declining_species=('SPECIES_ID', 'nunique')
    )
    hotspot = hotspot.merge(loc_coords, on='LOCALITY_ID', how='left')
    hotspot = hotspot.dropna(subset=['LATITUDE', 'LONGITUDE'])

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(hotspot['LONGITUDE'], hotspot['LATITUDE'],
                    s=np.clip(hotspot['n_declining_species'] * 10, 20, 200),
                    c=hotspot['n_declining_species'], cmap='Reds', alpha=0.8)
    plt.colorbar(sc, label='Number of declining species per locality')
    ax.set_title('Decline hotspots (co‑occurrence of declining species)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_hotspot_map.png"), dpi=300)
    plt.close()

# ------------------------------------------------------------------------
# Figure 6: ROC and Precision-Recall curves (all three classifiers)
# ------------------------------------------------------------------------
X = ml_df[predictors].values
y = ml_df['TARGET'].values
groups = ml_df['SPECIES_ID'].values

# Group-aware train/test split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(X, y, groups=groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Apply SMOTE as in training
if y_train.sum() >= 5 and y_train.sum() / len(y_train) < 0.15:
    sm = SMOTE(random_state=RANDOM_SEED)
    X_train, y_train = sm.fit_resample(X_train, y_train)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    'RandomForest': RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                           random_state=RANDOM_SEED, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200,
                              scale_pos_weight=(y_train == 0).sum() / max(1, (y_train == 1).sum()),
                              random_state=RANDOM_SEED, verbosity=0),
    'CatBoost': CatBoostClassifier(iterations=200, auto_class_weights='Balanced',
                                   random_seed=RANDOM_SEED, verbose=False)
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    ax2.plot(recall, precision, label=f'{name} (AP = {pr_auc:.2f})')

ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves')
ax1.legend()

ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision‑Recall Curves')
ax2.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig6_roc_pr_curves.png"), dpi=300)
plt.close()

# ------------------------------------------------------------------------
# Figure 7: SHAP summary plot (best model by PR-AUC from CV)
# ------------------------------------------------------------------------
# We'll train all three, pick the one with highest PR-AUC on this test set, 
# and then re-fit on full data for SHAP.
best_model = None
best_name = None
best_pr_auc = 0.0
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_prob)
    if pr_auc > best_pr_auc:
        best_pr_auc = pr_auc
        best_model = model
        best_name = name

print(f"Best model for SHAP: {best_name} (PR-AUC {best_pr_auc:.3f})")

# Re-fit the best model on the full dataset (scaled)
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X)
best_model.fit(X_scaled, y)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_scaled[:min(500, len(X_scaled))])
shap.summary_plot(shap_values, pd.DataFrame(X_scaled[:len(shap_values)], columns=predictors),
                   show=False, plot_size=(10, 6))
plt.title("SHAP Feature Importance (Retrospective Classification)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig7_shap_summary.png"), dpi=300)
plt.close()

print("All figures saved in", FIG_DIR)