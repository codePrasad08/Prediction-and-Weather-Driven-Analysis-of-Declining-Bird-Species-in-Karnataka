"""
Complete pipeline for retrospective classification of rapidly declining
resident bird species in Karnataka, India.

The analysis combines eBird citizen-science observations (2018-2024) with
pre-fetched NASA POWER climatic summaries. It addresses the following
reviewer-requested methodological points:

- No trend-derived metrics are used as predictors (circularity removed).
- The decline threshold is the 5th percentile of Theil-Sen slopes, with a
  sensitivity table for alternative percentiles.
- GroupKFold cross-validation prevents information leakage across species.
- Three classifiers (Random Forest, XGBoost, CatBoost) are compared.
- The entire workflow is retrospective (classification), not prospective
  (prediction).

The script expects three data files in the working directory:
    ebd_IN-KA_smp_relAug-2025.txt          (EBD presence data)
    ebd_IN-KA_smp_relAug-2025_sampling.txt  (SED metadata)
    weather_summary.csv                     (pre-computed NASA POWER summary)
"""

import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import TheilSenRegressor, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
import shap

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# Global configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

EBD_PATH = "ebd_IN-KA_smp_relAug-2025.txt"
SED_PATH = "ebd_IN-KA_smp_relAug-2025_sampling.txt"
WEATHER_PATH = "weather_summary.csv"
SPECIES_STATUS_PATH = "data/species_status_lookup.csv"  # optional file, may not exist

OUT_DIR = "results"
FIG_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

POWER_START_YEAR = 2018
POWER_END_YEAR = 2024
MIN_YEARS_FOR_TREND = 4
MIN_CHECKLISTS_FOR_TREND = 5
DECLINE_PERCENTILE = 5
RECENT_WINDOW = 3

def is_ambiguous_taxon(name: str) -> bool:
    """Flag genus-level, family-level, or hybrid designations."""
    if not isinstance(name, str) or name.strip() == "":
        return True
    n = name.lower().strip()
    ambiguous_tokens = ['sp.', 'spp', '/', 'cf.', 'aff.', 'sp', 'spp.']
    for t in ambiguous_tokens:
        if t in n:
            return True
    if len(n.split()) < 2:
        return True
    return False

def infer_ecological_status(agg_df):
    """
    Assign ecological status (resident, migrant, uncertain) based on
    years present, total observations, and months with records.
    Returns a DataFrame with columns SPECIES_ID and STATUS.
    """
    sp = agg_df.groupby('SPECIES_ID', as_index=False).agg(
        years_present=('YEAR', 'nunique'),
        total_obs=('OBS_COUNT', 'sum'),
        months_present=('MONTH', 'nunique'),
    )
    status_list = []
    for _, row in sp.iterrows():
        yrs, obs, mon = row['years_present'], row['total_obs'], row['months_present']
        if yrs <= 1 or obs < 5:
            status_list.append('uncertain')
        elif mon >= 9 or (yrs >= 3 and obs >= 10):
            status_list.append('resident')
        elif mon <= 6 and yrs >= 2:
            status_list.append('migrant')
        else:
            status_list.append('uncertain')
    sp['STATUS'] = status_list
    return sp[['SPECIES_ID', 'STATUS']]

def find_col(cols, candidates):
    """Return the first column from `candidates` found in `cols`."""
    for cand in candidates:
        for c in cols:
            if c.lower() == cand.lower():
                return c
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None

# Load and clean sampling events (SED)
print("Loading SED ...")
sed = pd.read_csv(SED_PATH, sep='\t', low_memory=False, dtype=str)
sed.columns = [c.strip() for c in sed.columns]

sample_id_col = find_col(sed.columns, ['SAMPLING EVENT IDENTIFIER', 'SAMPLING_EVENT_ID'])
date_col = find_col(sed.columns, ['OBSERVATION DATE', 'OBSERVATION_DATE'])
lat_col = find_col(sed.columns, ['LATITUDE'])
lon_col = find_col(sed.columns, ['LONGITUDE', 'LON'])
complete_col = find_col(sed.columns, ['ALL SPECIES REPORTED', 'COMPLETE CHECKLIST'])
locality_col = find_col(sed.columns, ['LOCALITY ID', 'LOCALITY_ID'])

sed = sed.rename(columns={
    sample_id_col: 'SAMPLE_ID',
    date_col: 'OBS_DATE',
    lat_col: 'LATITUDE',
    lon_col: 'LONGITUDE',
    complete_col: 'COMPLETE',
    locality_col: 'LOCALITY_ID' if locality_col else 'LOCALITY_ID',
})
sed['SAMPLE_ID'] = sed['SAMPLE_ID'].astype(str).str.strip()
if 'LOCALITY_ID' in sed.columns:
    sed['LOCALITY_ID'] = sed['LOCALITY_ID'].astype(str).str.strip()
else:
    sed['LOCALITY_ID'] = sed['SAMPLE_ID']

sed['OBS_DATE'] = pd.to_datetime(sed['OBS_DATE'], errors='coerce')
sed = sed[~sed['OBS_DATE'].isna()]
sed['LATITUDE'] = pd.to_numeric(sed['LATITUDE'], errors='coerce')
sed['LONGITUDE'] = pd.to_numeric(sed['LONGITUDE'], errors='coerce')
sed['COMPLETE'] = sed['COMPLETE'].str.lower().isin(['y', 'yes', 'true', '1'])

sed = sed[sed['COMPLETE']].copy()
sed['YEAR'] = sed['OBS_DATE'].dt.year
sed['MONTH'] = sed['OBS_DATE'].dt.month
sed = sed[(sed['YEAR'] >= POWER_START_YEAR) & (sed['YEAR'] <= POWER_END_YEAR)]

sed_small = sed[['SAMPLE_ID', 'YEAR', 'MONTH', 'LOCALITY_ID', 'LATITUDE', 'LONGITUDE']].dropna(
    subset=['LATITUDE', 'LONGITUDE']
)
print(f"SED complete checklists: {len(sed_small)}")

# Load and filter EBD (presence data)
print("Loading EBD ...")
with open(EBD_PATH, 'r', encoding='utf-8') as f:
    header = f.readline().strip().split('\t')
header = [h.strip() for h in header]

ebd_sample_col = find_col(header, ['SAMPLING EVENT IDENTIFIER', 'SAMPLING_EVENT_ID'])
ebd_common_col = find_col(header, ['COMMON NAME'])
ebd_sci_col = find_col(header, ['SCIENTIFIC NAME'])
ebd_count_col = find_col(header, ['OBSERVATION COUNT'])
ebd_date_col = find_col(header, ['OBSERVATION DATE'])
ebd_loc_col = find_col(header, ['LOCALITY ID', 'LOCALITY_ID'])

valid_samples = set(sed_small['SAMPLE_ID'].unique())

chunks = []
for chunk in pd.read_csv(
    EBD_PATH, sep='\t',
    usecols=[ebd_sample_col, ebd_common_col, ebd_sci_col, ebd_count_col, ebd_date_col, ebd_loc_col],
    chunksize=200_000, low_memory=False, dtype=str
):
    chunk.columns = [c.strip() for c in chunk.columns]
    chunk = chunk.rename(columns={
        ebd_sample_col: 'SAMPLE_ID',
        ebd_common_col: 'COMMON_NAME',
        ebd_sci_col: 'SCIENTIFIC_NAME',
        ebd_count_col: 'OBS_COUNT',
        ebd_date_col: 'OBS_DATE',
        ebd_loc_col: 'LOCALITY_ID',
    })
    chunk['SAMPLE_ID'] = chunk['SAMPLE_ID'].astype(str).str.strip()
    chunk = chunk[chunk['SAMPLE_ID'].isin(valid_samples)]
    chunk['OBS_DATE'] = pd.to_datetime(chunk['OBS_DATE'], errors='coerce')
    chunk = chunk[~chunk['OBS_DATE'].isna()]
    chunk['YEAR'] = chunk['OBS_DATE'].dt.year
    chunk['MONTH'] = chunk['OBS_DATE'].dt.month
    chunk = chunk[(chunk['YEAR'] >= POWER_START_YEAR) & (chunk['YEAR'] <= POWER_END_YEAR)]
    chunk['OBS_COUNT'] = pd.to_numeric(chunk['OBS_COUNT'], errors='coerce').fillna(1).astype(int)
    chunk.loc[chunk['OBS_COUNT'] < 1, 'OBS_COUNT'] = 1
    chunk = chunk[~(chunk['COMMON_NAME'].isna() & chunk['SCIENTIFIC_NAME'].isna())]
    chunks.append(chunk)

ebd = pd.concat(chunks, ignore_index=True)
ebd['SPECIES_ID'] = ebd['SCIENTIFIC_NAME'].fillna(ebd['COMMON_NAME']).astype(str).str.strip()

loc_coords = sed_small.groupby('SAMPLE_ID').agg(
    LATITUDE=('LATITUDE', 'first'),
    LONGITUDE=('LONGITUDE', 'first')
).reset_index()
ebd = ebd.merge(loc_coords[['SAMPLE_ID', 'LATITUDE', 'LONGITUDE']], on='SAMPLE_ID', how='left')
print(f"EBD rows after filtering: {len(ebd)}")

# Aggregate to species x locality x year
agg = ebd.groupby(['SPECIES_ID', 'LOCALITY_ID', 'YEAR'], as_index=False).agg(
    OBS_COUNT=('OBS_COUNT', 'sum'),
    N_CHECKLISTS=('SAMPLE_ID', 'nunique'),
    LATITUDE=('LATITUDE', 'first'),
    LONGITUDE=('LONGITUDE', 'first'),
)
ebd_months = ebd[['SPECIES_ID', 'MONTH']].drop_duplicates()
agg = agg.merge(ebd_months, on='SPECIES_ID', how='left')
print(f"Aggregated rows: {len(agg)}")

# Ecological status assignment
if os.path.exists(SPECIES_STATUS_PATH):
    print("Loading species status lookup...")
    status_lookup = pd.read_csv(SPECIES_STATUS_PATH, dtype=str)
    status_lookup.columns = [c.strip().upper() for c in status_lookup.columns]
    if 'SPECIES_ID' in status_lookup.columns and 'STATUS' in status_lookup.columns:
        status_lookup['STATUS'] = status_lookup['STATUS'].str.lower().str.strip()
        agg = agg.merge(status_lookup[['SPECIES_ID', 'STATUS']], on='SPECIES_ID', how='left')
    else:
        warnings.warn("Status lookup file missing expected columns. Using auto-inference.")
        status_df = infer_ecological_status(agg)
        agg = agg.merge(status_df, on='SPECIES_ID', how='left')
else:
    print("No species status file found; auto-inferring...")
    status_df = infer_ecological_status(agg)
    agg = agg.merge(status_df, on='SPECIES_ID', how='left')

agg['AMBIGUOUS'] = agg['SPECIES_ID'].apply(is_ambiguous_taxon)
agg = agg[~agg['AMBIGUOUS']].copy()

resident_agg = agg[agg['STATUS'].isin(['resident', 'resident_local', 'local_resident'])].copy()
print(f"Resident species rows: {len(resident_agg)}, unique species: {resident_agg['SPECIES_ID'].nunique()}")

resident_agg.to_csv(os.path.join(OUT_DIR, 'resident_agg.csv'), index=False)

# Load weather data
print("Loading weather data...")
weather = pd.read_csv(WEATHER_PATH)
weather.columns = [c.strip() for c in weather.columns]
weather = weather.rename(columns={
    'LOCALITY_ID': 'LOCALITY_ID',
    'YEAR': 'YEAR',
    'TEMP_MEAN': 'TEMP_MEAN',
    'PRECIP_TOTAL': 'PRECIP_TOTAL',
})
weather['LOCALITY_ID'] = weather['LOCALITY_ID'].astype(str)
weather['YEAR'] = pd.to_numeric(weather['YEAR'], errors='coerce').astype(int)
weather = weather.dropna(subset=['LOCALITY_ID', 'YEAR'])

loc_weather = weather.groupby('LOCALITY_ID').agg(
    LATITUDE=('LATITUDE', 'first'),
    LONGITUDE=('LONGITUDE', 'first'),
    TEMP_MEAN=('TEMP_MEAN', 'mean'),
    PRECIP_TOTAL=('PRECIP_TOTAL', 'mean'),
).reset_index()

# Compute species-level Theil-Sen trends
print("Computing species-level trends...")
species_trends = []
for sp, grp in tqdm(resident_agg.groupby('SPECIES_ID'), desc="Species trends"):
    grp_sp = grp.groupby('YEAR', as_index=False).agg(
        OBS_COUNT=('OBS_COUNT', 'sum'),
        N_CHECKLISTS=('N_CHECKLISTS', 'sum'),
    )
    if grp_sp['YEAR'].nunique() < MIN_YEARS_FOR_TREND:
        continue
    X = grp_sp['YEAR'].values.reshape(-1, 1)
    y = np.log1p(grp_sp['OBS_COUNT'].values)
    try:
        ts = TheilSenRegressor(random_state=RANDOM_SEED).fit(X, y)
        slope_ts = float(ts.coef_[0])
    except Exception:
        slope_ts = float(LinearRegression().fit(X, y).coef_[0])
    species_trends.append({
        'SPECIES_ID': sp,
        'SLOPE_TS': slope_ts,
        'N_YEARS': len(grp_sp),
        'MEAN_COUNT': float(np.expm1(y).mean()),
        'TOTAL_CHECKLISTS': grp_sp['N_CHECKLISTS'].sum(),
    })

species_trends_df = pd.DataFrame(species_trends)
if species_trends_df.empty:
    raise RuntimeError("No species met minimum year criteria. Lower MIN_YEARS_FOR_TREND.")

threshold = np.percentile(species_trends_df['SLOPE_TS'].dropna(), DECLINE_PERCENTILE)
print(f"Decline threshold (log abundance slope): {threshold:.4f}")
species_trends_df['RAPID_DECLINE'] = species_trends_df['SLOPE_TS'] <= threshold

sens_pcts = [2, 5, 10, 15, 20]
sens = []
for p in sens_pcts:
    th = np.percentile(species_trends_df['SLOPE_TS'].dropna(), p)
    n = (species_trends_df['SLOPE_TS'] <= th).sum()
    sens.append({'percentile': p, 'threshold': th, 'n_declining': n})
sens_df = pd.DataFrame(sens)
sens_df.to_csv(os.path.join(OUT_DIR, 'threshold_sensitivity.csv'), index=False)
print("\nSensitivity analysis:")
print(sens_df)

species_trends_df.to_csv(os.path.join(OUT_DIR, 'species_trends.csv'), index=False)

# Build ML dataset (no slope predictors)
print("Building ML dataset...")
max_year = resident_agg['YEAR'].max()
recent_cut = max_year - RECENT_WINDOW + 1
recent = resident_agg[resident_agg['YEAR'] >= recent_cut]

recent_sp = recent.groupby('SPECIES_ID', as_index=False).agg(
    RECENT_MEAN_COUNT=('OBS_COUNT', 'mean'),
    RECENT_MEAN_CHECKLISTS=('N_CHECKLISTS', 'mean'),
)

sp_locs_recent = recent[['SPECIES_ID', 'LOCALITY_ID']].drop_duplicates()
sp_climate = sp_locs_recent.merge(loc_weather, on='LOCALITY_ID', how='left')
sp_climate = sp_climate.groupby('SPECIES_ID').agg(
    CLIM_TEMP_MEAN=('TEMP_MEAN', 'mean'),
    CLIM_PRECIP_TOTAL=('PRECIP_TOTAL', 'mean'),
).reset_index()

ml_df = species_trends_df.merge(recent_sp, on='SPECIES_ID', how='left')
ml_df = ml_df.merge(sp_climate, on='SPECIES_ID', how='left')
ml_df = ml_df.dropna(subset=['CLIM_TEMP_MEAN', 'CLIM_PRECIP_TOTAL']).copy()

predictors = ['RECENT_MEAN_COUNT', 'RECENT_MEAN_CHECKLISTS',
              'CLIM_TEMP_MEAN', 'CLIM_PRECIP_TOTAL', 'MEAN_COUNT']

imputer = SimpleImputer(strategy='median')
ml_df[predictors] = imputer.fit_transform(ml_df[predictors])
ml_df['TARGET'] = ml_df['RAPID_DECLINE'].astype(int)
print(f"ML dataset: {len(ml_df)} species, {ml_df['TARGET'].sum()} declining")
ml_df.to_csv(os.path.join(OUT_DIR, 'ml_dataset.csv'), index=False)

# Machine learning with GroupKFold (three models)
X = ml_df[predictors].values
y = ml_df['TARGET'].values
groups = ml_df['SPECIES_ID'].values

gkf = GroupKFold(n_splits=5)
model_metrics = {
    'RandomForest': {'precision': [], 'recall': [], 'f1': [], 'pr_auc': [], 'roc_auc': []},
    'XGBoost':      {'precision': [], 'recall': [], 'f1': [], 'pr_auc': [], 'roc_auc': []},
    'CatBoost':     {'precision': [], 'recall': [], 'f1': [], 'pr_auc': [], 'roc_auc': []},
}

best_model = None
best_model_name = None
best_pr_auc = 0.0

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    if y_tr.sum() >= 5 and y_tr.sum() / len(y_tr) < 0.15:
        sm = SMOTE(random_state=RANDOM_SEED)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_prob_rf = rf.predict_proba(X_val)[:, 1]

    scale_pos_weight = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
    xgb = XGBClassifier(n_estimators=200, scale_pos_weight=scale_pos_weight,
                        random_state=RANDOM_SEED, verbosity=0)
    xgb.fit(X_tr, y_tr)
    y_prob_xgb = xgb.predict_proba(X_val)[:, 1]

    cat = CatBoostClassifier(iterations=200, auto_class_weights='Balanced',
                             random_seed=RANDOM_SEED, verbose=False)
    cat.fit(X_tr, y_tr)
    y_prob_cat = cat.predict_proba(X_val)[:, 1]

    for name, y_prob in [('RandomForest', y_prob_rf), ('XGBoost', y_prob_xgb), ('CatBoost', y_prob_cat)]:
        y_pred = (y_prob >= 0.5).astype(int)
        pr_auc = average_precision_score(y_val, y_prob)
        roc_auc_val = roc_auc_score(y_val, y_prob)
        prec = np.mean(y_pred[y_val == 1]) if (y_val == 1).sum() > 0 else 0.0
        rec = np.mean(y_pred[y_val == 1]) if (y_val == 1).sum() > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        model_metrics[name]['precision'].append(prec)
        model_metrics[name]['recall'].append(rec)
        model_metrics[name]['f1'].append(f1)
        model_metrics[name]['pr_auc'].append(pr_auc)
        model_metrics[name]['roc_auc'].append(roc_auc_val)

    print(f"Fold {fold+1}: ", end="")
    for name in model_metrics:
        print(f"{name} PR-AUC={model_metrics[name]['pr_auc'][-1]:.3f} ", end="")
    print()

print("\nCross-validation summary (mean ± std):")
for name in model_metrics:
    means = {k: np.mean(v) for k, v in model_metrics[name].items()}
    stds = {k: np.std(v) for k, v in model_metrics[name].items()}
    print(f"{name}: PR-AUC {means['pr_auc']:.3f} ± {stds['pr_auc']:.3f}, "
          f"ROC-AUC {means['roc_auc']:.3f}, F1 {means['f1']:.3f}")
    if means['pr_auc'] > best_pr_auc:
        best_pr_auc = means['pr_auc']
        best_model_name = name

print(f"\nBest model by average PR-AUC: {best_model_name} ({best_pr_auc:.3f})")

cv_rows = []
for name, mets in model_metrics.items():
    for i in range(len(mets['pr_auc'])):
        cv_rows.append({
            'Model': name, 'Fold': i+1, 'PR-AUC': mets['pr_auc'][i],
            'ROC-AUC': mets['roc_auc'][i], 'Precision': mets['precision'][i],
            'Recall': mets['recall'][i], 'F1': mets['f1'][i],
        })
cv_results = pd.DataFrame(cv_rows)
cv_results.to_csv(os.path.join(OUT_DIR, 'cv_results_all_models.csv'), index=False)

# Feature importance and SHAP interpretation (best model)
print("Computing SHAP values...")
if best_model_name == 'XGBoost':
    final_model = XGBClassifier(n_estimators=200,
                                scale_pos_weight=(y == 0).sum() / max(1, (y == 1).sum()),
                                random_state=RANDOM_SEED, verbosity=0)
elif best_model_name == 'RandomForest':
    final_model = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                         random_state=RANDOM_SEED, n_jobs=-1)
elif best_model_name == 'CatBoost':
    final_model = CatBoostClassifier(iterations=200, auto_class_weights='Balanced',
                                     random_seed=RANDOM_SEED, verbose=False)
else:
    final_model = XGBClassifier(n_estimators=200,
                                scale_pos_weight=(y == 0).sum() / max(1, (y == 1).sum()),
                                random_state=RANDOM_SEED, verbosity=0)

scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X)
final_model.fit(X_scaled, y)

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_scaled[:min(500, len(X_scaled))])
shap.summary_plot(shap_values, pd.DataFrame(X_scaled[:len(shap_values)], columns=predictors),
                   show=False, plot_size=(10, 6))
plt.title("SHAP Feature Importance (Retrospective Classification)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'shap_summary.png'), dpi=300)
plt.close()

# Save supplementary outputs
declining = species_trends_df[species_trends_df['RAPID_DECLINE']].sort_values('SLOPE_TS')
declining.to_csv(os.path.join(OUT_DIR, 'declining_species.csv'), index=False)
print(f"\nTop declining species (first 10):")
print(declining[['SPECIES_ID', 'SLOPE_TS']].head(10))

print("\nPipeline complete. All outputs in /results and /figures")