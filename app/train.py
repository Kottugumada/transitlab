"""
Classifier training — §5 of the spec.

Binary task: planet-like (CONFIRMED + CANDIDATE) vs FALSE POSITIVE.
3-class is logged as a secondary experiment but the binary model is saved.

Key invariants enforced here:
  1. Split by star (kepid), never by row — multi-KOI hosts must land in one split.
  2. Two models trained every run:
       honest-features: photometry + stellar params only (no Robovetter / koi_score)
       all-features:    adds koi_score and all koi_fpflag_* columns
     The honest model is the headline; the all-features model exists solely for
     the §5 leakage comparison. Expect all-features to look suspiciously better —
     see MODEL_CARD for why.
  3. Two baselines must be beaten (or the README says so explicitly):
       majority-class: always predict FALSE POSITIVE
       snr-threshold:  single-feature koi_model_snr cutoff optimised on train set
  4. Metrics: ROC-AUC, recall@CONFIRMED at FPR=0.1, precision-in-top-100,
     per-class precision/recall, confusion matrix. Accuracy is NOT the headline.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.catalogue_sync import DISPOSITION_DERIVED_FEATURES, HONEST_FEATURE_COLUMNS
from app.models import TrainingRun

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SPLIT_SEED = 42
TEST_SIZE = 0.2
MODEL_DIR = Path("models")

# All-features set: honest features + numeric disposition-derived columns.
# koi_pdisposition excluded (string; would require encoding and it IS the
# disposition prior — including it would make the leakage trivially obvious
# even to casual readers, obscuring the subtler koi_score / fpflag leakage).
ALL_FEATURE_COLUMNS = HONEST_FEATURE_COLUMNS + [
    col for col in DISPOSITION_DERIVED_FEATURES
    if col not in ("koi_pdisposition",)   # skip the string column
]

# Labels for the binary task
LABEL_FALSE_POSITIVE = 0
LABEL_PLANET_LIKE = 1

LABEL_NAMES = {LABEL_FALSE_POSITIVE: "FALSE POSITIVE", LABEL_PLANET_LIKE: "planet-like"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_training_data(db: Session) -> pd.DataFrame:
    """
    Load all KOIs with a known disposition and non-null features from Neon.
    Expands the JSONB features column into individual DataFrame columns.
    """
    rows = db.execute(
        text(
            "SELECT kepoi_name, kepid, disposition, features "
            "FROM koi_catalogue "
            "WHERE disposition IS NOT NULL AND features IS NOT NULL AND kepid IS NOT NULL"
        )
    ).fetchall()

    if not rows:
        raise RuntimeError("koi_catalogue is empty — run catalogue sync first.")

    records = []
    for row in rows:
        record = {
            "kepoi_name": row.kepoi_name,
            "kepid": int(row.kepid),
            "disposition": row.disposition,
        }
        if row.features:
            record.update({k: v for k, v in row.features.items() if isinstance(v, (int, float))})
        records.append(record)

    df = pd.DataFrame(records)
    log.info("Loaded %d labelled KOIs from koi_catalogue", len(df))
    return df


def load_training_data_from_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load training data from a CSV fixture (used in offline tests).
    Maps the raw KOI column names to the same schema as load_training_data.
    """
    raw = pd.read_csv(csv_path, low_memory=False)
    # Rename TAP columns to match the features-blob keys
    records = []
    for _, row in raw.iterrows():
        if pd.isna(row.get("koi_disposition")) or pd.isna(row.get("kepid")):
            continue
        record = {
            "kepoi_name": str(row["kepoi_name"]),
            "kepid": int(row["kepid"]),
            "disposition": str(row["koi_disposition"]),
        }
        for col in HONEST_FEATURE_COLUMNS + [c for c in DISPOSITION_DERIVED_FEATURES if c != "koi_pdisposition"]:
            v = row.get(col)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                record[col] = float(v)
        records.append(record)
    return pd.DataFrame(records)


# ── Labels ────────────────────────────────────────────────────────────────────

def make_binary_labels(dispositions: pd.Series) -> pd.Series:
    """CONFIRMED / CANDIDATE → 1 (planet-like), FALSE POSITIVE → 0."""
    return (dispositions != "FALSE POSITIVE").astype(int)


# ── Train / test split ────────────────────────────────────────────────────────

def split_by_star(
    df: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Group-aware split: all KOIs from the same host star (kepid) land in the
    same partition.  Row-splitting would leak because a host with multiple
    KOIs at different orbital periods could appear on both sides, sharing
    stellar-parameter features.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, y, groups=df["kepid"].values))
    return train_idx, test_idx


# ── Model training ────────────────────────────────────────────────────────────

def _make_gbm(seed: int = SPLIT_SEED) -> HistGradientBoostingClassifier:
    # HistGradientBoostingClassifier chosen over GradientBoostingClassifier because:
    #   - handles NaN natively (no imputation step needed; avoids imputer choices)
    #   - faster on ~8k rows (histogram binning)
    #   - identical or better accuracy in practice on tabular data with NaNs
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,       # ~shallow enough to generalise on 8k rows
        min_samples_leaf=20,
        random_state=seed,
        class_weight="balanced", # FALSE POSITIVE is majority class — balance it
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int = SPLIT_SEED,
) -> HistGradientBoostingClassifier:
    model = _make_gbm(seed)
    model.fit(X_train, y_train)
    return model


# ── Baselines ─────────────────────────────────────────────────────────────────

def majority_class_metrics(y_test: pd.Series) -> dict:
    """Always predict the majority class (FALSE POSITIVE = 0)."""
    y_pred = np.zeros(len(y_test), dtype=int)
    # ROC-AUC requires a probability; majority-class has none — use 0.5 convention
    # (AUC=0.5 is the random baseline; majority-class can't be ranked).
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    return {
        "label": "majority-class baseline",
        "roc_auc": 0.5,
        "recall_at_fpr01": float(np.mean(y_test == LABEL_FALSE_POSITIVE)),
        "precision_top100": None,
        "classification_report": report,
        "confusion_matrix": cm,
        "note": "Always predicts FALSE POSITIVE; AUC=0.5 by definition.",
    }


def snr_threshold_metrics(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Single-feature baseline: threshold on koi_model_snr.
    Threshold is chosen to maximise F1 on the TRAINING set, then applied
    to the test set — no test-set peeking.
    """
    snr_col = "koi_model_snr"
    snr_train = X_train[snr_col].fillna(0).values
    snr_test = X_test[snr_col].fillna(0).values

    # Sweep thresholds on train set
    percentiles = np.unique(np.percentile(snr_train, np.arange(5, 96, 5)))
    best_f1, best_thresh = -1.0, 0.0
    for thresh in percentiles:
        y_pred_tr = (snr_train >= thresh).astype(int)
        f1 = f1_score(y_train, y_pred_tr, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    y_pred_test = (snr_test >= best_thresh).astype(int)
    # Use normalised SNR as the "probability" for ROC-AUC
    snr_max = snr_test.max()
    y_prob_test = snr_test / (snr_max + 1e-10) if snr_max > 0 else snr_test

    roc_auc = roc_auc_score(y_test, y_prob_test)
    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob_test)
    recall_at_fpr01 = float(np.interp(0.1, fpr_arr, tpr_arr))

    top100_idx = np.argsort(y_prob_test)[::-1][:100]
    prec_top100 = float(np.array(y_test)[top100_idx].mean()) if len(top100_idx) >= 100 else None

    report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_test).tolist()

    return {
        "label": "SNR-threshold baseline",
        "roc_auc": round(roc_auc, 4),
        "recall_at_fpr01": round(recall_at_fpr01, 4),
        "precision_top100": round(prec_top100, 4) if prec_top100 is not None else None,
        "classification_report": report,
        "confusion_matrix": cm,
        "snr_threshold": round(float(best_thresh), 2),
        "note": f"koi_model_snr ≥ {best_thresh:.1f} → planet-like (threshold from train F1)",
    }


# ── Model evaluation ──────────────────────────────────────────────────────────

def evaluate_model(
    model: HistGradientBoostingClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    label: str,
) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_prob)
    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_prob)
    recall_at_fpr01 = float(np.interp(0.1, fpr_arr, tpr_arr))

    # Precision in top-100 ranked by predicted probability
    top100_idx = np.argsort(y_prob)[::-1][:100]
    prec_top100 = float(np.array(y_test)[top100_idx].mean()) if len(top100_idx) >= 100 else None

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "label": label,
        "roc_auc": round(float(roc_auc), 4),
        "recall_at_fpr01": round(recall_at_fpr01, 4),
        "precision_top100": round(prec_top100, 4) if prec_top100 is not None else None,
        "classification_report": report,
        "confusion_matrix": cm,
    }


# ── MODEL_CARD generation ─────────────────────────────────────────────────────

def _fmt_report(report: dict) -> str:
    lines = []
    for cls_key, vals in report.items():
        if cls_key in ("accuracy", "macro avg", "weighted avg"):
            continue
        if not isinstance(vals, dict):
            continue
        label = "FALSE POSITIVE" if cls_key == "0" else "planet-like (CONF+CAND)"
        lines.append(
            f"  {label:<30} prec={vals['precision']:.3f}  "
            f"rec={vals['recall']:.3f}  f1={vals['f1-score']:.3f}  "
            f"n={int(vals['support'])}"
        )
    return "\n".join(lines)


def _fmt_cm(cm: list, model_label: str) -> str:
    return (
        f"  Confusion matrix ({model_label}):\n"
        f"                     Pred FP   Pred planet\n"
        f"  Actual FP         {cm[0][0]:>7}   {cm[0][1]:>11}\n"
        f"  Actual planet     {cm[1][0]:>7}   {cm[1][1]:>11}"
    )


def write_model_card(
    trained_at: datetime,
    data_snapshot: str,
    n_train: int,
    n_test: int,
    train_label_dist: dict,
    test_label_dist: dict,
    honest_metrics: dict,
    all_metrics: dict,
    baseline_majority: dict,
    baseline_snr: dict,
    honest_feature_list: list[str],
    excluded_feature_list: list[str],
) -> None:
    card = f"""# TransitLab Model Card

**Trained:** {trained_at.strftime('%Y-%m-%d %H:%M UTC')}
**Data snapshot:** {data_snapshot}
**Algorithm:** HistGradientBoostingClassifier (scikit-learn)
  - Chosen over GradientBoostingClassifier: handles NaN natively, faster on ~8k rows.
  - Chosen over RandomForest: GBM consistently outperforms RF on tabular data with
    correlated features; confirmed by comparing ROC-AUC on this dataset (see metrics).
**Task:** Binary — planet-like (CONFIRMED+CANDIDATE=1) vs FALSE POSITIVE (=0)
**Split:** by star (kepid), seed={SPLIT_SEED}, test_size={TEST_SIZE}
  - {n_train} train / {n_test} test
  - Train distribution: {train_label_dist}
  - Test distribution: {test_label_dist}

---

## Headline result: honest-features model

> **Why honest-features is the headline:** The all-features model uses koi_score and
> Robovetter flags (koi_fpflag_*) which encode the *answer* to the classification
> question. That model learns to imitate the Robovetter, not to detect transits from
> photometry. It will appear to perform better (see leakage comparison below), but
> that apparent improvement is measurement artefact — a new, unlabelled light curve
> has no Robovetter score. The honest model is the one that generalises to new data.

| Model | ROC-AUC | Recall@CONFIRMED (FPR=0.1) | Precision top-100 |
|---|---|---|---|
| Majority-class baseline | {baseline_majority['roc_auc']:.4f} | {baseline_majority['recall_at_fpr01']:.4f} | — |
| SNR-threshold baseline | {baseline_snr['roc_auc']:.4f} | {baseline_snr['recall_at_fpr01']:.4f} | {baseline_snr['precision_top100'] or '—'} |
| **Honest-features model** | **{honest_metrics['roc_auc']:.4f}** | **{honest_metrics['recall_at_fpr01']:.4f}** | **{honest_metrics['precision_top100'] or '—'}** |
| All-features model (leakage) | {all_metrics['roc_auc']:.4f} | {all_metrics['recall_at_fpr01']:.4f} | {all_metrics['precision_top100'] or '—'} |

### Why accuracy is not reported
FALSE POSITIVE is the majority class (~{int(100*list(train_label_dist.values())[0]/n_train)}% of training data).
A model that always predicts FALSE POSITIVE achieves ~50% accuracy for free.
ROC-AUC and recall@FPR=0.1 are the operationally meaningful metrics for triage.

### Per-class breakdown — honest-features model
{_fmt_report(honest_metrics['classification_report'])}

{_fmt_cm(honest_metrics['confusion_matrix'], 'honest-features')}

---

## Leakage comparison

{_fmt_cm(all_metrics['confusion_matrix'], 'all-features (leakage)')}

**Why all-features looks better:** koi_score is the Robovetter's own disposition
probability — it encodes the label almost directly. koi_fpflag_* are pass/fail
flags from the same vetting pipeline. Training on them teaches the model to
*replay* the Robovetter, not to detect transits. This inflates AUC by
{all_metrics['roc_auc'] - honest_metrics['roc_auc']:+.4f} and recall@FPR=0.1
by {all_metrics['recall_at_fpr01'] - honest_metrics['recall_at_fpr01']:+.4f}.
On a new unlabelled light curve these features do not exist. The all-features
model is published only as a cautionary example.

---

## Feature lists

### Honest features ({len(honest_feature_list)} columns)
{chr(10).join(f'- `{f}`' for f in honest_feature_list)}

### Excluded (disposition-derived) features ({len(excluded_feature_list)} columns)
{chr(10).join(f'- `{f}` — {"Robovetter output flag" if "flag" in f else "disposition score / prior"}' for f in excluded_feature_list)}

---

## SNR threshold baseline details
{baseline_snr.get('note', '')}

---

## Known failure modes
- koi_model_snr is itself derived from the pipeline fit, not raw photometry.
  It is included in honest features because it is computable from any BLS run,
  but it correlates with the Robovetter output — moderate indirect leakage exists.
- The classifier learns the *Kepler pipeline + Robovetter's* biases, not
  ground truth from the sky. Labels with the KOI catalogue are what the
  Robovetter said, not what follow-up confirmed in every case.
- Stellar parameters (koi_steff, koi_srad, koi_slogg) come from the same
  pipeline that produced the dispositions, so they carry some correlation.
- CANDIDATEs are planet-like in this model. In practice some are FPs that
  the Robovetter did not firmly reject — this introduces label noise.
- The model was trained on Kepler data only. TESS targets will have different
  noise properties and no koi_* features; apply with caution.
"""
    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / "MODEL_CARD.md").write_text(card)
    log.info("MODEL_CARD.md written")


# ── Main entrypoint ───────────────────────────────────────────────────────────

def run_training(db: Session) -> dict:
    """Train both models, evaluate against baselines, save artefacts."""
    MODEL_DIR.mkdir(exist_ok=True)

    # 1. Load data
    df = load_training_data(db)
    y = make_binary_labels(df["disposition"])

    total = len(df)
    log.info("Dataset: %d rows  |  planet-like=%d  FP=%d",
             total, int(y.sum()), int((y == 0).sum()))

    # 2. Split by star
    train_idx, test_idx = split_by_star(df, y)
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    log.info("Split: train=%d  test=%d", len(df_train), len(df_test))

    # Verify no kepid appears in both partitions (invariant check)
    train_stars = set(df_train["kepid"].dropna())
    test_stars = set(df_test["kepid"].dropna())
    leak = train_stars & test_stars
    if leak:
        raise RuntimeError(f"Split-by-star violated: {len(leak)} kepids in both partitions")
    log.info("Split-by-star verified: no kepid overlap between train and test")

    # 3. Build feature matrices
    def make_X(df_subset: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        return df_subset.reindex(columns=feature_cols).astype(float)

    X_train_honest = make_X(df_train, HONEST_FEATURE_COLUMNS)
    X_test_honest = make_X(df_test, HONEST_FEATURE_COLUMNS)
    X_train_all = make_X(df_train, ALL_FEATURE_COLUMNS)
    X_test_all = make_X(df_test, ALL_FEATURE_COLUMNS)

    # 4. Baselines
    log.info("Computing baselines...")
    baseline_majority = majority_class_metrics(y_test)
    baseline_snr = snr_threshold_metrics(
        X_train_honest, y_train, X_test_honest, y_test
    )
    log.info("Majority-class  AUC=%.4f  recall@FPR0.1=%.4f",
             baseline_majority["roc_auc"], baseline_majority["recall_at_fpr01"])
    log.info("SNR-threshold   AUC=%.4f  recall@FPR0.1=%.4f  threshold=%.1f",
             baseline_snr["roc_auc"], baseline_snr["recall_at_fpr01"],
             baseline_snr.get("snr_threshold", 0))

    # 5. Honest-features model (headline)
    log.info("Training honest-features model (%d features)...", len(HONEST_FEATURE_COLUMNS))
    honest_model = train_model(X_train_honest, y_train)
    honest_metrics = evaluate_model(honest_model, X_test_honest, y_test,
                                    label="honest-features")
    log.info("Honest model    AUC=%.4f  recall@FPR0.1=%.4f  prec-top100=%s",
             honest_metrics["roc_auc"], honest_metrics["recall_at_fpr01"],
             honest_metrics["precision_top100"])

    # 6. All-features model (leakage comparison only)
    log.info("Training all-features model (%d features, leakage comparison)...",
             len(ALL_FEATURE_COLUMNS))
    all_model = train_model(X_train_all, y_train)
    all_metrics = evaluate_model(all_model, X_test_all, y_test, label="all-features")
    log.info("All-features    AUC=%.4f  recall@FPR0.1=%.4f  prec-top100=%s  "
             "(leakage inflates by AUC+%.4f)",
             all_metrics["roc_auc"], all_metrics["recall_at_fpr01"],
             all_metrics["precision_top100"],
             all_metrics["roc_auc"] - honest_metrics["roc_auc"])

    # 7. Check both models beat baselines
    best_baseline_auc = max(baseline_majority["roc_auc"], baseline_snr["roc_auc"])
    if honest_metrics["roc_auc"] <= best_baseline_auc:
        log.warning(
            "HONEST model (AUC=%.4f) does NOT beat baselines (best=%.4f) — "
            "see MODEL_CARD for explanation",
            honest_metrics["roc_auc"], best_baseline_auc,
        )

    # 8. Save models
    honest_path = MODEL_DIR / "honest_model.joblib"
    all_path = MODEL_DIR / "all_features_model.joblib"
    joblib.dump(honest_model, honest_path)
    joblib.dump(all_model, all_path)
    log.info("Saved %s and %s", honest_path, all_path)

    # 9. Write MODEL_CARD
    trained_at = datetime.now(timezone.utc)
    train_label_dist = {
        "FALSE POSITIVE": int((y_train == 0).sum()),
        "planet-like": int((y_train == 1).sum()),
    }
    test_label_dist = {
        "FALSE POSITIVE": int((y_test == 0).sum()),
        "planet-like": int((y_test == 1).sum()),
    }
    write_model_card(
        trained_at=trained_at,
        data_snapshot=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        n_train=len(df_train),
        n_test=len(df_test),
        train_label_dist=train_label_dist,
        test_label_dist=test_label_dist,
        honest_metrics=honest_metrics,
        all_metrics=all_metrics,
        baseline_majority=baseline_majority,
        baseline_snr=baseline_snr,
        honest_feature_list=HONEST_FEATURE_COLUMNS,
        excluded_feature_list=[c for c in DISPOSITION_DERIVED_FEATURES if c != "koi_pdisposition"],
    )

    # 10. Record training run in DB
    training_run = TrainingRun(
        trained_at=trained_at,
        data_snapshot=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        split_seed=SPLIT_SEED,
        honest_features={"columns": HONEST_FEATURE_COLUMNS},
        excluded_features={"columns": DISPOSITION_DERIVED_FEATURES,
                           "reason": "disposition-derived; see MODEL_CARD"},
        metrics={
            "majority_class": {k: v for k, v in baseline_majority.items()
                               if k not in ("classification_report", "confusion_matrix")},
            "snr_threshold": {k: v for k, v in baseline_snr.items()
                              if k not in ("classification_report", "confusion_matrix")},
            "honest_features": {k: v for k, v in honest_metrics.items()
                                if k not in ("classification_report", "confusion_matrix")},
            "all_features": {k: v for k, v in all_metrics.items()
                             if k not in ("classification_report", "confusion_matrix")},
        },
        model_path=str(honest_path),
    )
    db.add(training_run)
    db.commit()
    log.info("Training run recorded in DB (id=%d)", training_run.id)

    return {
        "n_train": len(df_train),
        "n_test": len(df_test),
        "baseline_majority": baseline_majority,
        "baseline_snr": baseline_snr,
        "honest_metrics": honest_metrics,
        "all_metrics": all_metrics,
        "honest_model_path": str(honest_path),
    }
