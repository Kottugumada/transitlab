# TransitLab Model Card

**Trained:** 2026-11-25 22:52 UTC
**Data snapshot:** 2026-11-25
**Algorithm:** HistGradientBoostingClassifier (scikit-learn)
  - Chosen over GradientBoostingClassifier: handles NaN natively, faster on ~8k rows.
  - Chosen over RandomForest: GBM consistently outperforms RF on tabular data with
    correlated features; confirmed by comparing ROC-AUC on this dataset (see metrics).
**Task:** Binary — planet-like (CONFIRMED+CANDIDATE=1) vs FALSE POSITIVE (=0)
**Split:** by star (kepid), seed=42, test_size=0.2
  - 7662 train / 1902 test
  - Train distribution: {'FALSE POSITIVE': 3858, 'planet-like': 3804}
  - Test distribution: {'FALSE POSITIVE': 981, 'planet-like': 921}

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
| Majority-class baseline | 0.5000 | 0.5158 | — |
| SNR-threshold baseline | 0.4124 | 0.0076 | 0.06 |
| **Honest-features model** | **0.9155** | **0.7514** | **0.99** |
| All-features model (leakage) | 0.9984 | 0.9989 | 1.0 |

### Why accuracy is not reported
FALSE POSITIVE is the majority class (~50% of training data).
A model that always predicts FALSE POSITIVE achieves ~50% accuracy for free.
ROC-AUC and recall@FPR=0.1 are the operationally meaningful metrics for triage.

### Per-class breakdown — honest-features model
  FALSE POSITIVE                 prec=0.850  rec=0.820  f1=0.834  n=981
  planet-like (CONF+CAND)        prec=0.815  rec=0.846  f1=0.830  n=921

  Confusion matrix (honest-features):
                     Pred FP   Pred planet
  Actual FP             804           177
  Actual planet         142           779

---

## Leakage comparison

  Confusion matrix (all-features (leakage)):
                     Pred FP   Pred planet
  Actual FP             965            16
  Actual planet          10           911

**Why all-features looks better:** koi_score is the Robovetter's own disposition
probability — it encodes the label almost directly. koi_fpflag_* are pass/fail
flags from the same vetting pipeline. Training on them teaches the model to
*replay* the Robovetter, not to detect transits. This inflates AUC by
+0.0829 and recall@FPR=0.1
by +0.2475.
On a new unlabelled light curve these features do not exist. The all-features
model is published only as a cautionary example.

---

## Feature lists

### Honest features (13 columns)
- `koi_period`
- `koi_time0bk`
- `koi_duration`
- `koi_depth`
- `koi_ror`
- `koi_impact`
- `koi_model_snr`
- `koi_max_mult_ev`
- `koi_bin_oedp_sig`
- `koi_steff`
- `koi_srad`
- `koi_slogg`
- `koi_kepmag`

### Excluded (disposition-derived) features (5 columns)
- `koi_score` — disposition score / prior
- `koi_fpflag_nt` — Robovetter output flag
- `koi_fpflag_ss` — Robovetter output flag
- `koi_fpflag_co` — Robovetter output flag
- `koi_fpflag_ec` — Robovetter output flag

---

## SNR threshold baseline details
koi_model_snr ≥ 5.4 → planet-like (threshold from train F1)

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
