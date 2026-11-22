# TransitLab

One photometry platform, two modes, one transit engine.

**Survey mode** hunts for exoplanet transit signals in real NASA Kepler/TESS light curves from MAST and triages detections with a classifier trained on NASA's own KOI dispositions.  
**Backyard mode** predicts which known exoplanet transits are observable tonight from a configured location.

> **All v1 features work with zero user-supplied data.** The NASA archive is the content.

---

## Quick start

```bash
cp .env.example .env        # fill in DATABASE_URL and optionally location
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.api.main:app --reload
# open http://localhost:8000/api/v1/health
```

---

## Architecture

```
Batch (GitHub Actions):
  catalogue_sync  TAP → Neon (KOI + ps)              [weekly or on-demand]
  train           KOI features → model + metrics      [after each sync]
  hunt            target list → MAST → engine → candidates

Shared engine (pure library, no I/O):
  clean → detrend → BLS → fold → fit → features → score

Neon Postgres → FastAPI (Render free) → static UI
```

$0 running cost: batch on GitHub Actions, Neon free-tier Postgres, Render free tier.

---

## Validation results

> *Filled in at Step 4 (engine) and Step 6 (hunt) of the build order.*

### §6.1 Synthetic injection recovery

| Injected depth | Period (d) | Recovered period (within 1%)? | Depth error | Notes |
|---|---|---|---|---|
| (pending) | | | | |

### §6.2 Known-system recovery (≥50 confirmed Kepler hosts)

| Total targets | Correct period (top-1, within 1%) | Miss rate | Main miss reasons |
|---|---|---|---|
| (pending) | | | |

---

## Classifier results

> *Filled in at Step 3 of the build order.*

**Headline: honest-features model** (photometry + stellar params only, no Robovetter-derived columns).

| Model | ROC-AUC | Recall@CONFIRMED (FPR=0.1) | Precision top-100 |
|---|---|---|---|
| Majority-class baseline | — | — | — |
| SNR-threshold baseline | — | — | — |
| Honest-features model | — | — | — |
| All-features model | — | — | — |

See [models/MODEL_CARD.md](models/MODEL_CARD.md) for the full feature list, exclusion reasons, and confusion matrices.

---

## Known limitations

- Kepler KOI dispositions are the Kepler pipeline's + Robovetter's judgement, not independent ground truth from the sky. The classifier learns to imitate that vetting, biases included.
- Catalogue features and light-curve features are different views. Scores are labelled by which produced them.
- BLS assumes periodic box-shaped dips: single transits, very long periods, and strongly variable stars will be missed or mangled. TESS 27-day sectors cap detectable periods.
- Detrending can destroy shallow or long-duration transits. The injection tests bound, but do not eliminate, this.
- A high score means "worth a human's look" — professional confirmation needs follow-up (radial velocity, centroid analysis, statistical validation) that this app does not perform.

---

## Data sources

- NASA Exoplanet Archive TAP service — no key required
- MAST via Lightkurve — no key for public Kepler/TESS data
- Open-Meteo — cloud cover for Backyard page, no key required
