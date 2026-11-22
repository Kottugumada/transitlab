# TransitLab — Build Spec for Claude

**Project name:** TransitLab
**One-liner:** One photometry platform with two modes that share a single transit engine: **Survey mode** hunts for exoplanet transits in real NASA Kepler/TESS light curves (MAST) and triages them with a classifier trained on NASA's own labelled dispositions; **Backyard mode** predicts which known exoplanet transits are observable from Morgan Hill tonight and (v1.5) fits transits in the user's own Seestar data.

**The design rule (hard requirement, decided by Karthik 2026-11-21):**
> **The app must be complete and useful with NO user-supplied data.** The NASA archive is the content. Seestar ingestion (v1.5) is an optional adapter; no v1 feature may depend on it, and the app must never present itself as waiting for, or diminished without, user data.

**Why this project:** Real telescope data (not summary APIs), a real detection algorithm (BLS — the one transit surveys actually use), supervised ML with genuine ground-truth labels, and a vetting UI modelled on what professional astronomers look at. Same $0 deployment family as FireCluster/AuroraWatch: batch compute + Neon free tier + Render free tier.

**Engineering standard (Karthik's standing rule, 2026-11-21): $0 is a design constraint, never an excuse for sloppy engineering.** Free tiers are achieved by *architecture* (batch instead of always-on, free-tier managed DB instead of self-hosted, static/on-demand serving), never by cutting corners on quality. Non-negotiables at $0, same as at any budget: planned build order with a verified done-check per step; clean separation (pure engine library vs ingestion vs API); tests with committed fixtures that run offline; idempotent, resumable batch jobs; structured logging and run records; explicit config (no magic numbers in code); documented design decisions and known limitations in the README; honest validation results, published whatever they show. If a quality practice and the $0 target ever appear to conflict, stop and flag it to Karthik — do not silently trade quality for cost.

---

## 1. Scope

### v1 — Survey mode (the whole product)
1. **Catalogue sync (batch):** pull the Kepler cumulative KOI table from the NASA Exoplanet Archive (labels + features) and the confirmed-planets table (`ps`) — store in Neon
2. **Classifier training (batch, offline):** train a model to predict KOI disposition (CONFIRMED / CANDIDATE / FALSE POSITIVE) from catalogue features; evaluate honestly (§5); save model + metrics report
3. **Light-curve pipeline:** for any Kepler/TESS target, fetch its light curve from MAST via Lightkurve, clean + detrend it, run BLS, phase-fold, fit a transit model, extract vetting features (§4)
4. **Hunt mode (batch):** run the pipeline over a configurable target list (start: a few hundred Kepler targets with known KOIs, then unlabelled TESS targets), score each detection with the classifier + BLS statistics, store ranked candidates
5. **Vetting UI:** candidate list ranked by score; per-candidate page with raw light curve, phase-folded curve with fit overlay, odd/even transit comparison, and the feature table — the page a human uses to say "planet-like" or "false positive"
6. **Known-system explorer:** look up any confirmed planet (from `ps`), see its light curve folded at its *known* period — the guaranteed-satisfying demo path that works even before the hunt finds anything
7. FastAPI API (§7), tests with committed fixtures, README with an honest results section (§6)

### v1 also includes — Backyard prediction page (no user data involved)
8. From `ps` ephemerides (period + mid-transit time + duration + depth + host magnitude), compute which known transits occur during tonight's astronomical darkness at the configured location (default Morgan Hill), ranked by observability (depth, host brightness, transit fully/partially in darkness, moon, cloud via Open-Meteo — reuse the AuroraWatch patterns if available). This page is informational only in v1.

### v1.5 — Backyard photometry (OPTIONAL adapter, separate milestone, only after v1 acceptance)
9. Upload a folder of Seestar FITS subs (or a Seestar stacked FITS + subs) for a target/date → differential photometry: plate-solve or use FITS WCS to locate the host star, aperture photometry on host + auto-selected comparison stars, build the light curve, feed it to the **same** transit-fit code as Survey mode with period/time constrained to the prediction → verdict: dip detected / not detected / inconclusive, with measured depth ± uncertainty vs the archive depth.
   v1.5 has its own acceptance test (§9) and must not be started early. If it is never built, v1 remains a complete product (per the design rule).

### Non-goals
- No claim of discovering new planets. Language everywhere: "candidate", "planet-like signal", "consistent with". The hunt re-finds *known* signals first (that's the validation), then ranks unlabelled ones as candidates for human vetting — nothing more.
- No raw light curves stored in the DB (features, fold plots data, and metrics only — curves are re-fetchable from MAST; cache downloads on disk in batch runs).
- No CNN on raw pixels/light curves in v1. Feature-based classification first (interpretable, trainable in minutes); a 1-D CNN on folded curves is a named v2 experiment with a required head-to-head comparison.

---

## 2. Data Sources (verified 2026-11-21)

### 2.1 NASA Exoplanet Archive — TAP service (no key, no auth)
- Endpoint: `https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=<ADQL>&format=csv`
- **KOI cumulative table — `cumulative`** (~9,564 KOIs): the training set. Key columns:
  - Label: `koi_disposition` ∈ {CONFIRMED, CANDIDATE, FALSE POSITIVE}, plus `koi_pdisposition`, `koi_score`
  - Features (use these families): `koi_period`, `koi_time0bk`, `koi_duration`, `koi_depth`, `koi_ror` (Rp/R*), `koi_impact`, `koi_model_snr`, `koi_max_mult_ev`, `koi_bin_oedp_sig` (odd-even depth significance), `koi_limbdark_mod`, stellar: `koi_steff`, `koi_srad`, `koi_slogg`, `koi_kepmag`, and the Robovetter-style vetting flags present in the table (centroid, ghost, etc. — enumerate at build time from the table's column docs and record which were used)
  - IDs: `kepid`, `kepoi_name`, `kepler_name`, `koi_tce_plnt_num`
- **Planetary Systems table — `ps`** (use `where default_flag=1`): confirmed planets for the explorer + Backyard predictions. Key columns: `pl_name`, `hostname`, `pl_orbper`, `pl_tranmid`, `pl_trandur`, `pl_trandep`, `pl_rade`, `pl_bmasse`/`pl_bmassj`, `st_spectype`, `sy_vmag`, `sy_kmag`, `ra`, `dec`, `disc_facility`.
- **TOI table** (`toi`, TESS Objects of Interest): optional extra candidate source for hunt mode phase 2 — verify exact table name/columns at build time.
- TAP traps: URL-encode the ADQL; request only needed columns (never `select *` — `ps` has hundreds); TAP can be slow — catalogue sync is a batch job with retries and a local CSV cache, never an API-request-path call.

### 2.2 MAST via Lightkurve — light curves (no key for public data)
- Python `lightkurve`: `search_lightcurve(target, mission='Kepler'|'TESS')` → download → `to_lightcurve()` → use **PDCSAP** flux where available (it's the mission's own corrected photometry), fall back to SAP and say so.
- Prefer Kepler for v1 hunt validation (long baselines, many transits, KOI labels exist for the same targets — the loop closes perfectly). TESS sectors are ~27 days: fine for short-period planets, note the limitation.
- Cache downloaded FITS under `data/cache/` keyed by target+mission+author; batch jobs must be resumable (skip already-processed targets).
- Time systems: Kepler/TESS times are in BKJD/BTJD — Lightkurve handles conversion; store everything as BJD/BTJD consistently and label axes honestly. Do not mix time systems in the fold.

### 2.3 Open-Meteo (Backyard page only) — free, no key
Cloud cover by layer for tonight at the location, same usage pattern as the AuroraWatch spec §2.3.

---

## 3. Architecture ($0 batch + on-demand API)

```
  Batch (GitHub Actions, manual/scheduled — NOT high-frequency):
    catalogue_sync   TAP → Neon (KOI + ps)                      [weekly or on demand]
    train            KOI features → model + metrics report      [after each sync]
    hunt             target list → Lightkurve/MAST → engine → candidates in Neon
                              │
  Shared engine (pure library, no I/O):  clean → detrend → BLS → fold → fit → features → score
                              │
  Neon Postgres (free tier, own database) ──▶ FastAPI (Render free) ──▶ UI
                                                  ▲
  v1.5 only: user FITS upload ──▶ photometry adapter ──▶ (same engine, constrained fit)
```

Stack (fixed): Python 3.12; `lightkurve`, `astropy` (BLS = `astropy.timeseries.BoxLeastSquares`, Lomb-Scargle for variable-star sanity checks), `numpy`, `pandas` (batch only), `scikit-learn` (gradient boosting or random forest — pick in training step and justify from the metrics), FastAPI, SQLAlchemy 2, Alembic, `matplotlib` for server-side fold plots (render to PNG/SVG in batch, store the *data* for client plotting where cheap — decide in Step 5, default: store binned fold data as JSON, plot client-side). Frontend: static HTML + vanilla JS (a small plotting lib via CDN is fine). No GPU, no paid services, no Redis.

Compute honesty: BLS over hundreds of Kepler curves is CPU-minutes-to-hours, not seconds. Hunt mode is a batch job with a target budget (config: max targets per run, resumable), never a synchronous API call. Training on ~9.5k catalogue rows takes seconds-to-minutes — the light curves are *not* the training input in v1; catalogue features are. Keep those two facts straight in the README; conflating them is the classic beginner error in this domain.

---

## 4. The Transit Engine (shared core — pure functions, heavily tested)

Pipeline per light curve, each step a separate function with its own tests:
1. **Load + select:** PDCSAP preferred; drop NaN/quality-flagged cadences (use the mission quality bitmask via Lightkurve defaults; record how many cadences were dropped).
2. **Detrend:** `flatten()` (Savitzky-Golay via Lightkurve) with a window long compared to transit duration; *also* keep a lightly-processed copy — over-flattening eats shallow transits, so window length is a config constant with a comment and a test on an injected synthetic transit (see §6).
3. **BLS search:** period grid appropriate to the baseline (config: 0.5 d to ~1/3 of baseline), durations 0.05–0.3 × period-scaled set (use astropy's `autopower` as the starting point). Record best period, power, depth, duration, epoch, and the BLS SNR.
4. **Phase fold + bin** at the best period; fit a **trapezoid transit model** (v1) to the folded curve → fitted depth/duration/epoch with uncertainties from the fit residuals. (`batman` physical models are a v2 upgrade; trapezoid is honest and sufficient for vetting features.)
5. **Vetting features (the classifier's eyes):**
   - BLS power / model SNR; fitted depth and depth significance
   - **Odd/even depth difference** (in units of its own σ) — the classic eclipsing-binary tell
   - **Secondary eclipse search:** fold at phase 0.5, measure any dip's depth/significance — a secondary ≈ EB or hot Jupiter emission; record, don't over-interpret
   - Transit shape: fitted ingress/egress fraction vs total duration (V-shaped → grazing EB suspicion; U-shaped → planet-like)
   - Number of transits observed; per-transit depth scatter
   - Centroid/difference-image features: **not available** from light curves alone in v1 — explicitly out of scope; the classifier must not pretend to have them (the KOI catalogue features used in training, §5, are a *separate* feature set — see the warning there)
6. **Score:** two outputs, never merged into one opaque number:
   - `bls_score` — detection statistics from this curve alone
   - `vetting_features` — the table above, shown raw in the UI
   The catalogue classifier (§5) scores *catalogue rows*; applying it to hunt outputs is allowed only for the feature intersection the two share, and the UI must label such scores "catalogue-model estimate". Honesty in labelling is an acceptance criterion.

---

## 5. The Classifier (catalogue model)

- **Task:** 3-class (CONFIRMED / CANDIDATE / FALSE POSITIVE) or binary (planet-like = CONFIRMED+CANDIDATE vs FALSE POSITIVE) — build binary first (it's the operationally useful question), report 3-class as a secondary experiment if time permits.
- **Split discipline (required):** split by **star (kepid)**, not by row — multiple KOIs share a host, and row-splitting leaks. Stratified, fixed seed, split recorded in the metrics report.
- **Leakage audit (required, this is the heart of the exercise):** some catalogue columns are *derived from the disposition process itself* (e.g. `koi_score`, `koi_pdisposition`, Robovetter flags that effectively encode the answer). Train two models: (a) all-features, (b) **honest-features** (only quantities measurable from photometry + stellar params, excluding disposition-derived columns — enumerate the excluded list in code with reasons). Report both; the honest model is the headline result. Expect (a) to look suspiciously better — explaining *why* in the README is mandatory.
- **Metrics:** per-class precision/recall, confusion matrix, ROC-AUC, and — the number that matters for triage — **recall on CONFIRMED at a fixed false-positive rate**, plus precision in the top-100 ranked candidates. Accuracy alone is not an acceptable headline metric on this imbalanced set (FALSE POSITIVE is the majority class).
- **Baselines (required):** majority-class, and a single-feature `koi_model_snr` threshold. The model must beat both or the README says so plainly.
- Persist: model file (joblib) in the repo's `models/` with a `MODEL_CARD.md` — training date, data snapshot, split, features (honest list), metrics, baselines, known failure modes.

---

## 6. Validation (required before v1 is "done")
1. **Synthetic injection test:** inject trapezoid transits of known depth/period into 20 real quiet Kepler curves → engine must recover period within 1% and depth within 25% for depth ≥ 3× noise; report the recovery table. Also inject an EB-like signal (deep, V-shaped, strong odd/even) → vetting features must flag it via odd/even or shape.
2. **Known-system recovery:** run hunt mode over ≥50 Kepler targets with CONFIRMED planets → report how many are re-detected at the correct period (top-1 BLS match within 1%) and where the misses went (too shallow, single transit, detrending casualty). This table is the product's credibility.
3. **Classifier report** per §5, in the README, including the leakage comparison.
4. **Explorer spot-check:** 3 famous systems (e.g. Kepler-186f's host, a hot Jupiter like HAT-P-7b / Kepler-7b, TRAPPIST-1 via TESS if curves are available) render folded curves at their archive periods with visible dips.

---

## 7. Data Model (Neon, own database)

```sql
koi_catalogue (kepoi_name PK, kepid, kepler_name, disposition, pdisposition, koi_score,
               period, time0, duration, depth, ror, impact, model_snr, odd_even_sig,
               steff, srad, kepmag, features jsonb, synced_at)
ps_planets (pl_name PK, hostname, period, tranmid, trandur, trandep, rade, sy_vmag, ra, dec, synced_at)
targets (target_id PK, mission, kepid NULL, tic_id NULL, label_disposition NULL, hunt_status, updated_at)
detections (id PK, target_id FK, run_id, period, epoch_bjd, duration_h, depth_ppm,
            bls_power, bls_snr, fit_depth_ppm, fit_depth_sigma,
            odd_even_sigma, secondary_depth_ppm, secondary_sigma, shape_ratio,
            n_transits, depth_scatter, vetting jsonb, fold_data jsonb,  -- binned folded curve for UI
            catalogue_model_score double precision NULL, created_at)
hunt_runs (id, started_at, finished_at, n_targets, n_completed, n_detections, config jsonb, status)
training_runs (id, trained_at, data_snapshot, split_seed, honest_features jsonb, excluded_features jsonb,
               metrics jsonb, model_path)
backyard_predictions (date_local, pl_name, hostname, ingress, mid_transit, egress,
                      depth_pct, sy_vmag, observable_rank, reasons jsonb, computed_at)  -- v1 page
-- v1.5 only:
user_observations (id, pl_name, date_local, n_subs, status, measured_depth_pct, depth_sigma,
                   verdict, curve jsonb, created_at)
```

---

## 8. API (FastAPI, base `/api/v1`)

| Endpoint | Returns |
|---|---|
| `GET /health` | `{status, db, last_sync_at, last_hunt_at, model_version}` |
| `GET /candidates?min_score=&mission=&sort=score` | Ranked detections with target, period, depth, SNR, score, label (if known) |
| `GET /candidates/{id}` | Full vetting page data: fold data, fit params, vetting feature table, raw-curve summary stats |
| `GET /systems/{pl_name}` | Confirmed-planet explorer: archive params + folded-curve data at archive period (computed in batch or on first request then cached in DB) |
| `GET /targets/{id}/lightcurve?folded=true` | Binned curve data for plotting (never raw FITS) |
| `GET /model` | Model card contents as JSON (metrics, honest feature list, baselines) |
| `GET /backyard/tonight` | Ranked observable transits for tonight at the configured location, with reasons |
| `POST /backyard/observations` (v1.5) | Upload FITS set → async job id; `GET /backyard/observations/{id}` for status/verdict |

---

## 9. UI
- **Candidates page:** ranked table (score, target, period, depth, SNR, transits, known label badge), click → vetting page.
- **Vetting page:** folded curve with trapezoid fit overlay, odd/even folded curves side by side (this one visual teaches the whole EB problem), secondary-phase view, feature table with plain-English glosses ("Odd/even difference 4.2σ — transits alternate deep/shallow, typical of an eclipsing binary").
- **Explorer page:** search a confirmed planet, see its real folded light curve.
- **Backyard page:** tonight's observable transits table + (v1.5) upload + verdict history.
- Global footer: "Candidates are signals for human vetting, not discovered planets · Data: NASA Exoplanet Archive & MAST (Kepler/TESS)".

---

## 10. Repo Layout

```
transitlab/
  .github/workflows/           # catalogue sync + hunt as manual/scheduled batch workflows
  docker-compose.yml           # local dev: api + postgis
  .env.example                 # DATABASE_URL=, LOCATION_* (backyard), HUNT_MAX_TARGETS=
  README.md                    # incl. §5 leakage write-up + §6 validation tables + MODEL_CARD summary
  models/MODEL_CARD.md
  pyproject.toml               # lightkurve, astropy, scikit-learn, fastapi...
  app/
    config.py
    archive_client.py          # TAP queries (column lists explicit), CSV cache, retries
    mast_client.py             # lightkurve wrapper, FITS cache, resumable target iteration
    engine/                    # PURE library — no DB, no network
      clean.py, detrend.py, bls.py, fold.py, fit.py, features.py
    catalogue_sync.py, train.py, hunt.py, backyard_predict.py
    batch.py                   # entrypoints per job type: --job sync|train|hunt|predict
    api/main.py, api/routes.py, api/schemas.py
    static/index.html, static/app.js
  photometry/                    # v1.5 ONLY — do not create in v1
  tests/
    fixtures/                  # 2 short real Kepler curve excerpts (FITS or CSV of time/flux), KOI CSV excerpt
    test_engine_synthetic.py   # §6.1 injection tests (offline, on fixtures)
    test_features.py, test_train_split.py (split-by-star assertion), test_api.py
  data/cache/                    # gitignored
```

---

## 11. Build Order (verify each step)
1. **Skeleton + DB:** Compose, `/health`, Neon database, Alembic. ✅ Health + cloud connection pass.
2. **Catalogue sync:** TAP pulls for `cumulative` + `ps` (explicit columns), stored + counted. ✅ Row counts in range (~9.5k KOI), label distribution printed and sane.
3. **Classifier:** split-by-star, leakage audit (both models), baselines, metrics, MODEL_CARD. ✅ Metrics beat baselines (or README-grade explanation ready); honest feature list exists in code.
4. **Engine on fixtures:** clean → detrend → BLS → fold → trapezoid fit → features, all pure + tested. ✅ Synthetic injection tests (§6.1) pass offline.
5. **Explorer:** one confirmed system end-to-end (fetch → fold at archive period → UI page with visible dip). ✅ Spot-check #1 renders.
6. **Hunt batch:** resumable, budgeted; run the §6.2 recovery set (≥50 confirmed hosts). ✅ Recovery table produced; candidates stored with fold data.
7. **API + vetting UI.** ✅ Candidates list + vetting page (incl. odd/even view) work from real hunt output.
8. **Backyard prediction page.** ✅ Tonight's list computes for Morgan Hill with reasons; sanity-checked against one hand-verified transit (cross-check one prediction against the Exoplanet Archive / transit calculators and record it).
9. **Deploy + README:** Render free for API, batch via GitHub Actions, README with §5/§6 results written honestly. ✅ Live URL serves real candidates.
10. **(v1.5, separate approval)** Backyard photometry adapter per §1.9 + §12.

---

## 12. v1.5 Acceptance (Backyard photometry — only when built)
- Pipeline ingests a real Seestar FITS set, produces a light curve whose out-of-transit scatter is reported honestly (expect ~0.5–2% per point binned — state the measured number, not a hope).
- On a known bright hot-Jupiter transit (host V ≲ 10, depth ≥ 1% — first-target shortlist to be researched at v1.5 start, e.g. from the archive's brightest/deepest `ps` entries observable from Morgan Hill): verdict logic must return "dip detected" only if the fitted dip is ≥3σ, at the predicted time within its uncertainty, with depth consistent with the archive value within 2σ. Otherwise "inconclusive" is the correct and acceptable answer — **a forced detection is a failed build.**
- Comparison-star selection, aperture sizes, and any frames rejected (clouds/drift) are logged and shown. No silent data massage.

---

## 13. Known Limitations (state in README, in substance)
- Kepler KOI dispositions are the Kepler pipeline's + Robovetter's judgement, not ground truth from the sky — the classifier learns to imitate that vetting, biases included.
- Catalogue features and light-curve features are different views; scores are labelled by which produced them (§4.6).
- BLS assumes periodic box-shaped dips: single transits, very long periods, and strongly variable stars will be missed or mangled. TESS's 27-day sectors cap detectable periods.
- Detrending can destroy shallow or long-duration transits; the injection tests bound, but do not eliminate, this.
- A high score means "worth a human's look" — professional confirmation needs follow-up (radial velocity, centroid analysis, statistical validation) that this app does not do.

---

## 14. Copy-paste prompt to start Claude

> Build the project in this spec file step by step, in the Build Order in §11. Stop after each numbered step, run its tests / done-check, show me the output, and wait for my go-ahead before the next step. Three rules outrank everything else: (1) the design rule at the top — v1 must be complete with no user-supplied data, and the `photometry/` v1.5 work must not be started until I approve it separately; (2) the §5 leakage audit — train both models, headline the honest one, and split by star, never by row; (3) §6 validation — v1 is not done until the injection tests and the known-system recovery table exist and are in the README, whatever numbers they show. Engine code is a pure library with no network or DB access. If a TAP table name or column differs from the spec, check the Archive's column documentation and tell me what you found rather than guessing silently. Commit after each step.

---

## 15. Acceptance Criteria for v1
- [ ] Complete and useful with zero user data (design rule) — every v1 page works from NASA data alone
- [ ] Catalogue sync stores ~9.5k labelled KOIs + confirmed planets; counts/label distribution logged
- [ ] Classifier: split-by-star verified by test; honest vs all-features metrics + baselines in MODEL_CARD and README
- [ ] Engine injection tests pass at the §6.1 thresholds; EB-like injection flagged by vetting features
- [ ] Known-system recovery table (≥50 confirmed hosts) in README with honest hit rate
- [ ] Candidates API + vetting UI (fold + fit overlay, odd/even view, glossed feature table) live on Render
- [ ] Backyard tonight page computes ranked observable transits for Morgan Hill, one prediction hand-verified
- [ ] No "discovered planet" language anywhere in UI or README
- [ ] $0 running cost: batch on GitHub Actions, Neon free tier, Render free tier

*Spec prepared 2026-11-21. TAP endpoint, `cumulative` table size (~9,564 KOIs) and key columns, and `ps` default_flag usage verified against NASA Exoplanet Archive documentation on that date; TOI table details and Robovetter flag columns are flagged in-spec for build-time verification. Seestar transit detectability (v1.5) is framed as a measured experiment with an explicit "inconclusive is acceptable" rule, not a promised result.*
