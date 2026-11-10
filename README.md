# Regime-Dependent Price Discovery in the FX Majors Triangle

Which leg of the EUR/USD–USD/JPY–EUR/JPY no-arbitrage triangle incorporates new
information first, and does that leadership **migrate** across trading sessions and
volatility regimes in a way you could identify in real time?

**Finding.** The quoted cross (EUR/JPY) leads its synthetic replica
(EUR/USD + USD/JPY). The *direction* holds under every staleness control, and
Hayashi–Yoshida on raw ticks agrees; the *size* depends on the control — the cross
carries **57–65%** of the permanent component (0.65 on a 1 s grid, 0.57 in refresh
time). The falsifiable claim — real-time-identifiable migration across sessions or
volatility states — is **not supported**: no pre-registered migration hypothesis
survives Holm. The null is not clean either: in refresh time the cross's share rises
through the London trading day, a post-hoc pattern that needs a frozen-spec test.
Three apparent regime effects (a Tokyo share bump, a "deep-window" speed advantage,
faster correction under stress) were staleness or time-support artifacts and were
withdrawn; applying the same controls to the headline shrinks it too. The contribution
is the staleness discipline, applied to everything — including the headline.

## The triangle

The three log mid-prices obey `p_EURJPY = p_EURUSD + p_USDJPY`, so the basis
`z = p_EURJPY − p_EURUSD − p_USDJPY` is stationary and arbitrage-bounded. The
cointegrating vector is known a priori (β = (1, −1, −1)), so it is **imposed, not
estimated** — no Johansen — and the whole estimation budget goes into the interesting
question: whose adjustment loading is ~zero (the leader), and whether that changes
with the clock and with volatility.

One design point matters throughout: three instruments share **two** common trends,
so a single-number Hasbrouck information share in the trivariate frame is wrong.
Discovery is measured in the **bivariate frame** — quoted cross vs synthetic cross —
which has one common trend, where textbook Gonzalo–Granger and Hasbrouck apply.

One exactness caveat: the identity holds for rates with the **same value date**. Spot
settles T+2 on each pair's own holiday calendar, so around Japanese, euro-area and
year-end holidays the legs settle on different days and z carries the forward points
for the mismatch — a level shift of 0.37–3.7 bp on 25 of 154 days. The holiday
calendar predicts exactly those 25 days (R² = 0.99, no intercept, implying a USD–EUR
rate gap of ≈1.9%). The shift is constant within a day, so each window's intercept
absorbs it.

## What I found

**Basis (01).** `z` is near-I(0) and tiny: over the FX trading week its std is 0.31 bp,
and **0.068 bp in liquid hours** (07–17 UTC); ADF rejects a unit root and KPSS does not
reject stationarity. The extreme (−32.8 bp) is the first second of the FX week, when
quotes reopen thin; the dollar legs post a new quote in only 31–35% of 1 s bins
(EUR/JPY 58%). This justifies
imposing β rather than testing for it.

**Who error-corrects (02).** Every leg error-corrects with the expected signs — when
the cross is rich (z > 0) it falls (α ≈ −0.079) and the dollar legs rise. The basis
closes with a dynamic first-passage half-life of **≈ 8 s** (the AR(1) proxy, 2.7 s,
ignores the lag dynamics and understates). A Gonzalo–Granger preview puts ≈ 64% of the
permanent component on the quoted cross — mildly against the "deep dollar legs lead"
prior.

**Static discovery (03).** Over 7 months / 151 windows, **GG_cross ≈ 0.65**
[IQR 0.58, 0.73] on the 1 s grid, stable across frequency (0.60 @100ms → 0.65 @5s).
Hasbrouck IS is reported only as Cholesky **bounds**, and they are wide and
frequency-dependent — bound width 0.53 @100ms → 0.97 @5s (IS_cross [0.07, 0.98] at
1 s) — because cross–synth correlation rises with bar width (the **Epps effect**,
shown not hidden). GG is the primary, point-identified number; IS is informative only
at fine grids.

**Lead-lag in real time (04).** On the raw async ticks (no grid), Hayashi–Yoshida has
the cross leading its synthetic in **84% of 69 windows** (sign test p = 7.6e-9), at a
sub-resolution horizon (median θ\* = −5 ms, below the ~150–260 ms inter-tick floor;
the peak is a flat plateau). The direction agrees with the VECM/GG result *without any
grid*, so it is not a synchronization artifact. Leg-level, leadership is not a total
order: the cross leads EUR/USD, but USD/JPY leads the cross. The hot loop is numba,
checked against a brute-force all-pairs oracle. A pybind11 C++ port was also built and
parity-tested to 1e-12, and the **honest benchmark** is reported: numba (12.4 ms)
beats it (18.0 ms) on one real day, so the library runs numba only.

**Sessions (05).** Conditioning on trading session is lookahead-immune (the clock
can't peek at the future). On the 1 s grid the cross's share is flat across
TOK / LDN / NY and the overlaps — GG_cross 0.63–0.68, all block-bootstrap CIs
overlapping and above 0.5 — and no hypothesis survives Holm (all Holm p = 1). There
is no Tokyo elevation. The earlier "LDN+NY error-corrects fastest" **speed** finding
was a time-support + pooling artifact — the Tokyo session had been measured on its
overnight hours — and is **withdrawn**. Scope: inside the 07–17 UTC liquid window,
"TOK" is a single winter hour (07–08 UTC, Dec–Mar) and "NY" the last 30–90 minutes
before 17:00, so this is evidence about the London trading day.

**Volatility regimes (06).** With a **filtered** 2-state HMM (a hand-written forward
filter, never `predict_proba`; EM fit on a train split; causal standardisation — the
lookahead discipline is *provable* via a truncation-invariance test), the share is
again stable (V1 n.s., Holm 0.34). A speed effect (V2 — stressed periods correct
faster, −1.85 s, Holm 0.012) rejects on the grid but shrinks to 0.57 s on fresh bars
only, and stressed periods are fresher, so it is not a clean structural effect. Two
states are imposed for interpretability (BIC prefers three). Volatility states are not
merely a relabelling of sessions.

**Robustness (07).** The staleness controls that withdrew the regime effects, applied
to the headline (1 s, k = 10, liquid hours):

| clock | GG_cross | IQR | windows > 0.5 |
|---|---|---|---|
| 1 s grid, all bars | 0.652 | 0.58–0.73 | 93% |
| 1 s grid, fresh bars only | 0.587 | 0.50–0.71 | 75% |
| refresh time | 0.570 | 0.53–0.62 | 86% |

Refresh time is lower on 126 of 151 paired days (p = 2e-17): EUR/JPY quotes update
about as often as both dollar legs combined, so on a last-observation-carried-forward
grid the stale synthetic "catches up" after the fact and the cross looks more like the
leader than it is. Re-running the session family in refresh time shows the cross's
share **rising through the day** (TOK 0.48 → NY 0.61): the two-sided H3 has
p = 0.005 (Holm 0.025 within that family, 0.05 across both clocks — borderline) and TOK sits *below* London and New York — the opposite of the
pre-registered Tokyo prior. Refresh time is post-hoc and changes the estimand, and the
fresh-bar grid shows no such gradient, so this is an unresolved, clock-dependent
pattern rather than a finding. No session pair is shown equivalent within ±0.05
(TOST, Holm). Widening the window to 07–20 UTC, which gives NY 3.5–4.5 hours, leaves
the grid results unchanged.

**Robustness discipline.** Every hypothesis is pre-registered and Holm-corrected
across the family; overlapping windows use a moving-block bootstrap over days, never
i.i.d.; a **refresh-time** control re-samples on genuine quote updates; and an
activity floor counts genuine quote updates per window (a bar count would admit a
no-trading Christmas Day as 36,000 forward-filled bars). Scheduled-news windows and a
GBP-triangle replication (where the "cross", EUR/USD, is the *deepest* instrument —
flipping the prior) are the remaining robustness cuts.

## Methods

- VECM with **imposed** β, equation-by-equation OLS, Newey–West HAC SEs; per-(day ×
  liquid-hours) windows, aggregated as median + IQR.
- Gonzalo–Granger component shares (point-identified) + Hasbrouck IS as Cholesky
  [lo, hi] bounds over {100ms, 250ms, 1s, 5s}.
- Three clocks: 1 s previous-tick grid (primary), fresh-bar filter, and multivariate
  refresh-time sampling (Barndorff-Nielsen–Hansen–Lunde–Shephard 2011).
- Hayashi–Yoshida covariation + lead-lag θ-scan on raw ticks (numba; a pybind11 C++
  port `fxpd_fast` is kept as an optional benchmark).
- Realized kernel (Parzen, BNHLS bandwidth) volatility proxy; filtered-only Gaussian
  HMM.
- Moving-block bootstrap (block = 5 days, B = 2000) + Holm multiplicity control;
  bootstrap TOST for equivalence claims.
- Spot value-date calendar (T+2 on each pair's holiday calendar) for the basis level.

Every estimator has a synthetic-data unit test with a known answer; the HY sign
convention is *defined* by one. `pytest -q` runs 75 network-free tests.

## Repo layout

```
src/fxpd/
  config.py              triangle identities, instruments, paths
  data/                  dukascopy, store, mids, synchronize, basis,
                         clean (QC), processed (per-day cache), refresh (staleness control)
  calendar/              sessions (DST-aware), news (announcement bands), value_dates (T+2)
  econo/                 vecm, discovery (GG + IS; grid or refresh clock), hy (numba)
  vol/realized.py        realized kernel
  regimes/hmm.py         filtered-only HMM
  inference/bootstrap.py moving-block bootstrap, Holm, TOST
  sanity/                ADF/KPSS + magnitude stats, Week-1 figure
cpp/                     optional pybind11 port of the HY loop (benchmark only)
notebooks/  01…07        one driver per stage (library = .py, drivers = notebooks)
tests/                   75 tests + fixtures/ (~1h real ticks, network-free CI)
```

## Setup & reproduce

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e . --no-deps
./.venv/bin/python -m ipykernel install --user --name fxpd --display-name "Python (fxpd)"
./.venv/bin/python -m pytest -q                       # 75 tests, network-free
./.venv/bin/python -m pip install -e ./cpp            # optional: C++ benchmark port
```

Open the notebooks with the *Python (fxpd)* kernel, or run one headless:

```bash
./.venv/bin/python -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=fxpd notebooks/05_sessions.ipynb
```

## Data & limitations

Source is the Dukascopy / SWFX **aggregator quote feed**: UTC-ms bid/ask + top-of-book
sizes, **no trades** (quote-based proxies only — no Lee-Ready, no true order flow), so
results are conditional on one aggregator's view of the market. `data/` (raw ticks +
processed panels, ~1.6 GB) is gitignored and reproducible from the resumable puller; a
1-hour tick sample lives in `tests/fixtures/` for network-free CI. Main estimations run
on weekdays in liquid hours (07–17 UTC) with the 5pm-NY rollover excluded; LOCF
staleness, the millisecond resolution floor (no sub-ms lead claims), DST-correct
session labels and T+2 value-date mismatches are handled throughout. The refresh-time
results are post-hoc robustness, not pre-registered tests.
