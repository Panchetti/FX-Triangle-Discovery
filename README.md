# Regime-Dependent Price Discovery in the FX Majors Triangle

Which leg of the EUR/USD–USD/JPY–EUR/JPY no-arbitrage triangle incorporates new
information first, and does that leadership **migrate** across trading sessions and
volatility regimes in a way you could identify in real time?

**Finding.** The quoted cross (EUR/JPY) leads its synthetic replica
(EUR/USD + USD/JPY) — it carries ≈65% of the permanent component — and that
leadership is **stable**: it does not migrate across sessions or volatility states.
The apparent regime effects (a Tokyo share bump, a "deep-window" speed advantage)
turned out to be liquidity/staleness artifacts and were withdrawn under the controls
built for exactly that. So the falsifiable claim — real-time-identifiable migration —
is **not** supported; the headline is a clean efficiency result, and the
methodological contribution is the staleness / refresh-time controls that stopped
three false positives.

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

## What I found

**Basis (01).** `z` is near-I(0) and tiny: std 0.32 bp full-sample → ~0.06 bp in
liquid hours. The ±8 bp tails are overnight LOCF staleness (the dollar legs tick in
only ~30% of 1s bins), not arbitrage. This justifies imposing β rather than testing
for it.

**Who error-corrects (02).** Every leg error-corrects with the expected signs — when
the cross is rich (z > 0) it falls (α ≈ −0.079) and the dollar legs rise. The basis
closes with a **half-life ≈ 2.7 s**, the triangle's arbitrage speed. A Gonzalo–Granger
preview puts ≈ 64% of the permanent component on the quoted cross — mildly against the
"deep dollar legs lead" prior.

**Static discovery (03).** Over 7 months / 152 windows, **GG_cross ≈ 0.65**
[IQR 0.58, 0.72], stable across frequency (0.60 @100ms → 0.67 @5s). Hasbrouck IS is
reported only as Cholesky **bounds**, and they are wide and frequency-dependent —
IS_cross [0.28, 0.82] @100ms blowing out to [0.03, 0.99] @5s — because cross–synth
residual correlation rises 0.53 → 0.97 as bars coarsen (the **Epps effect**, shown not
hidden). GG is the primary, point-identified number; IS is informative only at fine
grids.

**Lead-lag in real time (04).** On the raw async ticks (no grid), Hayashi–Yoshida has
the cross leading its synthetic in **84% of 69 windows** (sign test p = 7.6e-9), at a
sub-resolution horizon (median θ\* = −5 ms, below the ~150–260 ms inter-tick floor;
the peak is a flat plateau). The direction agrees with the VECM/GG result *without any
grid*, so it is not a synchronization artifact. Leg-level, leadership is not a total
order: the cross leads EUR/USD, but USD/JPY leads the cross. The hot loop is a pybind11
C++ module, parity-tested to 1e-12 — and the **honest benchmark** is reported: Numba
(12 ms) actually beat the C++ extension (19 ms) on one real day.

**Sessions — the headline (05).** Conditioning on trading session is lookahead-immune
(the clock can't peek at the future). The cross's leadership **share is stable** across
TOK / LDN / NY and the overlaps: GG_cross ~0.63–0.69, all block-bootstrap CIs
overlapping and well above 0.5. No migration hypothesis survives Holm (H1 0.89,
H2 0.28, H3 0.89). The small Tokyo elevation was staleness (TOK synth fresh only 19% of
1s bars; it converges under fresh-only filtering). The earlier "LDN+NY error-corrects
fastest" **speed** finding was traced to a time-support + pooling artifact — the Tokyo
session had been measured on its overnight hours — and is **withdrawn**.

**Volatility regimes (06).** With a **filtered** 2-state HMM (a hand-written forward
filter, never `predict_proba`; EM fit on a train split; causal standardisation — the
lookahead discipline is *provable* via a truncation-invariance test), the share is
again stable (V1 n.s.). A speed effect (V2 — stressed periods correct faster, −1.85 s)
rejects on the 1s grid but is **estimator-sensitive and partly a freshness confound**
(stressed periods are ~2× fresher), so it is not a clean structural effect. Volatility
states are not merely a relabelling of sessions.

**Robustness discipline.** Every hypothesis is pre-registered and Holm-corrected across
the family; overlapping windows use a moving-block bootstrap over days, never i.i.d.; a
**refresh-time** control re-samples on genuine quote updates and is what withdrew the
directional speed findings. Scheduled-news windows and a GBP-triangle replication
(where the "cross", EUR/USD, is the *deepest* instrument — flipping the prior) are the
remaining robustness cuts.

## Methods

- VECM with **imposed** β, equation-by-equation OLS, Newey–West HAC SEs; per-(day ×
  liquid-hours) windows, aggregated as median + IQR.
- Gonzalo–Granger component shares (point-identified) + Hasbrouck IS as Cholesky
  [lo, hi] bounds over {100ms, 250ms, 1s, 5s}.
- Hayashi–Yoshida covariation + lead-lag θ-scan on raw ticks (numpy → numba → C++
  `fxpd_fast`, graceful fallback).
- Realized kernel (Parzen, BNHLS bandwidth) volatility proxy; filtered-only Gaussian
  HMM.
- Moving-block bootstrap (block = 5 days, B = 2000) + Holm multiplicity control.

Every estimator has a synthetic-data unit test with a known answer; the HY sign
convention is *defined* by one. `pytest -q` runs 61 network-free tests.

## Repo layout

```
src/fxpd/
  config.py              triangle identities, instruments, paths
  data/                  dukascopy, store, mids, synchronize, basis,
                         clean (QC), processed (per-day cache), refresh (staleness control)
  calendar/              sessions (DST-aware), news (announcement bands)
  econo/                 vecm, discovery (GG + IS), hy (+ _hy_numpy / _hy_numba)
  vol/realized.py        realized kernel
  regimes/hmm.py         filtered-only HMM
  inference/bootstrap.py moving-block bootstrap + Holm
  sanity/                ADF/KPSS + magnitude stats, Week-1 figure
cpp/                     pybind11 `fxpd_fast` (optional): HY hot loop
notebooks/  01…06        one driver per stage (library = .py, drivers = notebooks)
tests/                   61 tests + fixtures/ (~1h real ticks, network-free CI)
```

## Setup & reproduce

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install -e . --no-deps
./.venv/bin/python -m ipykernel install --user --name fxpd --display-name "Python (fxpd)"
./.venv/bin/python -m pytest -q                       # 61 tests, network-free
./.venv/bin/python -m pip install -e ./cpp            # optional C++ backend
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
in liquid hours (07–17 UTC) with the 5pm-NY rollover excluded; LOCF staleness, the
millisecond resolution floor (no sub-ms lead claims), and DST-correct session labels
are handled throughout.
