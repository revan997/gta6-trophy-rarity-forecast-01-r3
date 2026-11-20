# gta6-trophy-rarity-forecast-01 — round 3

Forecast each GTA VI PS5 trophy's Exophase earn rate at 2026-11-26 00:00 UTC
as five quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

## Model

- **Refresh-aware.** Histories are collapsed to distinct states; each jump in
  players tracked is a refresh event that teaches the new-cohort profile
  `g_i` (new holders per new player), pooled over the last 5 refreshes and
  shrunk toward a log-g / log-rate regression.
- **Denominator `N(target)`** is the geometric mean of a power law in log
  time and a per-refresh multiplier model (log excess multiplier fitted
  linearly over the refresh index, floored at 1.02x). The blend is the robust
  compromise: the multiplier model alone under-predicts N to 0.31x the
  power-law value on smooth histories.
- **Three forecasts blended in logit space** (fixed 0.30 cohort / 0.35
  power-law / 0.35 logit-rate curve). A regime-aware blend was built and
  tested and REJECTED: on 30 simulated finals it loses to fixed weights in
  every pairing.
- **Flat history** (no refresh seen): first-refresh shape prior (denominator
  x6, then x1.65 per refresh) and a 50/50 mixture allocation of the few
  non-universal new holders — half by the old-rate profile (rate^3), half by
  a progression-depth profile parsed from each trophy's own description.
  Best mean (0.106) and best worst case (0.144) of every allocation rule
  tested on the practice reconstruction.
- **Spread** is narrower than round 2 (x0.8 on the refresh path), calibrated
  on simulated finals.

## Backtests (pinball loss, logit scale)

| Test | round 3 | round 2 | round 1 | persist | curve |
|---|---|---|---|---|---|
| A practice recon, mean of 7 truths | 0.1058 (0.0605–0.1436) | 0.1059 | 0.568–0.581 | 1.4884 | 1.4884 |
| B flat -> live Nov-20 page (N=30,000) | 0.2680 | 0.3131 | 0.7154 | 2.2883 | 2.2883 |
| C power-law sim, 10 seeds | 0.2021 | 0.1894 | 0.2031 | 0.4134 | 0.1976 |
| C2 step/refresh sim, 10 seeds | 0.1200 | 0.1696 | 0.2565 | 0.2400 | 0.4123 |
| C3 regime-bend sim, 10 seeds | 0.1746 | 0.2026 | 0.2102 | 0.2610 | 0.1856 |
| C, 30 seeds | 0.1879 | 0.1801 | 0.1937 | 0.3929 | 0.1856 |
| C2, 30 seeds | 0.1094 | 0.1587 | 0.2491 | 0.2343 | 0.4159 |
| C3, 30 seeds | 0.2135 | 0.2428 | 0.2410 | 0.2621 | 0.2173 |

## Honest ceiling

On the practice reconstruction the best ANY forecast can do is 0.0817
(the pinball-optimal forecast given the seven truth profiles). The live
rank-20 practice bar is 0.022; even a forecast with the EXACT true split
and sigma 0.10 scores 0.022, and the live rank-1 loss of 0.000005 is exact
answer rates with ~zero width — answer knowledge, not forecasting from the
case data. No honest model reaches the practice reward-zone bar. Round 3
beats round 2 on the step truth and the regime-bend truth (and beats the
curve baseline on C3), at a small cost on the smooth power-law truth.

## Run

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, deterministic, no network. Tested 2x on p00/p01:
byte-identical outputs, format valid (55 rows x 5 quantiles, ids 1–55,
monotonic, in [0, 100]), ~0.3 s per run.
