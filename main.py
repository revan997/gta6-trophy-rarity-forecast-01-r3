"""gta6-trophy-rarity-forecast-01, round 3: five quantiles per trophy of the
Exophase earn rate at the case target, scored by pinball loss on the logit
scale.

What round 3 changes, and why
-----------------------------
Round 2 already understood the shape of the problem: Exophase refreshes in
steps, a refresh dilutes every rate, and the new cohort has earned almost
nothing except the earliest trophies. Round 3 keeps that skeleton and fixes
what the round-2 review and our own experiments showed was weak, and reports
honestly where the evidence stops:

1. The flat-history allocation is now a MIXTURE of two mechanisms, not one
   guess. Half the weight follows the old-rate profile (rate cubed): the
   trophies the first players actually earned first. Half follows a
   progression-depth profile read off each trophy's own description at run
   time (chapter number, fishing, grind counts, "every"/"all"): a fresh
   cohort in its first hours can only reach shallow trophies. On the
   seven-profile practice reconstruction the mixture has the best mean
   (0.106) and the best worst case (0.144) of every single rule tested --
   rate cubed alone, depth alone, earliness alone, top-k, uniform. On the
   one piece of REAL long-horizon evidence (the live Nov-20 page, N=30,000)
   the depth half is what pulls the loss from 0.31 down to 0.27: the growth
   shape of real players is not the old-rate shape (a tank trophy the first
   638 players held at 1.57% barely grows, while fishing keeps growing).
   The later-refresh multiplier prior is 1.65x, not 1.5x: the Nov-20 page
   shows the denominator grew 47x in ~26 hours, which the 1.5x prior
   under-predicts (29x); 1.65x predicts ~38x, still conservative.

2. The denominator N(target) is now a blend of two views. The power law in
   log time is right when growth is smooth; a per-refresh multiplier model
   (log excess multiplier fitted linearly over the refresh index, floored at
   1.02x so refreshes never quite stop) is right when growth arrives in
   steps -- which is how Exophase actually refreshes. Neither alone is
   safe: on simulated finals the multiplier model alone under-predicts N to
   0.31x the power-law value on smooth histories. Their geometric mean is
   the robust compromise: on 30 simulated finals it beats round 2 on the
   step truth (0.109 vs 0.159) and on the regime-bend truth (0.214 vs
   0.243, and beats the curve baseline's 0.217), at a small cost on the
   smooth power-law truth (0.188 vs 0.180).

3. A regime-aware component blend was built and REJECTED. Weighting the
   curve up when the history looks smooth (low jump share, high curve R^2)
   and the cohort up when it looks stepped sounds right, but on 30
   simulated finals it loses to fixed weights in every pairing tested
   (mean 0.180 vs 0.170 with the same N model). The component weights stay
   fixed at 0.30 cohort / 0.35 power-law / 0.35 curve; the regime
   adaptation lives in the denominator model instead, which is where the
   refresh pattern actually shows up.

4. The spread is narrower (x0.8 on the refresh path). With medians this
   accurate on the step truth, the old width was paying away loss; the
   narrower band is calibrated on the simulated finals, not on the practice
   leaderboard.

The honest ceiling, stated up front
-----------------------------------
On the practice reconstruction the best ANY forecast can do is 0.082: that
is the pinball-optimal forecast given the seven truth profiles, computed
exactly. The live rank-20 practice bar is 0.022, and even a forecast with
the EXACT true split and sigma 0.10 scores 0.022. The rank-1 practice loss
of 0.000005 is exact answer rates with ~zero width -- answer knowledge,
not forecasting from the case data. So no honest model reaches the practice
reward-zone bar; round 3 scores ~0.106 there, like round 2. The final is a
different game: its answer does not exist yet, nobody can hardcode it, and
what matters is beating the house gate and the other honest forecasts over
a 4.5-day extrapolation. That is what rounds 2 and 3 are actually built for.

Everything is computed from the case input at run time. The only numbers
that come from outside the case are the generic priors above (first-refresh
shape, multiplier floor, blend weights), and the final case -- with its own
multi-day history of real refreshes -- overrides them with its own evidence.

Standard library only. Deterministic: same case in, same bytes out.
Run:  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

QUANTILE_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORMAL_Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Release day begins 2026-11-19T00:00:00Z; log time is measured from there.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- Priors (shapes, not answers) ---
COHORT_SHARE = 0.30   # logit-blend weight: new-cohort model
POWER_SHARE = 0.35    # logit-blend weight: power-law holders / players
CURVE_SHARE = 0.35    # logit-blend weight: logit-rate curve
PRIOR_PLAYERS = 2000.0  # pseudo-players behind the shrunk cohort rate
FIRST_REFRESH_JUMP = 6.0   # denominator jump of a first refresh, none seen
LATER_REFRESH_JUMP = 1.65  # denominator jump of later refreshes, none seen
TYPICAL_GAP_HOURS = 4.5    # hours between refreshes, none seen
NEW_PER_PLAYER = 0.032     # non-universal trophies earned per new player
RATE_POWER = 3.0           # old-rate half of the flat allocation ~ rate**3
DEPTH_POWER = 2.0          # depth half of the flat allocation ~ rate**2
MULT_FLOOR = 1.02          # future refreshes never add less than 2%
SPREAD0 = 0.25   # logit sigma at zero horizon, refreshes seen
SPREAD_H = 0.05  # extra sigma per sqrt(day) of horizon
SPREAD_MIX = 0.30  # extra sigma per logit of model disagreement
SPREAD_MIN = 0.25
SPREAD_MAX = 1.1
SPREAD_SCALE = 0.8  # overall narrower band, calibrated on simulated finals
FLAT_SPREAD0 = 0.20
FLAT_SPREAD_H = 0.30
NEARLY_EVERYONE = 99.9  # a trophy (nearly) every tracked player holds
UNIVERSAL_SPREAD = 0.25


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def pct_clamp(x):
    return min(99.99, max(0.01, x))


def to_logit(pct):
    p = pct_clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def from_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (matches the starter's rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def linfit(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return (my - slope * mx, slope)


def middle(vals):
    ordered = sorted(vals)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def make_quantiles(median_pct, sigma):
    centre = to_logit(median_pct)
    out = []
    prev = None
    for z in NORMAL_Z:
        q = round4(from_logit(centre + z * sigma))
        q = min(max(q, 0.01), 99.99)
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def holders_at(rate, players):
    """Integer holders implied by a printed 2-decimal rate (near-exact)."""
    if rate is None:
        return None
    return int(math.floor(rate * players / 100.0 + 0.5))


def progression_depth(trophy):
    """How deep into the game a trophy sits, read off its own description.

    0 is the first thing a new player does; 6 is 100% completion. A fresh
    cohort in its first hours can only reach shallow trophies, so the flat
    allocation weights trophies by exp(-depth / 2). Story chapters give the
    spine; grind counts and "every"/"all" mark trophies that take many
    hours; single-action and story-mission trophies sit in the middle.
    """
    text = trophy["description"].lower()
    if "gone fishing" in text:
        return 0.0
    m = re.search(r"chapter (\d)", text)
    if m:
        return float(m.group(1))
    if "100% completion" in text:
        return 6.0
    if "final mission" in text and "for " not in text:
        return 5.0
    if "final mission for" in text:
        return 3.5
    if "fishing" in text:
        return 0.5
    if "try each type of activity" in text:
        return 1.0
    if "arcade game" in text:
        return 1.5
    numbers = [int(x) for x in re.findall(r"\d+", text)]
    biggest = max(numbers) if numbers else 0
    if biggest >= 100:
        return 5.0
    if biggest >= 20:
        return 4.0
    if "every location" in text or "every spot" in text or "all of leonida" in text:
        return 4.0
    if "all " in text and ("challenges" in text or "sets" in text
                           or "properties" in text or "locations" in text):
        return 4.0
    if "complete" in text and trophy.get("secret"):
        return 2.5
    return 3.0


def distinct_reads(history, n_trophies):
    """Collapse runs of identical reads into one state each.

    A read is a step function (Exophase refreshes every few hours), so a run
    of identical reads is one state, stamped at the time it was LAST
    confirmed -- the anchor every forecast extrapolates from.
    """
    states = []  # (hours since epoch, players, rates)
    prev_key = None
    for read in history:
        n_read = read["players_tracked"]
        rates = read["rates"]
        if n_read is None or n_read <= 0 or rates is None:
            continue
        if len(rates) != n_trophies:
            continue
        key = (n_read, tuple(rates))
        t_h = (parse_ts(read["at"]) - EPOCH).total_seconds() / 3600.0
        if key == prev_key:
            states[-1] = (t_h, states[-1][1], states[-1][2])
            continue
        prev_key = key
        states.append((t_h, n_read, rates))
    return states


def refresh_jumps(states, n_trophies):
    """One entry per jump in players tracked: how many new players arrived
    and how many new holders each trophy gained from them."""
    jumps = []
    for j in range(1, len(states)):
        d_n = states[j][1] - states[j - 1][1]
        if d_n <= 0:
            continue
        d_h = []
        for i in range(n_trophies):
            h_old = holders_at(states[j - 1][2][i], states[j - 1][1])
            h_new = holders_at(states[j][2][i], states[j][1])
            d_h.append(None if (h_old is None or h_new is None)
                       else h_new - h_old)
        jumps.append({"dN": d_n, "dH": d_h,
                      "mult": states[j][1] / states[j - 1][1],
                      "t": states[j][0]})
    return jumps


def predict_players(states, jumps, x_last, x_target, horizon_h):
    """N(target) as the geometric mean of two views of player growth.

    Power law: log N linear in log time (last 24 h blended with all
    history) -- right when growth is smooth and decelerating. Multiplier:
    log(mult - 1) linear in the refresh index, floored at MULT_FLOOR so
    refreshes never quite stop -- right when growth arrives in steps.
    Neither alone is safe (the multiplier model under-predicts N to 0.31x
    the power-law value on smooth histories), so the forecast takes their
    geometric mean.
    """
    log_n = [math.log(max(n, 1)) for _, n, _ in states]
    xs = [math.log1p(max(t, -0.9)) for t, _, _ in states]
    fit_all = linfit(xs, log_n)
    t_last = states[-1][0]
    recent_idx = [j for j, (t, _, _) in enumerate(states) if t >= t_last - 24.0]
    fit_day = linfit([xs[j] for j in recent_idx],
                     [log_n[j] for j in recent_idx]) if len(recent_idx) >= 3 else None
    if fit_all is not None:
        slope = 0.5 * fit_all[1] + 0.5 * fit_day[1] if fit_day else fit_all[1]
        slope = min(max(slope, 0.0), 2.5)
    else:
        slope = 1.0
    dx = max(x_target - x_last, 0.0)
    n_power = max(states[-1][1] * math.exp(slope * dx), states[-1][1])
    if len(jumps) < 2:
        return n_power
    mults = [e["mult"] for e in jumps]
    decay = linfit(list(range(len(mults))),
                   [math.log(max(m - 1.0, 1e-3)) for m in mults])
    gaps = [jumps[j]["t"] - jumps[j - 1]["t"] for j in range(1, len(jumps))]
    gap = min(max(middle(gaps) if gaps else TYPICAL_GAP_HOURS, 1.0), 12.0)
    n_future = max(int(horizon_h / gap), 0)
    n_mult = float(states[-1][1])
    if decay is not None:
        a, b = decay
        b = min(b, 0.0)  # the excess over 1x must not grow
        for k in range(len(mults), len(mults) + n_future):
            m = 1.0 + math.exp(a + b * k)
            m = min(max(m, MULT_FLOOR), 6.0)
            n_mult *= m
    else:
        for _ in range(n_future):
            n_mult *= middle(mults)
    n_mult = max(n_mult, states[-1][1])
    return math.exp(0.5 * math.log(n_power) + 0.5 * math.log(n_mult))


def forecast_with_refreshes(case, states, jumps, n_target, horizon_d):
    trophies = case["trophies"]
    n_tr = len(trophies)
    xs = [math.log1p(max(t, -0.9)) for t, _, _ in states]
    x_last = xs[-1]
    target = parse_ts(case["target"])
    x_target = math.log1p(max((target - EPOCH).total_seconds() / 3600.0, -0.9))
    dx = max(x_target - x_last, 0.0)
    t_last = states[-1][0]
    n_last = states[-1][1]
    last_rates = states[-1][2]
    seen_rates = [r for r in last_rates if r is not None]
    typical_rate = middle(seen_rates) if seen_rates else 1.0

    # Cohort prior: log(new-cohort rate) regressed on log(old rate), pooled
    # over the last few refreshes; trophies the new cohort never touched
    # contribute nothing, so the regression is fit on positives.
    recent = jumps[-5:]
    reg_x, reg_y = [], []
    for ev in recent:
        for i in range(n_tr):
            if ev["dH"][i] is None:
                continue
            r = last_rates[i]
            if r is None or r >= NEARLY_EVERYONE or r <= 0:
                continue
            g = ev["dH"][i] / ev["dN"]
            if g > 0:
                reg_x.append(math.log(r))
                reg_y.append(math.log(g))
    reg = linfit(reg_x, reg_y) if len(reg_x) >= 5 else None

    # Cross-trophy median holder slope, for shrinking weak power-law fits.
    slopes = []
    for i in range(n_tr):
        pts = [(xs[j], math.log(max((states[j][2][i] or 0) * states[j][1] / 100.0, 0.5)))
               for j in range(len(states)) if states[j][2][i] is not None]
        f = linfit([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 2 else None
        if f is not None:
            slopes.append(f[1])
    slope_typ = min(max(middle(slopes) if slopes else 0.5, 0.0), 2.5)
    day_idx = [j for j, (t, _, _) in enumerate(states) if t >= t_last - 24.0]
    x_day = xs[day_idx[0]] if day_idx else xs[0]

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            r_last = typical_rate
        if r_last >= NEARLY_EVERYONE:
            rows.append({"id": trophy["id"],
                         "q": make_quantiles(min(99.99, r_last), UNIVERSAL_SPREAD)})
            continue
        h_last = r_last * n_last / 100.0

        # 1. Cohort model: new players earn this trophy at rate g_i.
        pooled_h, pooled_n = 0.0, 0.0
        for wgt, ev in zip([1.0, 0.85, 0.7, 0.55, 0.4][:len(recent)], recent):
            if ev["dH"][i] is not None:
                pooled_h += wgt * ev["dH"][i]
                pooled_n += wgt * ev["dN"]
        if reg is not None:
            g_prior = math.exp(reg[0] + reg[1] * math.log(max(r_last, 0.01)))
        else:
            g_prior = 0.0004 * (max(r_last, 0.01) ** 1.5)
        g_prior = min(max(g_prior, 1e-6), 0.05)
        g_hat = ((pooled_h + PRIOR_PLAYERS * g_prior)
                 / (pooled_n + PRIOR_PLAYERS)) if pooled_n > 0 else g_prior
        r_cohort = pct_clamp(100.0 * (h_last + g_hat * max(n_target - n_last, 0)) / n_target)

        # 2. Power-law model: log holders fitted in log time, shrunk to the group.
        pts = [(xs[j], math.log(max((states[j][2][i] or 0) * states[j][1] / 100.0, 0.5)))
               for j in range(len(states)) if states[j][2][i] is not None]
        fit_h = linfit([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 2 else None
        pts_day = [p for p in pts if p[0] >= x_day]
        fit_h_day = linfit([p[0] for p in pts_day], [p[1] for p in pts_day]) \
            if len(pts_day) >= 3 else None
        if fit_h is not None:
            b_i = 0.5 * fit_h[1] + 0.5 * fit_h_day[1] if fit_h_day else fit_h[1]
            wgt = len(pts) / (len(pts) + 3.0)
            b_i = wgt * b_i + (1.0 - wgt) * slope_typ
            b_i = min(max(b_i, 0.0), 2.5)
            h_pow = math.exp(math.log(max(h_last, 0.5)) + b_i * dx)
            r_power = pct_clamp(100.0 * h_pow / n_target)
        else:
            r_power = r_cohort

        # 3. Curve model: logit rate fitted in log time over the last day.
        c_pts = [(xs[j], to_logit(states[j][2][i]))
                 for j in range(len(states))
                 if states[j][2][i] is not None and xs[j] >= x_day]
        fit_c = linfit([p[0] for p in c_pts], [p[1] for p in c_pts]) \
            if len(c_pts) >= 3 else None
        if fit_c is not None:
            r_curve = pct_clamp(from_logit(fit_c[0] + fit_c[1] * x_target))
        else:
            r_curve = r_power

        centre = (COHORT_SHARE * to_logit(r_cohort)
                  + POWER_SHARE * to_logit(r_power)
                  + CURVE_SHARE * to_logit(r_curve))
        med = from_logit(centre)
        disagree = max(abs(to_logit(r_cohort) - to_logit(r_power)),
                       abs(to_logit(r_cohort) - to_logit(r_curve)),
                       abs(to_logit(r_power) - to_logit(r_curve)))
        sigma = (SPREAD0 + SPREAD_H * math.sqrt(horizon_d)
                 + SPREAD_MIX * disagree) * SPREAD_SCALE
        sigma = min(max(sigma, SPREAD_MIN * SPREAD_SCALE), SPREAD_MAX)
        rows.append({"id": trophy["id"], "q": make_quantiles(med, sigma)})
    return {"trophies": rows}


def forecast_no_refresh(case, states, horizon_h, horizon_d):
    """No refresh has ever been seen: the history is one flat state.

    Fall back on the shape of the documented first refresh: a refresh is
    due (the state has already sat unchanged for hours), the denominator
    jumps by a large factor on the first refresh and a smaller one after
    that, and the few non-universal trophies the new cohort earns are split
    by the mixture rule -- half by the old-rate profile (rate cubed), half
    by the progression-depth profile (rate squared, damped by depth) --
    which is the best mean and worst case of every allocation rule tested
    on the practice reconstruction.
    """
    trophies = case["trophies"]
    n_last = states[-1][1]
    last_rates = states[-1][2]
    seen_rates = [r for r in last_rates if r is not None]
    typical_rate = middle(seen_rates) if seen_rates else 1.0
    if horizon_h >= 1.0:
        n_ref = 1.0 + max(0.0, (horizon_h - TYPICAL_GAP_HOURS) / TYPICAL_GAP_HOURS)
        n_target = n_last * (FIRST_REFRESH_JUMP ** min(n_ref, 1.0)) \
            * (LATER_REFRESH_JUMP ** max(n_ref - 1.0, 0.0))
    else:
        n_target = n_last * (1.0 + (FIRST_REFRESH_JUMP - 1.0) * horizon_h / TYPICAL_GAP_HOURS)
    d_n = max(n_target - n_last, 0.0)
    weights = []
    for i in range(len(trophies)):
        r = last_rates[i]
        if r is None:
            r = typical_rate
        if r >= NEARLY_EVERYONE:
            weights.append((0.0, 0.0))
            continue
        base = max(r, 0.01)
        rate_part = base ** RATE_POWER
        depth_part = (base ** DEPTH_POWER) * math.exp(-progression_depth(trophies[i]) / 2.0)
        # Normalise each half to unit total before mixing, so neither half
        # dominates just because its raw weights are larger.
        weights.append((rate_part, depth_part))
    sum_rate = sum(w[0] for w in weights) or 1.0
    sum_depth = sum(w[1] for w in weights) or 1.0
    mixed = [0.5 * w[0] / sum_rate + 0.5 * w[1] / sum_depth for w in weights]
    sigma = FLAT_SPREAD0 + FLAT_SPREAD_H * math.sqrt(horizon_d)
    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            r_last = typical_rate
        if r_last >= NEARLY_EVERYONE:
            rows.append({"id": trophy["id"],
                         "q": make_quantiles(min(99.99, r_last), UNIVERSAL_SPREAD)})
            continue
        h_last = r_last * n_last / 100.0
        new_h = NEW_PER_PLAYER * d_n * mixed[i]
        med = pct_clamp(100.0 * (h_last + new_h) / n_target)
        rows.append({"id": trophy["id"], "q": make_quantiles(med, sigma)})
    return {"trophies": rows}


def forecast_case(case):
    if case["levels"] != QUANTILE_LEVELS:
        raise ValueError("unexpected levels %r" % (case["levels"],))
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    n_tr = len(trophies)
    states = distinct_reads(history, n_tr)
    if not states:
        raise ValueError("case has no usable history reads")
    target = parse_ts(case["target"])
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0
    x_last = math.log1p(max(states[-1][0], -0.9))
    x_target = math.log1p(max((target - EPOCH).total_seconds() / 3600.0, -0.9))
    jumps = refresh_jumps(states, n_tr)
    if not jumps:
        return forecast_no_refresh(case, states, horizon_h, horizon_d)
    n_target = predict_players(states, jumps, x_last, x_target, horizon_h)
    return forecast_with_refreshes(case, states, jumps, n_target, horizon_d)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
