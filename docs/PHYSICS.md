# Physics of orbital decay and re-entry — notes behind the Phase 3c model

These notes record what the decay model has to get right, why the earlier forecasters
(SGP4 extrapolation, shell-level GNN) were the wrong tool for re-entry, and what to expect
before running anything. Equations are standard (Vallado, *Fundamentals of Astrodynamics*;
King-Hele, *Satellite Orbits in an Atmosphere*); papers are cited where a number is quoted.

## 1. Drag

The only non-conservative force that matters below ~800 km is aerodynamic drag:

    a_D = -½ · ρ · (C_D A / m) · |v_rel| · v_rel

- ρ: neutral mass density (kg/m³), the dominant uncertainty.
- B = C_D A/m: ballistic coefficient (m²/kg). For debris and rocket bodies it's unknown and
  changes with attitude (tumbling).
- v_rel: velocity relative to the co-rotating atmosphere. Co-rotation lowers the drag by a factor
  F = (1 − r_p ω_E cos i / v_p)² ≈ 0.9–1.1. Winds mostly cancel over an orbit.

Gauss's equations, with drag acting only along the track (T = −½ ρ B v²), give

    da/dt = 2 a² v T / μ          → da/dt = −ρ B a² v³ / μ
    de/dt = 2 (e + cos ν) T / v   → de/dt = −ρ B v (e + cos ν)

Averaged over one orbit, using the weight dM = (1 − e cos E) dE, these become secular rates.
For a circular orbit: **da/dt = −ρ B √(μ a)**, about 1–2 km/day at 300 km near solar maximum
for typical debris. Drag acts at perigee, so an eccentric orbit loses apogee first and
circularises, then decays quickly. Lifetime includes the time to circularise (King-Hele).

## 2. The atmosphere: why space weather is the whole game

Density is roughly exponential, ρ ≈ ρ₀ exp(−(h − h₀)/H). The scale height H ≈ kT/(m g)
grows with exospheric temperature T∞, which is set by:

- **Solar EUV**, measured by the F10.7 proxy (81-day mean plus the daily value). Between solar
  minimum and maximum, density at 400–500 km changes by **at least ×10** (Knowles, TAMU
  workshop 2025). This is the slow, dominant driver: an 11-year cycle plus 27-day rotation.
- **Geomagnetic heating**, measured by Ap/Kp. Storms add Joule heating at high latitudes and can
  double density within hours. In May 2024 (Gannon storm) satellites lost hundreds of metres of
  semi-major axis per day, and 12 Starlinks were lost in the pre-conditioned atmosphere
  (arXiv 2410.16254). Post-storm cooling (NO emission) is fast, and models handle it poorly.

Model accuracy: NRLMSIS 2.0 tends to *over*-predict density by about 28–44 % in storms
(JB2008 by about 13–30 %), and quiet-time errors are about 10–20 %. A fitted ballistic
coefficient absorbs any *constant* model bias. It cannot absorb errors that change over time.

## 3. Why SGP4 + B\* is the wrong long-horizon physics

SGP4's density is a **fixed** power-law profile that never sees F10.7 or Ap. All solar
activity enters through B\*, a scalar fitted over the few days of the orbit-determination
window. B\* therefore stands for "density × ballistic coefficient as they were that week".
Extrapolating it six months freezes the atmosphere at today's state. B\* is also noisy, often
negative, and absorbs force-model errors. This is exactly the Phase 3b failure: SGP4 decay got
worse with horizon and lost to persistence at 26 weeks.

## 4. Doing it properly: ballistic coefficient from mean-motion history

The standard TLE-only re-entry method (Saunders et al.; Sang et al.; refined by Gondelach,
Armellin and Lidtke 2017, *Math. Probl. Eng.* 7309637) works like this:

1. a_TLE = (μ / n²)^{1/3} from the mean motion of two TLEs, which is the most accurate element in a
   TLE. Δa_TLE = a₂ − a₁.
2. Integrate the decay model between the two epochs with the *observed* density (known in the
   past) to get Δa_prop(B). In the orbit-averaged model Δa_prop is nearly linear in B, so the
   secant iteration reduces to B ≈ B₀ · Δa_TLE / Δa_prop(B₀), refined once.
3. Use that B for the forecast, starting from the *same* state used in the fit: errors in the
   state are then absorbed consistently.
4. Filter: negative B\*, mean-motion outliers, perigee outliers.

That paper reports 62–72 % of rocket-body predictions within 10 % of remaining lifetime.
The usual operational target is ±20 % (Salmaso, Trisolini and Colombo, *Aerospace* 2023).

## 5. Where the error comes from, and where ML can help

| Error source | Size | Who can fix it |
|---|---|---|
| Future solar flux / Ap | dominant beyond weeks; F10.7 forecast RMS 5–27 sfu at 1–24 months | space-weather *forecast*, not past indices |
| Density-model bias | 10–40 % | partly absorbed by fitted B |
| B changing over time (tumbling, C_D) | object-dependent | ML on object history |
| TLE noise, stale epochs | large for small debris | filtering, longer fit baselines |

A secondary report of Shambaugh's (unverified, arXiv) puts the median one-year re-entry error
at 44.9 days with forecast inputs vs 3.9 days with the true future inputs, and attributes about
89 % of forecast variance to solar flux. For debris, Podladchikova et al. (2021) find most
long-term predictions are off by more than a year even with observed flux.

**Consequences for this project:**

- Feeding *past* F10.7/Ap into a learned model can't add much: the fitted B already
  encodes the current density, and the future, which matters, isn't in the inputs. This
  explains the null space-weather ablations in Phases 3 and 3b. It isn't a bug.
- The clean, publishable experiment measures the **value of space-weather information**:
  run the same physics with (a) frozen current conditions, (b) 81-day persistence,
  (c) the true future indices (oracle). The gap between (b) and (c) is the most any
  space-weather forecast could improve re-entry prediction, broken down by altitude and
  solar phase.
- The ML that should work is a **residual corrector on the physics lifetime**: it learns the
  systematic bias in log(lifetime) as a function of perigee, eccentricity, B history, B\* trend,
  object type and solar phase. It shouldn't try to learn the physics.
- Expected failure modes: objects with very few or stale TLEs (small debris); high-eccentricity
  objects where lunisolar perturbations move the perigee (GTO, which this model ignores);
  manoeuvring payloads (excluded, since the cohort is debris and rocket bodies only).

## 6. Model choices for Phase 3c

- **State**: mean a, e and inclination from TLEs. Perigee altitude uses a spherical Earth
  (oblateness adds up to ±10 km of perigee error at high latitude; ignored to first order).
- **Dynamics**: orbit-averaged Gauss equations over 32 eccentric-anomaly nodes, with
  atmospheric co-rotation. Steps adapt so each step changes perigee by less than about
  H/4. Re-entry is declared when perigee altitude < 120 km for a near-circular orbit
  (the remaining lifetime is then under a day).
- **Density**: NRLMSIS 2.1 via pymsis (its default version), averaged over latitude and local time, tabulated
  daily in log ρ against altitude (80–1000 km) for the observed F10.7, F10.7a and Ap.
  It is interpolated linearly in log ρ.
- **Ballistic coefficient**: Saunders-style fit over the previous 28–56 days of weekly TLEs,
  with observed density.
- **Truth**: satcat decay dates. They are day-precision, so errors under about 1 day are
  meaningless.

Sources: Gondelach et al. 2017 (doi:10.1155/2017/7309637); Salmaso et al. 2023,
*Aerospace* 10(3):297; Geul, Mooij and Noomen (ESA SDC) on density uncertainty dominating
decay-time uncertainty; arXiv 2410.16254 (May 2024 Starlink losses); arXiv 2605.19850
(SGP4 vs high-fidelity on Starlink TLEs); Podladchikova et al. 2021 (F10.7 prediction and
re-entry); MSIS 2.0 vs JB2008 storm-time comparisons (Chinese Space Sci. Tech. 2025, 45(1)).

## 7. What the 10-year test showed

326k forecasts from 135 issue dates (2016–2026), with test issues from 2023 on (full table in PLAN §7):

- Expectations confirmed: SGP4 extrapolation fails (146 % median error at one year). Fitted-B
  physics with 81-day persistence reaches 20 %, and 14 % for rocket bodies. With the true future
  space weather it reaches 13 %.
- B\* used directly as a ballistic coefficient fails even with perfect space weather (151 %). The
  fit in §4 is not optional.
- The persistence→oracle gap peaks when the solar cycle turns fast (2021: 61 % vs 10 %), as
  expected from §2.
- A learned residual overfits to the solar phase it was trained on and does not generalise. This
  matches the warning in §5: without the future in its inputs, ML cannot recover the space-weather
  error.

## 8. Version 2: fixing what the first test exposed

Breaking the version 1 errors down by source showed where the trouble was:

| Source | Evidence | Fix |
|---|---|---|
| Missing Sun and Moon pull | e > 0.2 (36 % of objects, mostly GTO stages): 45 % error even with true future space weather; 32 % of real re-entries predicted to never come down | J2 + lunisolar secular dynamics (8.1) |
| Future solar activity | Near-circular orbits: 10 % error with true future space weather, 21 % with persistence | Solar-cycle analog forecast (8.2) |
| Object (B drift, fit noise) | Error with true space weather is unbiased in every year, spread ~0.11 in log lifetime | Object-only learned residual, calibrated windows (8.3) |

### 8.1 Sun, Moon and J2 (what moves a GTO perigee)

Third-body gravity does not change the orbit-averaged *a*, but it exchanges angular momentum
with the orbit. For a GTO stage (apogee ~36,000 km), the Sun and Moon move perigee height by
tens of km over weeks. Since density at 150–250 km changes by e-folds over 30–40 km, this decides
whether the stage re-enters in weeks or survives for years.

We use the vector (Milankovitch) form, with the eccentricity vector **e** and **j** = √(1−e²)ĥ.
The third-body quadrupole potential averaged over one satellite orbit is
Φ = −(G m₃ a²/4R₃³)[1 − 6e² − 3(**j**·d̂)² + 15(**e**·d̂)²]. This gives
(Tremaine, Touma and Kazandjian 2009; Rosengren and Scheeres 2014):

    dj/dt = A [5(e·d) e×d − (j·d) j×d]
    de/dt = A [5(e·d) j×d − 2 j×e − (j·d) e×d],     A = 3 G m₃ / (2 R₃³ n)

The Sun and Moon positions come from JPL DE440s at 6-hour steps. J2 enters through its averaged
potential and gives the textbook nodal regression and apsidal precession, which tests check.
J2 matters here because it turns ω and Ω, which sets the geometry the Sun and Moon act on.
Checks:
- unit tests reproduce the J2 rates and the ring-averaged lunisolar nodal rate −¾(n₃²/n)cos i;
- on 2,900 real e > 0.5 orbits over about 3 months, the correlation between predicted and
  observed perigee change rises from −0.06 to 0.78, and the median perigee error falls from
  20 to 13 km.

The remaining error comes from octupole terms, TLE noise and drag-fit noise.

### 8.2 Solar-cycle analog forecast

Holding today's flux (persistence) is correct on average but blind to the cycle's shape. The
11-year cycle repeats well enough that earlier cycles, read forward from the same phase (days
since minimum), forecast the next 1–2 years better. At each issue date:

1. Locate the current cycle minimum, using only data available that day: the minimum of the
   13-month centred mean, complete 7 months before.
2. Read each earlier cycle since 1957 forward from that phase.
3. Rescale each one so its 81-day flux matches today's; the rescaling fades over 2 years.

The median of these past cycles is the forecast. Backtest from 1990 to 2025, error in the 81-day
F10.7 (sfu):

| Horizon | 3 months | 6 months | 1 year | 2 years |
|---|---|---|---|---|
| Persistence | 16.4 | 22.2 | 29.8 | 45.8 |
| Analog | 16.2 | 20.2 | 24.5 | 33.3 |

Weighting the past cycles by how well they match the last year does not help. The ensemble's
spread is too narrow (it covers the truth 46 % of the time), so windows are calibrated
empirically instead (8.3).

### 8.3 What ML is for

With the true future space weather, the leftover error comes from the object: B changes with
attitude and the fit is noisy. That error has no solar-phase signature, so it can be learned on
one part of the cycle and applied to another. We train it only on object features, on
log(actual / oracle), and apply it on top of the analog forecast. Quantile regression gives
80 % windows, scored by whether they contain the truth 80 % of the time on the test years.
Operational centres issue re-entry *windows* rather than dates for the same reason.

### 8.4 Version 2 results (10 years, 135 issue dates, 326k forecasts)

Test = issues from 2023 on (49 dates). "All years" = 2016–2026. The 1-year horizon:

| Median timing error, 1 year | Test | All years | Count error per date (all years) |
|---|---|---|---|
| SGP4, frozen B\* | 146 % | 200 % | 124 |
| Physics, 81-day persistence | 19 % | 22 % | 61 |
| **Physics, solar-cycle analog** | **20 %** | **18 %** | **38** |
| Physics, analog + object residual | 26 % | 18 % | 40 |
| Physics, true weather, no Sun/Moon | 13 % | 12 % | 20 |
| Physics, true weather (upper bound) | 13 % | 12 % | 19 |

What held up:
- **Sun and Moon fix the high-eccentricity objects.** For e ≥ 0.2 on the test years, the error
  with true weather falls from 34 % to 16 % (v1: 45 %), and F1 rises from 0.69 to 0.85. At
  e < 0.2 they change nothing, as expected.
- **The analog forecast removes persistence's worst years.** Persistence fails when the cycle
  turns: 33 % in 2020 and 57 % in 2021. The analog stays at 15–17 % in those years. Over the
  decade it lowers the median from 22 % to 18 % and the count error from 61 to 38 per date.
  The worst year goes from 57 % to 31 %.

What did not:
- **On the test years the analog only ties persistence** (20 % vs 19 %). In 2025, after the
  Cycle 25 maximum, it ran ~10 % high on flux, and objects lived longer than forecast (bias +0.14
  in log lifetime). In 2024, the run with the true weather was itself 11 % late. That is an
  NRLMSIS density bias near solar maximum, not a forecast error.
- **The object residual is a second negative result** (26 % vs 20 % on test). The error with true
  weather is already nearly unbiased each year (|median| ≤ 0.11), so there is little to learn,
  and the 2022 data that dominates training does not transfer.
- **Quantile-regression windows covered only 40 %.** They were replaced (below).

**Calibrated 80 % windows.** At each issue date, the window is the empirical quantiles of
log(actual / analog), taken from forecasts whose re-entry had already happened by that date. The
quantiles are grouped by predicted lead time and perigee band. The level is adapted online to
the recent miss rate (adaptive conformal inference; Gibbs and Candès 2021).

| Coverage of the 80 % window | 30 d | 90 d | 180 d | 1 year | Median width |
|---|---|---|---|---|---|
| Test years | 0.71 | 0.77 | 0.76 | 0.73 | ×1.9 |
| All years | 0.74 | 0.78 | 0.78 | 0.77 | ×2.0 |

Coverage by year (1-year horizon): 2018–19 about 0.90, 2022–23 about 0.82, 2024 0.71, 2025 0.56.

A version that splits the window into object error and solar error does no better. The object
error comes from stationary residuals; the solar error comes from 60 years of analog-forecast
errors, mapped to lifetime through a fitted sensitivity. That sensitivity is 0.09 below 300 km,
0.39 at 300–450 km and 0.60 above, rising with altitude as thermospheric physics predicts.

Why one year can't be fixed: solar and atmosphere errors are shared by every object at a given
date. A year like 2025 therefore misses together. Calibration can only catch up once those
objects have re-entered, and with one cycle of test data that is not enough. The remedies are
upstream: a better cycle forecast (archived NOAA/SWPC predictions) and an atmosphere-model
bias correction near solar maximum (e.g. HASDM-calibrated density).

## 9. Correcting NRLMSIS with observed densities

Section 8.4 left one question open. Was the 2024 lateness of the true-weather run an
NRLMSIS bias near solar maximum? We tested that with open satellite densities.

**Data** (ESA swarm-diss server, no account; `scripts/fetch_density.py`). We sampled every
second or third day and kept 5-minute samples:

| Satellite | Years | Altitude | Source |
|---|---|---|---|
| CHAMP | 2000–2010 | 300–450 km | accelerometer (TU Delft) |
| GRACE 1 | 2002–2017 | 400–500 km | accelerometer (TU Delft) |
| Swarm B, C | 2014–2026 | 450–520 km | precise orbits (ESA) |
| GRACE-FO 1 | 2018–2026 | ~490 km | accelerometer (TU Delft) |

For each sample, NRLMSIS 2.1 runs at the satellite's own position and time, with the same
F10.7/Ap drivers as the re-entry model. This gives δ = ln(ρ_obs / ρ_MSIS). We keep daily
medians, about 8,900 days in all.

**What the bias looks like.** δ does not change monotonically with solar flux. It is about −0.55
at solar minimum (2018–2020), about 0 at moderate flux, and about −0.1 to −0.13 at high flux
and through 2024–26. The five satellites agree to within about 0.02.

A constant δ does not matter for re-entry, because the fitted ballistic coefficient absorbs it.
Only the change in δ between the fit window and the forecast window matters.

**Model** (`src/solace/risk/density_bias.py`):

δ = g(F81) + s(F81)·z + b·ln(1+Ap) + satellite offset, with z = (h−450)/100.

- g and s are piecewise linear in F81 and smoothed.
- F81 is clamped to the training range, so the model never extrapolates.
- At each issue date the model is fitted only to data more than 90 days old.
- The correction is added to log ρ for both the fit window and the forecast.
- Below the lowest observed altitude it is either held constant or tapered to zero at 120 km.

**Out-of-sample check on δ itself.** We scored the error in the predicted 1-year change of δ:

| Model | Error | Change vs none |
|---|---|---|
| No correction | 0.125 | — |
| Linear-in-F model | 0.154 | worse |
| Spline, Swarm/GRACE-FO only | 0.096 | −23 % |
| Spline, with CHAMP/GRACE (cycle 23 maximum) | 0.084 | −33 % |

**Re-entry results** (`scripts/run_phase3d.py`): the same 135 dates and 326k forecasts as §8.4,
at the 1-year horizon.

| | Test median | All-years median | Count error per date (all years) |
|---|---|---|---|
| True weather | 13 % | 12 % | 19 |
| True weather + correction | 13 % | 12 % | **15** |
| Analog | 20 % | 18 % | 38 |
| Analog + correction | 20 % | 18 % | 42 |

Held and tapered corrections give the same results to two decimals. Median ln(actual/predicted)
for true weather:

| Year | 2019 | 2020 | 2021 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| Uncorrected | −0.01 | −0.10 | −0.05 | +0.01 | +0.11 | +0.02 |
| Corrected | +0.01 | −0.01 | +0.02 | −0.01 | +0.09 | −0.01 |

What this shows:
- **The correction does what it should with the true weather.** The solar-minimum bias in
  2020–21 disappears, and re-entry count errors fall by about 20 %. Medians barely move, because
  the fitted ballistic coefficient already absorbs most of the bias.
- **2024 is only partly explained by MSIS.** The bias falls from +0.11 to +0.09. Most of the
  2024 lateness lies outside what the density data at 300–520 km show. Likely causes are the
  May and October 2024 storms, and altitudes below 300 km where most objects spend their final
  months.
- **The correction does not help the analog forecast and should stay off there.** In 2018–21 the
  analog's flux error had been cancelling the atmosphere bias. Removing one exposes the other
  (2020 analog bias goes from +0.01 to +0.14).
- **The 2025 miss is a solar-forecast miss, not an atmosphere miss.** With the true weather 2025
  is fine (13 % median, bias −0.01). With the analog it is 31 % (bias +0.12 to +0.14).

The remaining lever is the cycle forecast itself (archived NOAA/SWPC predictions), not the
density model.

## 10. Benchmark: NASA MSFC's archived solar-cycle forecasts

Section 9 showed that the 2025 miss came from the solar-flux forecast. The usual flux forecast
for lifetime and re-entry planning is NASA Marshall's monthly forecast of 13-month smoothed F10.7
and Ap (5/50/95 % percentiles). MSFC has published one every month since 1999, and every issue
is still online. `scripts/fetch_msfc.py` downloads all 331 of them.

**How we use it** (`solace.risk.solar.MSFCForecasts`):
- **No look-ahead.** At each issue date we use the latest forecast from an earlier calendar month.
- **Anchoring.** The median flux is rescaled to today's 81-day F10.7, and the rescaling fades over
  a year, exactly as the analog does. Without anchoring, MSFC's lagged smoothing starts too low
  in 2024 (bias −0.18 at 90 days).
- **Three variants:**
  - **msfc**: MSFC flux and Ap;
  - **blend**: geometric mean of MSFC and analog flux, with the analog's Ap;
  - **msfc_hold**: MSFC on the density-corrected atmosphere of §9.
- **Same inputs as Phase 3c.** All three use the Phase 3c objects and fitted ballistic
  coefficients (§9's for msfc_hold), so only the future weather changes.

**Flux skill.** Median |ln(forecast/observed)| of mean F10.7 over the next L days, at 135 dates:

| | 90 d | 1 year | 2 years | 1 year, 2016–22 | 1 year, 2023+ |
|---|---|---|---|---|---|
| Persistence | 0.065 | 0.086 | 0.127 | 0.057 | 0.149 |
| Analog | 0.054 | 0.067 | 0.080 | 0.062 | 0.103 |
| MSFC, anchored | **0.050** | **0.053** | 0.065 | **0.021** | 0.111 |

**Re-entry, 1-year horizon** (median timing error / count error per date):

| | Test 2023+ | All years | 2020 | 2021 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| Persistence | 19 % / 70 | 22 % / 61 | 33 % | 57 % | 21 % | 22 % |
| Analog | 20 % / 61 | 18 % / 38 | 17 % | 15 % | 20 % | **31 %** |
| MSFC | **18 % / 47** | 18 % / 37 | 20 % | 20 % | 21 % | **20 %** |
| Blend | 19 % / 53 | **17 % / 35** | 18 % | 17 % | 21 % | 24 % |
| MSFC + corrected atmosphere | 20 % / 51 | 18 % / 36 | 17 % | 13 % | 23 % | 21 % |
| True weather (bound) | 13 % / 30 | 12 % / 19 | 17 % | 10 % | 14 % | 13 % |

**80 % window coverage** (online conformal, §8.4), 1-year horizon:

| Around | Test | All years | Worst year | 2025 | Median width |
|---|---|---|---|---|---|
| Analog | 0.73 | 0.77 | 0.56 (2025) | 0.56 | ×2.0 |
| MSFC | 0.79 | 0.82 | 0.58 (2020) | 0.76 | ×2.1 |
| MSFC + corrected atmosphere | **0.81** | 0.79 | **0.73** (2020) | **0.80** | ×2.1 |

What this shows:
- **MSFC fixes the analog's 2025 failure.** After the Cycle 25 maximum the analog expected
  another peak; MSFC expected the decline that happened (`assets/solar_forecasts.png`). The
  2025 error falls from 31 % to 20 %, and the count error on the test years from 61 to 47.
- **A better flux forecast exposes the atmosphere bias.** In 2020–21 MSFC is closer to the real
  flux, yet re-entry gets worse (20 % vs 15–17 %). There the analog's flux error was cancelling
  NRLMSIS's solar-minimum bias. Adding §9's density correction restores those years (17 % and
  13 %). This is the first configuration in which both pieces pull in the same direction.
- **Windows are calibrated every year only with both fixes.** Around MSFC on the corrected
  atmosphere, coverage is 0.73–0.90 every year from 2017 to 2025, against a low of 0.56 around
  the analog. This is the most useful result for operations: the stated uncertainty can be
  trusted through a whole solar cycle, including both turning points.
- **The point forecasts are statistically close.** All realistic forecasts sit at 17–20 % median,
  against 12 % with the true weather. The remaining gap is the unpredictable part of solar
  activity (storms, 27-day rotation, the size of individual peaks) and does not close with a
  better cycle forecast alone. What MSFC buys is robustness: the worst year improves from 31 %
  to 20–21 %.

**Recommended configuration:** the MSFC forecast on the density-corrected atmosphere, with the
calibrated windows around it. Use the blend if only the point error matters.

## 11. How sure are these numbers?

**Ground truth.** The decay dates come from the Spacebook satellite catalogue. Checked against
CelesTrak's SATCAT (`scripts/validate_truth.py`): 87 % of the 6,899 cohort decays since June 2016 match
exactly, 91 % within a day and 99.4 % within a week; only 10 objects are listed as decayed in one
catalogue and not the other.

**Block bootstrap** (`scripts/significance.py`). Forecasts issued on the same date share the same
solar error, so we resample issue dates in circular blocks of six (~6 months), 2,000 times. At one year,
2016–2026:

| | Median error [95 % CI] | Count error per date |
|---|---|---|
| Persistence | 22.5 % [18.2, 28.1] | 55.5 [33.5, 82.3] |
| Analog | 17.5 % [15.2, 20.8] | 36.0 [24.3, 51.6] |
| MSFC | 17.9 % [15.7, 20.5] | 33.3 [23.1, 45.1] |
| True weather | 11.6 % [10.6, 13.3] | 17.2 [12.4, 22.8] |

Cycle-aware forecasts beat persistence (MSFC by 4.6 points [1.5, 8.7], P = 0.002); the realistic flux
forecasts do not differ from each other on the median.

**A score over every forecast** (`scripts/proper_scores.py`). The median error only looks at objects
that re-entered. The Brier score for "re-enters within H days" covers all of them; whether that
happened is known for every object once H has passed, so no censoring correction is needed. Each
forecast becomes a probability through the residuals of earlier, fully resolved forecasts (no
look-ahead). On the 2023–2026 test years MSFC beats the analog at 90, 180 and 365 days (P = 0.018,
0.001, 0.006); over 2018–2026 they tie at one year. Skill against climatology at one year after 2023:
SGP4 0.56, persistence 0.80, analog 0.84, MSFC 0.87, true weather 0.90. The probabilities from the true
weather are close to perfectly reliable (`assets/reliability.png`).

## 12. Payloads and manoeuvre detection

Satellites were left out of the cohort because thrust breaks the drag model. To include them
(`scripts/run_payloads.py`) we need to know, at the issue date, which ones are still being flown.
Drag can only lower the mean semi-major axis, so a rise between two TLEs means thrust. For debris and
rocket bodies the 99.99th percentile of rises between TLEs at most 10 days apart is 0.058 km (TLE noise),
so we flag a manoeuvre when the mean semi-major axis rises by more than 0.1 km
(`solace.risk.reentry.manoeuvre_epochs`) and screen out a payload if it manoeuvred in the 120 days before
the issue date. A long lookback is needed because station-kept satellites often coast for weeks between
burns.

Over 135 issue dates this keeps 314,629 forecasts of 13,807 payloads and screens out about 2,000 per date
(over 7,000 by mid-2026, mostly Starlink). The payloads that kept coasting are forecast as well as debris:
median 1-year error 17.9 % with MSFC and 12.2 % with the true weather (debris 17.9 % and 11.6 %).
6.6 % of payload re-entries within a year came after a later manoeuvre, and for 79 % of those the forecast
was no re-entry at all: they were deliberately de-orbited. Payload windows cover 70 % instead of 80 %:
payload re-entries are fewer and more bunched in time than debris, so the online calibration has less
to learn from. Details in `reports/payloads.json` and paper Section 4.7.
