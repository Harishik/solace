# Re-entry forecasting through a solar cycle: a ten-year open hindcast with calibrated uncertainty

*Draft for* Space Weather *(AGU). Harishik Jamwal. Code and data products: github.com/Harishik/solace.*

## Key Points

1. A ten-year hindcast of 326,410 re-entry forecasts for 8,060 debris objects and rocket bodies, issued
   every 28 days from 2016 to 2026 with only information available at the time.
2. Physics with drag fitted to each object's own decay cuts the 1-year timing error from 200 % (SGP4)
   to 17–18 %; cycle-aware flux forecasts beat persistence significantly, but not each other.
3. Online adaptive conformal windows stay calibrated through both turning points of the cycle only
   when an archived operational flux forecast and a density-model correction are combined.

## Plain Language Summary

Thousands of dead satellites and rocket stages fall back to Earth every year. When each one will come
down depends mostly on how much the upper atmosphere is heated by the Sun, which follows an 11-year
cycle that is hard to predict. We re-ran ten years of forecasts exactly as a forecaster would have made
them at the time, for every low-flying piece of debris and rocket body, and compared them with when the
objects really fell. Standard orbit predictions are off by a factor of two to three a year ahead. Our
physics model, which learns each object's drag from its own past decay, is off by about 17 % in the
typical case. Knowing the Sun's future activity perfectly would bring this to 12 %, so most of the
remaining error comes from the Sun, not the model. We also issue a date range meant to contain the true
date 80 % of the time. It only holds that promise through a whole solar cycle when we use NASA's own
monthly solar forecasts and correct the atmosphere model with satellite measurements.

## Abstract

Re-entry dates of uncontrolled objects months to a year ahead are dominated by uncertainty in future
solar activity, yet published evaluations of lifetime prediction rarely span a solar cycle or test
whether stated uncertainties hold. We present an open hindcast of 326,410 forecasts for 8,060
debris objects and rocket bodies with perigee below 600 km, issued every 28 days from June 2016 to
September 2026 from weekly public catalogue snapshots, using only data available at each issue date.
Each object's ballistic coefficient is fitted to its own decay over the preceding 21–70 days and
propagated with NRLMSIS 2.1 density, J2 and Sun–Moon secular perturbations. Against 39,814 re-entries
within a year, the median 1-year timing error is 200 % for SGP4 with frozen B\*, 22.5 % [95 % CI 18.2,
28.1] under persistence of the 81-day flux, 17.5 % [15.2, 20.8] with a solar-cycle analog, 17.9 %
[15.7, 20.5] with NASA Marshall's archived monthly forecasts used as issued, and 11.6 % [10.6, 13.3]
with the true future weather. Cycle-aware forecasts beat persistence significantly over the decade
(block-bootstrap P = 0.002 for MSFC), but no realistic flux forecast beats another: the remaining gap
to the true-weather bound is a solar-predictability floor. A correction to NRLMSIS fitted to CHAMP,
GRACE, GRACE-FO and Swarm densities removes the model's solar-minimum bias (2020 median log error
−0.10 → −0.01) and lowers re-entry count errors by 20 % with known weather, without changing timing.
Online adaptive conformal 80 % windows reach 0.81 coverage after 2023 and stay at 0.73–0.90 in every
year from 2017 to 2025 only when the MSFC forecast and the density correction are combined; around
the analog alone, coverage falls to 0.56 in 2025.

## 1. Introduction

Uncontrolled re-entries of spacecraft, upper stages and fragmentation debris are a routine part of the
orbital environment and a growing one: the public catalogue of tracked objects on orbit grew from
17,791 to 32,768 between January 2019 and October 2026 (our snapshots). How
long an object stays up is governed by atmospheric drag, which in the thermosphere varies by an order of
magnitude over the 11-year solar cycle. For horizons of weeks, operational centres reach mean errors of
about 10 % (Pardini & Anselmo, 2018, analysing 316 predictions from 20 IADC campaigns); for horizons of
months to a year, the dominant uncertainty is the future solar extreme-ultraviolet input, usually
represented by the F10.7 radio flux (Vaughan et al., 1999).

Three gaps motivate this work. First, evaluations of lifetime prediction are usually case studies or
single campaigns; we know of no open evaluation that issues forecasts for a whole population, at fixed
cadence, through a full phase of the solar cycle, with no information from the future. Second, the
solar-flux forecasts used for lifetime planning, such as NASA Marshall Space Flight Center's (MSFC)
monthly statistical forecast (Niehuss et al., 1996; Suggs, 2017), are assessed on flux, not on the
re-entry dates they drive. Third, uncertainty statements for re-entry dates are rules of thumb (e.g.
±20–30 % windows; Pardini & Anselmo, 2018), and are rarely tested for calibration when the solar cycle
turns.

We address all three with a hindcast built from public data: weekly catalogue snapshots, the observed
space-weather record, MSFC's archive of monthly forecasts, and open accelerometer- and GPS-derived
thermospheric densities (Siemes et al., 2023; van den IJssel et al., 2020). Our contributions are:

1. an open, leakage-free ten-year hindcast of re-entry timing for 8,060 objects, with block-bootstrap
   uncertainty on every headline number;
2. an error decomposition that separates object (drag-model) error from solar-forecast error by
   re-running every forecast with the true future weather;
3. the first evaluation, to our knowledge, of MSFC's archived forecasts *as issued* through re-entry
   outcomes, against persistence and a solar-cycle analog;
4. a lightweight correction of NRLMSIS 2.1 fitted to 8,900 days of satellite densities, and its effect
   on re-entry forecasts;
5. online adaptive conformal windows (Gibbs & Candès, 2021) for re-entry dates, evaluated for coverage
   and interval score year by year.

## 2. Data

**Catalogue history.** 565 weekly snapshots of the public two-line-element (TLE) catalogue from COMSPOC
Spacebook (January 2016 to October 2026; no account required). From these we keep debris and rocket
bodies (object types from the Spacebook satellite catalogue) and every distinct TLE with perigee below
800 km. Payloads are excluded because many manoeuvre.

**Truth.** Decay dates from the Spacebook satellite catalogue. An object counts as re-entered within
horizon *H* of an issue date only if the decay date is known and the catalogue extends at least *H*
beyond the issue date (right-censoring is handled by the `known_until` field).

**Space weather.** Daily observed F10.7 and Ap from the CelesTrak space-weather file (1957–2026).

**Solar forecasts.** All 331 MSFC monthly forecasts of 13-month smoothed F10.7 and Ap (5/50/95 %
percentiles), April 1999 to October 2026, from NASA's public archive.

**Thermospheric densities.** From ESA's Swarm dissemination server: GRACE-FO 1 (2018–2026) and
CHAMP (2000–2010) and GRACE 1 (2002–2017) accelerometer-derived densities from TU Delft (Siemes et al.,
2023), and Swarm B and C GPS-derived densities (2014–2026; van den IJssel et al., 2020). Files were
sampled every 2–3 days and reduced to 5-minute samples; 8,921 daily medians in total.

## 3. Methods

### 3.1 Issue dates and objects

Issue dates run every 28 days from 6 June 2016 to 14 September 2026 (135 dates). At each date an object
enters the forecast set if its latest TLE is at most 10 days old, its perigee is between the re-entry
altitude and 600 km, and an earlier TLE exists 21–70 days before the latest (the drag-fit baseline).
This yields 326,410 forecasts of 8,060 distinct objects (26 % rocket bodies), of which 5,350 re-entered
during the study. Years from 2023 onward (49 dates, covering the Cycle 25 maximum) are reported
separately as the test period; no model component was fitted on them.

### 3.2 Orbit and drag model

Mean semi-major axis and eccentricity are propagated with orbit-averaged drag (32-node Gauss quadrature
over eccentric anomaly), J2 secular rates and Sun–Moon secular perturbations of eccentricity, inclination
and argument of perigee (point-mass third bodies, low-precision ephemerides). Density is NRLMSIS 2.1
(Emmert et al., 2022; its thermosphere is that of NRLMSIS 2.0, Emmert et al., 2021), tabulated against
altitude and averaged over latitude (area-weighted) and local time, with the previous day's F10.7, the
trailing 81-day mean and the daily Ap. Re-entry is declared at 120 km.

The ballistic coefficient *B* of each object is fitted by secant iteration so that the modelled decay
of mean semi-major axis between the two baseline TLEs equals the observed decay, under the observed
space weather (in the spirit of Saunders et al., 2012; Sang et al., 2013). The same *B* is used for every
future-weather scenario of that object and date.

### 3.3 Future space weather

* **Persistence:** F10.7 and Ap held at their trailing 81-day means.
* **Solar-cycle analog:** the current cycle's phase (days since the last minimum known at the issue
  date) is read forward in every earlier cycle since 1957; each is rescaled to today's 81-day flux, with
  the rescaling fading over two years. The point forecast is the ensemble median.
* **MSFC:** the latest monthly forecast issued in an earlier calendar month than the issue date (so it
  was public), median F10.7 and Ap, interpolated to days, rescaled to today's 81-day flux with the
  rescaling fading over one year.
* **Blend:** geometric mean of the MSFC and analog flux.
* **Oracle:** the observed weather (an upper bound on what any flux forecast can achieve).

### 3.4 Density-model correction

For each satellite sample, NRLMSIS 2.1 is run at the satellite's position and time with the same
drivers as the re-entry model, giving δ = ln(ρ_obs/ρ_MSIS). Daily medians are modelled as

  δ = g(F̄) + s(F̄)·(h − 450 km)/100 km + b·ln(1 + Ap) + satellite offset,

with g and s piecewise linear in the 81-day flux F̄ (knots 65–240 sfu, second-difference penalty) and F̄
clamped to the training range. At each issue date the model is fitted only to observations more than
90 days old, and added to log density in both the fit window and the forecast. Below the lowest observed
altitude the correction is held constant (a taper to zero at 120 km gave identical results).

### 3.5 Calibrated windows

Windows are formed on the log ratio of actual to forecast time. At each issue date, the calibration set
is every earlier forecast whose object had already re-entered by that date (no look-ahead). Quantiles are
taken within groups of predicted lead time (0–30, 30–90, 90–180, 180–365, >365 days) × perigee band
(<300, 300–450, ≥450 km), falling back to lead time alone below 100 samples. The miscoverage level α is
updated as outcomes arrive, α ← α + γ(0.2 − observed miss rate), with γ = 0.05 (adaptive conformal
inference; Gibbs & Candès, 2021).

### 3.6 Evaluation

The primary metric is the median relative timing error |t̂ − t|/t at a 1-year horizon, over re-entries
within the horizon. We also report the share within 20 %, the absolute error in the number of re-entries
per issue date, coverage of the 80 % windows, and the interval score in log-time (Gneiting & Raftery,
2007), which rewards narrow windows and penalises misses. Forecasts issued on one date share the same
solar error, and neighbouring dates the same phase of the cycle, so we resample issue dates in circular
blocks of six (~6 months) with 2,000 replicates; paired differences use the same draws.

## 4. Results

### 4.1 Headline

**Table 1.** 1-year horizon. Median relative timing error and re-entry count error per issue date, with
95 % block-bootstrap intervals.

| Method | Median error, 2016–2026 | Median error, 2023+ | Count error, 2016–2026 |
|---|---|---|---|
| SGP4, frozen B\* | 200 % [140, 528] | 146 % [98, 212] | 112 [65, 173] |
| Physics, persistence | 22.5 % [18.2, 28.1] | 19.4 % [14.8, 25.2] | 55.5 [33.5, 82.3] |
| Physics, solar-cycle analog | 17.5 % [15.2, 20.8] | 19.5 % [15.3, 27.3] | 36.0 [24.3, 51.6] |
| Physics, MSFC as issued | 17.9 % [15.7, 20.5] | 18.4 % [14.9, 22.3] | 33.3 [23.1, 45.1] |
| Physics, MSFC/analog blend | 17.1 % [15.3, 20.0] | 18.9 % [14.6, 24.4] | 32.2 [22.3, 43.7] |
| Physics, MSFC + corrected density | 17.6 % [15.0, 20.5] | 19.7 % [15.4, 23.7] | 32.4 [22.6, 44.4] |
| Physics, true weather (bound) | 11.6 % [10.6, 13.3] | 12.8 % [11.6, 14.2] | 17.2 [12.4, 22.8] |

Fitting drag to each object's own decay removes most of the error of the standard TLE-based approach.
Over the decade, cycle-aware forecasts beat persistence: analog by 5.0 points [0.2, 11.1] (P = 0.021),
MSFC by 4.6 points [1.5, 8.7] (P = 0.002), with count errors 20–22 lower per date. Among the realistic
flux forecasts, no difference is significant (e.g. MSFC − analog: +0.4 points [−2.7, +3.6] over the
decade, −1.1 [−6.8, +2.2] after 2023).

### 4.2 Where the error comes from

Re-running every forecast with the true future weather sets an upper bound of 11.6 % (12.8 % after
2023). The gap between realistic forecasts and this bound (6 points over the decade) is the part of the
error attributable to the solar forecast; the bound itself is the object and density-model error.
Persistence fails when the cycle turns (33 % in 2020, 57 % in 2021); the analog and MSFC stay at 15–20 %
in those years. In 2025, after the Cycle 25 maximum, the analog expected a second peak that never came
(31 %), while MSFC followed the decline (20 %). With the true weather 2025 is 13 %, so the 2025 miss is a
solar-forecast miss. *(Figures: `assets/error_by_year.png`, `assets/solar_forecasts.png`.)*

**Table 2.** Flux skill: median |ln(forecast/observed)| of mean F10.7 over the next L days.

| | 90 d | 1 year | 2 years | 1 year, 2016–22 | 1 year, 2023+ |
|---|---|---|---|---|---|
| Persistence | 0.065 | 0.086 | 0.127 | 0.057 | 0.149 |
| Analog | 0.054 | 0.067 | 0.080 | 0.062 | 0.103 |
| MSFC, anchored | 0.050 | 0.053 | 0.065 | 0.021 | 0.111 |

MSFC's anchored forecast is the most skilful flux forecast at 90 days to 1 year, yet its re-entry errors
are statistically indistinguishable from the analog's. Two effects explain this. Errors in flux are only
partly transmitted to timing (the lifetime sensitivity to flux rises with altitude, from ~0.1 below
300 km to ~0.6 above 450 km), and in 2020–21 a more accurate flux exposes a density-model bias that the
analog's flux error happened to cancel (Section 4.3).

### 4.3 Density-model bias

Across 8,921 days, δ is not monotonic in solar flux: about −0.55 at solar minimum (2018–2020),
consistent with the anomalously low densities first reported for 2008 (Emmert et al., 2010), near zero
at moderate flux, and −0.10 to −0.13 at high flux and through 2024–26. The five satellites agree to
within 0.02. The fitted model reduces the out-of-sample error in the predicted 1-year change of δ by
33 % (0.125 → 0.084).

In re-entry forecasts with the true weather, the correction removes the solar-minimum bias (median
log error 2020: −0.10 → −0.01; 2021: −0.05 → +0.02) and lowers the count error from 19 to 15 per
date, but leaves the median timing error unchanged (11.6 % → 11.7 %), because the fitted ballistic
coefficient already absorbs any bias that is constant over the fit window. 2024 is only partly explained
(+0.11 → +0.09); storms and the lowest altitudes, which the satellite data do not reach, are likely
contributors.

### 4.4 Calibrated windows

**Table 3.** 80 % windows at the 1-year horizon: coverage and interval score (log-time; lower is better).

| Window around | Coverage 2016–26 | Coverage 2023+ | Interval score 2023+ | Worst year |
|---|---|---|---|---|
| Analog | 0.77 [0.70, 0.82] | 0.73 [0.61, 0.83] | 1.56 [1.32, 1.86] | 0.56 (2025) |
| MSFC | 0.82 [0.76, 0.85] | 0.79 [0.72, 0.85] | 1.36 [1.22, 1.50] | 0.58 (2020) |
| MSFC + corrected density | 0.79 [0.75, 0.83] | 0.81 [0.75, 0.87] | 1.43 [1.30, 1.57] | 0.73 (2020) |

Windows are about ×2 wide in time (median ratio of upper to lower bound). Because solar and density
errors are shared by every object at a given date, a single bad year can only be absorbed after its
objects re-enter; the adaptive level then widens subsequent windows. Only the combination of MSFC and
the density correction keeps every year between 0.73 and 0.90, including both turning points of the
cycle (2020 minimum, 2024–25 maximum). *(Figure: `assets/window_coverage.png`.)*

### 4.5 Eccentric orbits

For objects with e ≥ 0.2 (GTO and Molniya-type stages), Sun–Moon secular perturbations reduce the
true-weather error after 2023 from 34 % to 16 %; at e < 0.2 they change nothing. A Soviet SL-6 stage
(apogee ~40,000 km) never re-enters without them and re-enters 27 days early with them.

### 4.6 Ablations

*(Pending: NRLMSISE-00 in place of NRLMSIS 2.1; drag-fit baselines of 14–40 and 40–100 days.)*

### 4.7 What did not work

* **Learned residuals.** A gradient-boosted correction of the analog forecast trained on object features
  raised the test error from 20 % to 26 %; the true-weather error is already nearly unbiased each year,
  and a training set dominated by 2022 does not transfer across the cycle.
* **Quantile regression windows** covered only 40 % on the test years.
* **Splitting windows** into object and solar parts did no better than the joint conformal windows.

## 5. Discussion

**A solar-predictability floor.** All realistic flux forecasts sit at 17–20 % median 1-year error,
against 12 % with perfect knowledge of the weather. Better cycle-amplitude forecasts move flux skill
(Table 2) more than they move re-entry skill, because storms, 27-day rotation and the size of
individual peaks are not predictable months ahead and act on every object at once. The practical
consequence is that the value of a better solar forecast lies less in the median than in the tails: the
worst year improves from 31 % (analog) to 20 % (MSFC), and calibrated windows become trustworthy through
a full cycle.

**Operational use.** The method needs only public TLEs, the space-weather record and the MSFC archive;
it runs on a laptop and is issued daily in the accompanying open application. Calibration requires no
tuning beyond the hindcast itself, and the calibration set grows as objects re-enter.

**Limitations.** (i) The truth is the public catalogue decay date; we did not compare with operational
predictions such as 18th Space Defense Squadron decay messages, which require an account. (ii) Density is
averaged over latitude and local time; this is appropriate for lifetime but not for the final orbits.
(iii) The analysis covers the declining phase of Cycle 24 and Cycle 25 only. (iv) Several method choices
(MSFC anchoring, the density correction) were made after seeing the 2024–25 results; we therefore report
all variants and their uncertainty, and emphasise the full-decade numbers. (v) Payloads are excluded.

## 6. Conclusions

On a decade of public data, a drag-fitted physics model reduces the 1-year re-entry timing error of the
standard TLE approach by an order of magnitude. Cycle-aware solar forecasts beat persistence, but the
choice among them hardly matters for the median, which sits about 6 points above a true-weather bound.
What does matter is uncertainty: combining NASA Marshall's archived forecasts with a satellite-calibrated
density correction gives 80 % windows that stay calibrated through both the minimum and maximum of a
solar cycle.

## Open Research

Code, scripts that rebuild every number and figure, and the hindcast tables are at
github.com/Harishik/solace (to be archived on Zenodo with a DOI at submission). Inputs are public:
COMSPOC Spacebook (catalogue snapshots and satellite catalogue), CelesTrak (space weather), NASA MSFC
(solar-cycle forecast archive) and ESA's Swarm dissemination server (Swarm, CHAMP, GRACE and GRACE-FO
densities).

## References

*(Bibliographic details to be completed from the publishers' records before submission; every
entry below was located in a literature database.)*

Emmert, J. T., et al. (2010). Record-low thermospheric density during the 2008 solar minimum.
*Geophysical Research Letters*.

Emmert, J. T., et al. (2021). NRLMSIS 2.0: A whole-atmosphere empirical model of temperature and neutral
species densities. *Earth and Space Science*.

Emmert, J. T., et al. (2022). NRLMSIS 2.1: An empirical model of nitric oxide incorporated into MSIS.
*Journal of Geophysical Research: Space Physics*, 127.

Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under distribution shift. *Advances in
Neural Information Processing Systems*, 34.

Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation.
*Journal of the American Statistical Association*, 102(477), 359–378.

Niehuss, K. O., Euler, H. C., & Vaughan, W. W. (1996). *Statistical technique for intermediate and
long-range estimation of 13-month smoothed solar flux and geomagnetic index* (NASA TM-4759).

Pardini, C., & Anselmo, L. (2018). Assessing the risk and the uncertainty affecting the uncontrolled
re-entry of manmade space objects. *Journal of Space Safety Engineering*.

Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical
Association*, 89(428), 1303–1313.

Sang, J., et al. (2013). Estimation of ballistic coefficients of low altitude debris objects from
historical two line elements. *Advances in Space Research*.

Saunders, A., Swinerd, G. G., & Lewis, H. G. (2012). Deriving accurate satellite ballistic coefficients
from two-line element data. *Journal of Spacecraft and Rockets*, 49(1), 175–184.

Siemes, C., et al. (2023). New thermosphere neutral mass density and crosswind datasets from CHAMP,
GRACE, and GRACE-FO. *Journal of Space Weather and Space Climate*, 13.

Suggs, R. (2017). *The MSFC Solar Activity Future Estimation (MSAFE) model*. NASA NTRS 20170008081.

van den IJssel, J., et al. (2020). Thermosphere densities derived from Swarm GPS observations.
*Advances in Space Research*.

Vaughan, W. W., et al. (1999). The NASA Marshall solar activity model for use in predicting satellite
lifetime. *Advances in Space Research*.
