# orbitguard — Project Plan

Orbital debris conjunction screening, hypervelocity impact modelling, space-weather-driven risk
forecasting, and a three-layer live sky tracker. Python library + web app.

---

## 0. Corrections to the original spec

| Original claim | Problem | What we do instead |
|---|---|---|
| Use NISAR SAR imagery to track debris | SAR images the ground. It doesn't see objects in orbit. | **NISAR is one of the protected assets** (LEO, ~747 km, sun-synchronous), along with Sentinel-1, the ISS, Hubble and others. |
| Get the debris catalog from earthaccess / GIBS / CEDA | Those serve Earth-observation products, not a space-object catalog. | CelesTrak (current) + Space-Track (history), plus the fused sources in §3. |
| Compute MOID between satellites | MOID is for heliocentric orbits. | Satellites use TCA, miss distance and Pc. MOID is used in the asteroid tracker (§5B). |
| Use DONKI flares as the main space-weather driver | F10.7 and Kp drive thermospheric drag. | Use F10.7, Ap and Kp as the drivers, and DONKI flares/CMEs/storms as event features. |
| ST-GNN predicts "structural risk" | No labelled structural-damage data exists. | The GNN predicts **shell density and conjunction rate**. Structural risk = flux × P(penetration) from the BLE. |
| Run SPH for every risk estimate | Too slow. | Use the BLE in the risk loop. SPH is for visualisation and spot-checks. |

---

## 1. Architecture

```
            ┌───────────────────── ingest (§3) ─────────────────────┐
 catalog:   │ CelesTrak GP · Space-Track GP history · SATCAT (RCS)  │
 physical:  │ ESA DISCOS (size/mass/shape) · fragmentation events   │
 accuracy:  │ Spacebook synthetic covariance · Starlink ephemerides │
 ground     │ Space-Track public CDMs · SOCRATES · ESA Kelvins CDMs │
  truth:    │                                                       │──▶ data/
 sp. wx:    │ NOAA SWPC F10.7/Kp · CelesTrak SW-All · DONKI · GFZ   │
 atmos:     │ NRLMSISE-00 (pymsis)                                  │
 sky:       │ JPL SBDB/CAD/Sentry · JPL DE440 · Hipparcos           │
            └───────────────────────────────────────────────────────┘
                     │
   ┌─────────────────┼──────────────────────┬────────────────────┐
   ▼                 ▼                      ▼                    ▼
 orbit/           impact/                risk/               live/
 SGP4, screen,    BLE (Whipple),         shell graph,        FastAPI + WebSocket,
 TCA, Pc          2-D SPH hydrocode      ST-GNN forecaster,  CesiumJS: satellites,
                                         risk index          asteroids, sky
   └─────────────────┴──────────┬───────────┴────────────────────┘
                                ▼
            cli: `orbitguard screen | impact | forecast | serve`
```

Package layout: `src/orbitguard/{ingest,orbit,impact,risk,live,viz}`, plus `tests/`, `notebooks/` and `web/`.

---

## 2. Phases (built in order, no fixed timeline)

### Phase 1: Conjunction engine *(code written, now testing)*
- Batch SGP4 propagation, screening (perigee/apogee filter → coarse grid → Brent refinement to TCA), and Foster 2-D Pc.
- Covariance comes in three tiers. Assumed RIC sigmas are the baseline. **Spacebook synthetic covariance** is the upgrade. Starlink operator covariance is used where it exists.
- Hard-body radius comes from DISCOS dimensions where known, otherwise from the SATCAT radar size class (RCS).
- **Validate:** against SOCRATES and Space-Track public CDMs for the same pairs.

### Phase 2: Impact physics
- **BLE:** Christiansen Whipple-shield equations (3 velocity regimes).
- **SPH:** 2-D hydrocode (Mie–Grüneisen aluminium, Monaghan viscosity, KD-tree neighbours).
- **Validate:** SPH hole diameter against empirical fits, and BLE curves against published NASA plots.

### Phase 3: Risk forecasting
- Nodes are altitude × inclination shells, built from weekly historical catalog snapshots (Spacebook, no account; Space-Track optional).
- Node features: object counts by size class, mean B*, conjunction counts, F10.7, Ap, Kp, flare and CME flags, and MSIS density at the shell altitude.
- **Physics baseline first**, then an ST-GNN (GCN + GRU) that has to beat it.
- **Targets:** 30-day shell density and conjunction rate. Conjunction labels come from historical screening plus public CDMs.
- **Risk index per asset:** forecast flux × size distribution × P(d > d_crit).

### Phase 4: Live tracker, all three layers (§5)

### Phase 5: Polish
Tests, CI, docs, a notebook walkthrough, a hosted demo, and a short video.

---

## 3. Data fusion: what we combine and why

NISAR was never our data source; it's one of the satellites we protect. The project already
combines catalog and space-weather data. Below is every source worth fusing, and what each adds.

### Tier 1: no account needed (use now)
| Dataset | Adds | Used in |
|---|---|---|
| **CelesTrak GP** (active + debris clouds) | Current orbits for ~30k objects | Screening, live globe |
| **CelesTrak SOCRATES** | Independent conjunction reports | Validation |
| **NOAA SWPC** F10.7, Kp | Current space weather | Forecast, live dashboard |
| **CelesTrak SW-All.csv** | Daily F10.7/Ap/Kp **back to 1957** | GNN training history |
| **NASA DONKI** (flares, CMEs, geomagnetic storms) | Event flags | GNN features |
| **NRLMSISE-00** via `pymsis` | Thermospheric density from the space-weather inputs | Physics baseline, a physics-informed feature |
| **Spacebook** (COMSPOC) synthetic covariance | Realistic per-object uncertainty in place of guessed values | Better Pc |
| **Spacebook historical catalog** (`/api/entity/tle/{date}`) | Full public catalog as of any date, back to at least 2005 | **Main GNN training data** (replaces Space-Track history) |
| **COMSPOC Spacebook** full catalog + satcat | All ~33k on-orbit objects with names and types | Default catalog for screening and risk |
| **ESA Kelvins CDM dataset** (Zenodo, 2015–2019, ~13k events) | Real operational CDMs with Pc evolution | Pc model validation, ML benchmark |
| **JPL SBDB / CAD / Sentry** | NEO orbits, close approaches, impact-risk list | Live layer B |
| **JPL DE440 + Hipparcos** (via Skyfield) | Planets, Moon, bright stars | Live layer C |

### Tier 2: free account (sign up once)
| Dataset | Adds | Used in |
|---|---|---|
| **Space-Track GP history** | Daily TLE snapshots over years | Optional; Spacebook history now covers this |
| **Space-Track SATCAT** | Radar size class (small/medium/large), launch/decay dates | Size distribution, hard-body radius |
| **Space-Track public CDMs** | Official 18 SDS conjunction messages | **Ground truth** for conjunction-rate labels (the one thing still needing an account) |
| **Starlink ephemerides** (Space-Track/starlink.com) | Operator positions *with covariance*, 72 h, updated 3×/day | Best-case Pc for Starlink pairs |
| **ESA DISCOS** | Size, mass and shape for ~19k objects, plus fragmentation events | Hard-body radius, impact energy, breakup-event features |

### Tier 3: validation-only (published literature)
| Dataset | Adds |
|---|---|
| Hubble solar-array and ISS returned-surface impact surveys | Counts of real impact craters, to check flux × BLE predictions against reality |
| NASA/ESA hypervelocity test reports | Ballistic-limit data points for the BLE and SPH |

**The fusion that matters most:** catalog history + space weather + MSIS density + public CDMs.
That combination is what makes the forecast physically grounded and testable against real
operational data. The rest improves fidelity.

---

## 4. Tech stack
Python 3.11, numpy/scipy, `sgp4`, `skyfield`, `pymsis`, PyTorch, FastAPI + uvicorn (WebSocket),
CesiumJS front end, plotly for static reports.

---

## 5. Real-time features: all three, in one "Live Sky" app

One CesiumJS app with three toggleable layers sharing a single clock, so you can scrub time and
everything moves together.

**A. Satellite & debris globe**
About 30k objects streamed at 1 Hz (bulk SGP4 takes ~10 ms per tick). Colour by type (payload,
rocket body, debris). Click an object for its orbit track, next conjunctions and Pc. Protected
assets (NISAR, ISS, Hubble, ...) are highlighted, with a live risk panel and the current
space weather.

**B. Near-Earth asteroid tracker**
- **Data:** JPL SBDB elements, CAD close approaches and the Sentry risk list.
- **Physics:** heliocentric Kepler propagation with the **Earth MOID computed** for each object.
- **Views:** a solar-system view, a countdown to the next close approach, and Torino/Palermo scale for Sentry objects.
- **Refresh:** JPL data once an hour; positions are recomputed every second.

**C. Planet & star sky view**
- **Data:** Skyfield with JPL DE440 and Hipparcos.
- **Content:** planets, the Moon and bright stars in alt/az for the viewer's location (browser geolocation), plus visible ISS and satellite passes tonight.
- **Views:** a dome view of the sky, plus a "what's overhead" list.

---

## 6. Novelty and publication path

### What already exists
| Existing work | What it does | Gap |
|---|---|---|
| Stuff in Space, LeoLabs visualiser, Spacebook | Live globes of the catalog | Visualisation only, no physics-based risk |
| CelesTrak SOCRATES, Space-Track CDMs | Conjunction lists with Pc | No impact consequence, no forecast |
| NASA BUMPER + ORDEM, ESA MASTER/DRAMA | Penetration risk from environment models | Closed or restricted tools, static environment, no space-weather forecasting |
| ESA Kelvins challenge, CDM ML papers | ML on individual CDM time series (final Pc of one event) | Event-level only, not population or shell-level |
| SDC9 (2025) Bayesian NN → graph models | Space weather → orbit perturbations, GNN planned | Work in progress, no conjunction-rate forecast or damage link |
| GitHub SGP4 conjunction projects | Screening | No validation against real CDMs, no impact physics |

### What would be new here
1. **A space-weather-conditioned spatio-temporal GNN that forecasts conjunction rate per orbital shell over 30 days.** I found no published study doing exactly this; the nearest is the 2025 SDC9 work, which is still in progress.
2. **Linking that forecast to penetration probability** through open ballistic-limit equations. The result is an open, BUMPER-like risk chain that runs for any asset in the public catalog.
3. **Validation against real operational data:** public CDMs and SOCRATES for conjunctions, and Hubble/ISS impact surveys for flux × BLE. Most hobby and portfolio projects skip validation entirely.
4. **Quantifying how much covariance quality changes Pc ranking** (assumed vs Spacebook synthetic vs Starlink operator). This is a small, clean side result.

The live trackers, the SPH solver and the SGP4 screening are **not novel**. They make the
portfolio strong, but they don't belong at the centre of a paper.

### Paper strategy
- **Main paper:** contribution 1 plus 2, i.e. the forecast + risk chain. The headline result must
  be "the GNN beats the physics and persistence baselines on held-out years, including storm
  periods", with uncertainty quantification. If it doesn't beat them, that's still publishable
  as a negative or benchmark result, but in a lower venue.
- **Candidate journals** (check current SJR quartiles before submitting): *Acta Astronautica*,
  *Advances in Space Research*, *Aerospace Science and Technology*, *Space Weather* (AGU),
  *Journal of Space Safety Engineering*.
- **Software alternative:** *SoftwareX* or *JOSS* for the open library itself; quicker, and it works alongside the main paper.
- **What a Q1 paper needs:** a clearly stated research question, a held-out time split, ablations
  (with and without space weather, with and without the graph), comparison to baselines, and
  released code and data.

---

## 6b. New tools worth adopting (checked Nov 2026)

| Tool | What it is | How we'd use it |
|---|---|---|
| **MOCAT-pySSEM** (MIT, open source, 2025) | Source–sink shell model of the debris environment | **Physics baseline for the GNN in the paper.** Reviewers will expect a comparison with a standard environment model |
| **ESA dSGP4** (PyTorch) | Differentiable SGP4 with neural correction layers (ML-dSGP4) | Covariance propagation, or an ML-corrected propagator as a follow-up |
| **Karman** (ESA/Surrey) | Open ML thermosphere-density benchmark; beats NRLMSISE-00 by ~60 % in its paper | Swap the MSIS density feature for a learned density |
| **NASA CCMC MLTDM** | Random-forest density at 400 km, run on demand | Validation of the density feature |
| ORDEM 4.0 | Next NASA debris model (adds debris shape) | Still in development; ORDEM 3.2 is current |

Several small open projects (SatGuard, orbtrisk-cockpit, SIMPLETON) now combine SGP4 screening, Pc
and CesiumJS globes. That confirms the screening and the live globe are portfolio features, not
the research contribution.

## 7. Current state

| Phase | Status |
|---|---|
| 1 Conjunction engine | **Done.** Screening runs in 1.7 s for 3 days × 18.6k objects. Validated against SOCRATES and closed-form Pc (see README). |
| 2 Impact physics | **Done.** BLE (Whipple + single wall) and 2-D SPH validated against the Hugoniot; animated HTML export. |
| 4 Live Sky | **Done.** All three layers work in the browser (globe, asteroids, sky). |
| 3 Risk forecasting | **First real results in** (`reports/phase3_results.json`). 565 weekly snapshots 2016–2026 from Spacebook (no account). The GNN ties the hybrid physics forecast and does not yet beat it; see findings below. |
| 5 Polish | **Done.** Lint clean, 22 offline tests, GitHub Actions CI, executed walkthrough notebook, Dockerfile + Render blueprint, Spacebook covariance in the screening engine and the live app. |

Findings so far, which feed the paper:
- The kinetic-gas conjunction model overestimates rates by ~100× in Starlink shells (4,500/day
  predicted in the 450 km band). Physics-only rate models fail for coordinated constellations,
  which is exactly the gap a learned model with CDM labels can fill.
- NISAR (747 km) carries ~15× the debris flux of the ISS on the full catalog. It sits in the fragmentation-debris band.
- **Phase 3, first results (28-day shell-population forecast, test = last 20 % ≈ Nov 2024–Oct 2026,
  mean of 2 seeds, RMSE of log1p counts, lower is better):**

  | Model | All weeks | Storm weeks (Ap ≥ 50) |
  |---|---|---|
  | Persistence | 0.213 | 0.214 |
  | Pure SGP4 decay (first run) | 0.272 | 0.222 |
  | **Hybrid physics** (payloads persist, debris decays) | **0.203** | 0.207 |
  | GNN, full | 0.207 | 0.208 |
  | GNN, no space weather | **0.203** | **0.204** |
  | GNN, no graph | 0.209 | 0.211 |

  What this means:
  - **Pure decay physics is worse than doing nothing** in constellation shells, because active satellites station-keep. Splitting payloads from debris fixes it. This is a publishable observation in itself.
  - **The graph helps a little** (0.207 vs 0.209).
  - **Space weather doesn't help at this target and horizon.** Total shell population over 4 weeks is dominated by launches, which neither physics nor space weather can see.
  - **The research question needs reframing** before a Q1 submission. Space weather should matter where drag dominates: debris-only decay rates, longer horizons, and storm-driven re-entries.

- **Catalog coverage matters:** CelesTrak's "active" group plus four debris clouds holds only ~2.7k
  debris and rocket bodies. The full public catalog holds ~12.4k. Switching to it doubled
  NISAR's 3-day conjunctions (20 → 46) and raised its debris flux ~2.7×.
- **Covariance realism reshapes the risk picture** (`reports/covariance_study.json`; 455 events,
  6 assets, 3 days, same geometry, two covariance sources):
  - Rank agreement between assumed and Spacebook Pc is weak (Spearman 0.43).
  - 38 events change alert tier, and events above 10⁻⁶ drop from 32 to 10.
  - Debris and rocket-body encounters move both ways (39 down >10×, 14 up >10× of 91).
  - Starlink encounters are systematically under-rated by generic sigmas: 256 of 335 rise >10×,
    with a median +10^10 among non-negligible events, because manoeuvres inflate their real TLE errors.

  A single generic error model mis-ranks events, and in particular under-alerts on manoeuvring
  constellations. This is a clean, publishable side result (contribution 4).

Still to do:
- **Phase 3 reframe:** forecast debris-only shell decay and re-entry counts at 1–6-month horizons
  (where drag and space weather dominate), add MOCAT-pySSEM as the standard environment-model
  baseline, then rerun the ablations. CDM labels (Space-Track) remain optional.
- **Hard-body radius:** DISCOS sizes (needs ESA access).
- **Calibration:** check the risk index against Hubble/ISS impact surveys.
- **Hosting:** push to GitHub and connect Render (needs your accounts).

**Needed from you (optional now):** a free Space-Track.org account, only for public CDM labels. Credentials go in `SPACETRACK_USER` /
`SPACETRACK_PASS` in your own shell. Optionally, ESA DISCOS access and a NASA API key.
