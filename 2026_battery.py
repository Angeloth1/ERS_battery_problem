import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Set your local cache path here
fastf1.Cache.enable_cache('/home/krystofh/Plocha/fast f1/cache')


# ── Physical Constants (2026 Regulations Estimate) ───────────────────────────
AIR_DENSITY     = 1.225   # kg/m³
CAR_MASS        = 768     # kg
CD_A            = 0.65    # m² (Drag coefficient × frontal area)
ROLLING_COEF    = 0.015
G               = 9.81

# ── Detection Parameters ──────────────────────────────────────────────────────
MIN_SPEED_KMH       = 200
THROTTLE_MIN        = 99
SG_WINDOW           = 15      # Savitzky-Golay window (must be odd)
SG_POLYORDER        = 3
BASELINE_DEG        = 3       # Polynomial degree for the engine baseline
BASELINE_N_BINS     = 20      # Number of speed bins for robust fitting
BASELINE_PERCENTILE = 25      # Percentile of force in each bin → "Engine Floor"
                              # (lower = more conservative baseline, more ERS detected)
RESIDUAL_PERCENTILE = 78      # Points ABOVE this percentile of residuals → ERS candidates
                              # (lower = more sensitive; 70–85 is a reasonable range)
MIN_ZONE_METERS     = 50
MERGE_GAP_METERS    = 50      # Merges adjacent boosts if gap is small


# ── Telemetry Cleaning ────────────────────────────────────────────────────────

def clean_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Speed", "Throttle", "Brake", "nGear", "Time", "Distance"]
    df = df[cols].dropna().copy().reset_index(drop=True)
    df["time_s"] = df["Time"].dt.total_seconds()

    dv = df["Speed"].diff()
    dt = df["time_s"].diff().replace(0, np.nan)
    df["accel_raw"] = dv / dt

    # Remove physics-defying spikes
    invalid = df["accel_raw"].abs() > 150
    df = df.iloc[1:][~invalid].reset_index(drop=True)

    if len(df) > SG_WINDOW:
        df["accel_smooth"] = savgol_filter(
            df["accel_raw"].fillna(0),
            window_length=SG_WINDOW,
            polyorder=SG_POLYORDER,
        )
    else:
        df["accel_smooth"] = df["accel_raw"]

    return df


# ── Physical Model ────────────────────────────────────────────────────────────

def drag_force(v_kmh: pd.Series) -> pd.Series:
    """Aerodynamic + Rolling resistance [N]."""
    v_ms = v_kmh / 3.6
    return 0.5 * AIR_DENSITY * CD_A * v_ms**2 + ROLLING_COEF * CAR_MASS * G


def compute_propulsive_force(df: pd.DataFrame) -> pd.Series:
    """Total propulsive force [N] = m·a + F_drag."""
    # Acceleration converted to m/s²: (accel_smooth / 3.6)
    return CAR_MASS * (df["accel_smooth"] / 3.6) + drag_force(df["Speed"])


def fit_engine_baseline(speeds: pd.Series, forces: pd.Series):
    """
    Robust baseline using percentiles in speed bins.
    
    Why percentiles instead of iterative polynomial?
    - We split the speed range into BASELINE_N_BINS.
    - In each bin, we take the BASELINE_PERCENTILE (default 25th).
    - This creates a "floor" representing the Internal Combustion Engine (ICE) 
      performance without significant electrical deployment (ERS).
    - A polynomial is then fitted through these points.
    """
    v = speeds.values
    f = forces.values

    bins = np.linspace(v.min(), v.max(), BASELINE_N_BINS + 1)
    bin_v, bin_f = [], []
    for i in range(BASELINE_N_BINS):
        mask = (v >= bins[i]) & (v < bins[i + 1])
        if mask.sum() > 3:
            bin_v.append(v[mask].mean())
            bin_f.append(np.percentile(f[mask], BASELINE_PERCENTILE))

    if len(bin_v) < BASELINE_DEG + 1:
        # Fallback if bins are empty
        c = np.polyfit(v, f, deg=BASELINE_DEG)
        return np.polyval(c, v), c

    c = np.polyfit(bin_v, bin_f, deg=BASELINE_DEG)
    return np.polyval(c, v), c


# ── ERS Zone Detection ────────────────────────────────────────────────────────

def find_ers_zones(df: pd.DataFrame):
    """
    1) Mask: Full throttle + No brake + High speed + No gear shifts
    2) Total propulsive force (accel + drag correction)
    3) Robust baseline via percentile speed-binning
    4) Residual = Measured_Force − Baseline_Force
    5) Threshold = Percentile of residuals (robust against outliers)
    6) Length filter + Merging nearby zones
    """
    mask = (
        (df["Throttle"] >= THROTTLE_MIN) &
        (df["Brake"] == 0) &
        (df["Speed"] >= MIN_SPEED_KMH) &
        (df["nGear"].diff().abs() == 0)
    )
    cand = df[mask].copy()

    if len(cand) < BASELINE_DEG + 2:
        print("Insufficient data for analysis.")
        return [], None, None

    cand["F_propulsive"] = compute_propulsive_force(cand)
    baseline, _          = fit_engine_baseline(cand["Speed"], cand["F_propulsive"])
    cand["F_baseline"]   = baseline
    cand["F_residual"]   = cand["F_propulsive"] - cand["F_baseline"]

    # ── Percentile Threshold ──
    threshold = np.percentile(cand["F_residual"], RESIDUAL_PERCENTILE)

    q25 = np.percentile(cand["F_residual"], 25)
    q75 = np.percentile(cand["F_residual"], 75)
    print(f"Force Residuals – Median : {np.median(cand['F_residual']):+.0f} N")
    print(f"                – IQR    : {q25:.0f} … {q75:.0f} N")
    print(f"                – Std Dev: {cand['F_residual'].std():.0f} N")
    print(f"ERS Threshold ({RESIDUAL_PERCENTILE}th percentile) : {threshold:.0f} N")
    
    avg_v_ms = cand["Speed"].mean() / 3.6
    print(f"Approx. Min ERS Power: {threshold * avg_v_ms / 1000:.1f} kW "
          f"(at avg speed {cand['Speed'].mean():.0f} km/h)")

    above = cand[cand["F_residual"] > threshold]
    if above.empty:
        print("No points found above threshold.")
        return [], cand, threshold

    # ── Grouping adjacent indices ─────────────────────────────────────────────
    raw_zones = []
    g_start = prev = above.index[0]
    for idx in above.index[1:]:
        if idx - prev > 5:
            raw_zones.append((g_start, prev))
            g_start = idx
        prev = idx
    raw_zones.append((g_start, prev))

    # ── Length Filter ─────────────────────────────────────────────────────────
    zones_m = []
    for s, e in raw_zones:
        d_start = df.loc[s, "Distance"]
        d_end   = df.loc[e, "Distance"]
        if d_end - d_start >= MIN_ZONE_METERS:
            zones_m.append((d_start, d_end, s, e))

    if not zones_m:
        print("No zones met the minimum length requirement.")
        return [], cand, threshold

    # ── Merging Close Zones ───────────────────────────────────────────────────
    merged = [zones_m[0]]
    for zone in zones_m[1:]:
        if zone[0] - merged[-1][1] <= MERGE_GAP_METERS:
            merged[-1] = (merged[-1][0], zone[1], merged[-1][2], zone[3])
        else:
            merged.append(zone)

    print(f"\nERS Zones Detected (after filtering & merging): {len(merged)}")
    for i, (d_start, d_end, *_) in enumerate(merged):
        zone_mask = (cand["Distance"] >= d_start) & (cand["Distance"] <= d_end)
        if zone_mask.any():
            mean_res = cand.loc[zone_mask, "F_residual"].mean()
            mean_v   = cand.loc[zone_mask, "Speed"].mean() / 3.6
            ers_kw   = mean_res * mean_v / 1000
        else:
            ers_kw = float("nan")
        print(f"  Zone {i+1}: {d_start:.0f} m → {d_end:.0f} m  "
              f"(Length: {d_end - d_start:.0f} m, "
              f"Est. ERS Power: {ers_kw:.0f} kW)")

    return merged, cand, threshold


# ── Visualization ─────────────────────────────────────────────────────────────

def plot(df, zones, candidate, threshold_force, driver_name="VER"):
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    ax1, ax2, ax3 = axes
    fig.subplots_adjust(top=0.94, hspace=0.08)

    ERS_COLOR = "orange"
    dist = df["Distance"]

    # ── Speed Plot ──
    ax1.plot(dist, df["Speed"], color="steelblue", lw=1.3, label="Speed")
    ax1.axhline(MIN_SPEED_KMH, color="gray", ls="--", lw=0.8,
                label=f"Min. {MIN_SPEED_KMH} km/h")
    ax1.set_ylabel("Speed (km/h)")
    ax1.legend(fontsize=8, loc="upper left")

    # ── Acceleration Plot ──
    ax2.plot(dist, df["accel_raw"],    color="#cccccc", lw=0.7, label="Accel (raw)")
    ax2.plot(dist, df["accel_smooth"], color="steelblue", lw=1.3,
             label="Accel (Savitzky-Golay)")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_ylabel("Acceleration (km/h/s)")
    ax2.set_ylim(-30, 80)
    ax2.legend(fontsize=8, loc="upper left")

    # ── Physics Residual Plot ──
    if candidate is not None and "F_residual" in candidate.columns:
        cdist = candidate["Distance"]
        res   = candidate["F_residual"]

        ax3.plot(cdist, candidate["F_propulsive"],
                 color="#bbbbbb", lw=0.9, alpha=0.7, label="Total Propulsive Force")
        ax3.plot(cdist, candidate["F_baseline"],
                 color="green", lw=1.5, ls="--", label="Engine Baseline (Percentile Fit)")
        ax3.plot(cdist, res, color="steelblue", lw=1.1,
                 label="Residual Force (F − Baseline)")
        ax3.fill_between(cdist, 0, res, where=res > 0,
                         color="steelblue", alpha=0.2)

        if threshold_force is not None:
            ax3.axhline(threshold_force, color=ERS_COLOR, ls=":", lw=1.3,
                        label=f"ERS Threshold ({RESIDUAL_PERCENTILE}th perc. = {threshold_force:.0f} N)")
        ax3.axhline(0, color="gray", lw=0.5)
        ax3.set_ylabel("Force (N)")
        ax3.set_xlabel("Distance (m)")
        ax3.legend(fontsize=8, loc="upper left")

    # ── ERS Zones Highlights ──
    for i, (d_start, d_end, *_) in enumerate(zones):
        lbl = "Estimated ERS Boost" if i == 0 else ""
        for j, ax in enumerate(axes):
            ax.axvspan(d_start, d_end, color=ERS_COLOR, alpha=0.35,
                       label=lbl if j == 0 else "")
    if zones:
        ax1.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"Telemetry Analysis: {driver_name} – Physical Estimation of ERS Boost Zones (2026 Regulations)",
        fontsize=13,
    )
    plt.show()


# ── Main Execution ────────────────────────────────────────────────────────────

# Note: Adjust season/event as per actual 2026 data availability
session = fastf1.get_session(2026, 1, 'R')
session.load()

driver_code = 'VER'
lap = session.laps.pick_drivers(driver_code).pick_fastest()
telemetry = lap.get_telemetry()

df                    = clean_telemetry(telemetry)
zones, candidate, thr = find_ers_zones(df)
plot(df, zones, candidate, thr, driver_name=driver_code)