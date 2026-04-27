import fastf1
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# 1. Načtení reálných dat
fastf1.Cache.enable_cache('fastf1_cache') # Uprav cestu ke své cache
session = fastf1.get_session(2026, 'Australia', 'R')
session.load()

# Vybereme VER kolo 43
laps = session.laps.pick_driver('VER')
lap = laps[laps['LapNumber'] == 43].iloc[0]
tel = lap.get_telemetry()

# 2. Simulace tvého "Track Median" (problémový vstup)
# Vytvoříme schody, které vznikají binováním
BIN_SIZE = 10
tel['DistanceBin'] = (tel['Distance'] // BIN_SIZE) * BIN_SIZE
# Simulujeme fallback na median (tady dáváme průměrnou výšku pro každý bin)
track_median = tel.groupby('DistanceBin')['Z'].transform('median')
z_input = track_median.values

# 3. Výpočet gradientu dvěma způsoby
# A) RAW Gradient (Změna výšky / Změna vzdálenosti)
dZ = np.diff(z_input, prepend=z_input[0])
dD = np.diff(tel['Distance'], prepend=tel['Distance'].iloc[0]-0.1)
dD[dD <= 0] = 0.001 # Ošetření nulových kroků

# B) Tvůj SG přístup
z_sg = savgol_filter(z_input, window_length=101, polyorder=3)
grad_sg = np.degrees(np.arctan2(np.gradient(z_sg), np.gradient(tel['Distance'])))

# C) Gaussian přístup (stabilní)
z_gauss = gaussian_filter1d(z_input, sigma=10)
grad_gauss = np.degrees(np.arctan2(np.gradient(z_gauss), np.gradient(tel['Distance'])))

# --- LOGOVÁNÍ PROBLÉMU ---
print("="*60)
print(f"REAL DATA DEBUG (VER Lap 43)")
print("="*60)
print(f"Vzorků: {len(tel)}")
print(f"Průměrný dDistance: {dD.mean():.3f} m")
print(f"Minimální dDistance: {dD.min():.6f} m")

# Najdeme místo, kde je gradient nejhorší
idx_max = np.argmax(np.abs(grad_sg))

print(f"\nKRITICKÝ BOD (Index {idx_max}):")
print(f"Distance: {tel['Distance'].iloc[idx_max]:.2f} m")
print(f"Lokální dDistance: {dD[idx_max]:.6f} m") 
print(f"Lokální dZ (skok v binu): {dZ[idx_max]:.4f} m")
print("-" * 30)
print(f"MAX Savitzky-Golay Gradient: {grad_sg.max():.2f}°")
print(f"MAX Gaussian Gradient:      {grad_gauss.max():.2f}°")

if grad_sg.max() > 15:
    print("\nPROBLÉM DETEKOVÁN:")
    print("Vzdálenost mezi vzorky (dD) je příliš malá vůči skoku ve výšce (dZ).")
    print("Dělení číslem blízkým nule vystřelí arctan k 90 stupňům.")