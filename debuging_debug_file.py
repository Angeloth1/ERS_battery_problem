import fastf1
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# 1. Konfigurace a načtení dat
fastf1.Cache.enable_cache('./fastf1_cache')
session = fastf1.get_session(2026, 'Australia', 'R')
session.load()

# Vybereme nejrychlejší kolo VER
lap = session.laps.pick_driver('VER').pick_fastest()
tel = lap.get_telemetry().copy()

# 2. VYTVOŘENÍ HLADKÉHO PROFILU TRATI (The "Master Map")
# Místo abychom cpali schody do telemetrie, uděláme si čistý profil trati stranou
BIN_SIZE = 10 # metrů
bins = np.arange(0, tel['Distance'].max() + BIN_SIZE, BIN_SIZE)

# Vytvoříme dataframe pro mapu trati
track_map = pd.DataFrame({'DistBin': bins})

# Získáme medián Z pro každý bin ze všech kol (nebo jen z jednoho, pokud chceš)
# Pro demo použijeme jen toto jedno kolo:
tel['DistBin'] = (tel['Distance'] // BIN_SIZE) * BIN_SIZE
z_medians = tel.groupby('DistBin')['Z'].median()

# Namapujeme mediány na naši track_map a doplníme případné díry
track_map['Z_raw'] = track_map['DistBin'].map(z_medians).ffill().bfill()

# --- KLÍČOVÝ KROK: Vyhlazení mapy před interpolací ---
# Tímto odstraníme "schody" dříve, než se dostanou k telemetrii
track_map['Z_smooth'] = gaussian_filter1d(track_map['Z_raw'], sigma=3) # sigma 3 na 10m binech = 30m smoothing

# 3. INTERPOLACE ZPĚT DO TELEMETRIE
# Teď lineárně propojíme body vyhlazené mapy s jemnými body telemetrie (např. každých 0.07 m)
tel['Z_final'] = np.interp(tel['Distance'], track_map['DistBin'], track_map['Z_smooth'])

# 4. VÝPOČET GRADIENTU
# dDistance je jemná, ale dZ už není skoková, ale plynulá
dist = tel['Distance'].values
z_vals = tel['Z_final'].values

# Výpočet sklonu ve stupních
dz = np.gradient(z_vals)
dd = np.gradient(dist)
grad_rad = np.arctan2(dz, dd)
tel['Gradient_Deg'] = np.degrees(grad_rad)

# --- DEBUG LOG ---
print("="*60)
print(f"ROBUST Z-PIPELINE DEBUG (VER Fastest Lap)")
print("="*60)
print(f"Telemetrie vzorků:  {len(tel)}")
print(f"Max Gradient:       {tel['Gradient_Deg'].max():.2f}°")
print(f"Min Gradient:       {tel['Gradient_Deg'].min():.2f}°")
print(f"Průměrná vzdálenost mezi vzorky: {np.diff(dist).mean():.4f} m")

# Kontrola artefaktů (pokud by dD bylo extrémně malé)
critical_jumps = (np.abs(tel['Gradient_Deg']) > 15).sum()
print(f"Kritické body (>15°): {critical_jumps}")

# 5. VOLITELNĚ: Rychlý náhled
# plt.plot(tel['Distance'], tel['Gradient_Deg'])
# plt.title("Hladký gradient bez schodišťového efektu")
# plt.show()