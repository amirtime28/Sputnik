import urllib.request
import json
import time

OUTPUT_FILE = "data.learn"

SOLAR_WIND_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
PROTON_URL = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json"

# ---------- helpers ----------
def get_json(url):
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read().decode())

def clamp(v, a, b):
    return max(a, min(v, b))

def scale(v, i1, i2, o1, o2):
    v = clamp(v, i1, i2)
    return o1 + (v - i1) * (o2 - o1) / (i2 - i1)

# ---------- main loop ----------
for i in range(100000):
    try:
        # ===== 1. СОЛНЕЧНЫЙ ВЕТЕР =====
        wind_data = get_json(SOLAR_WIND_URL)
        last_wind = wind_data[-1]

        # колонка "speed" — индекс 2
        solar_wind_speed = float(last_wind[2])  # km/s (300–800)

        # ===== 2. ПРОТОНЫ >10 MeV =====
        proton_data = get_json(PROTON_URL)
        last_proton = proton_data[-1]

        # поле flux
        proton_flux = float(last_proton["flux"])  # pfu

        # ===== 3. МАСШТАБИРОВАНИЕ =====
        param1 = scale(solar_wind_speed, 300, 800, 10, 50)
        param2 = scale(proton_flux, 1, 1000, 50, 500)

        # ===== 4. РЕЗУЛЬТАТ (угроза) =====
        threat_raw = param1 * param2
        result = scale(threat_raw, 500, 20000, 500, 10000)

        # ===== 5. ЗАПИСЬ =====
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(f"{param1:.2f} {param2:.2f} {result:.2f}\n")

        print(f"OK → {param1:.2f} {param2:.2f} {result:.2f}")

    except Exception as e:
        print("ОШИБКА:", e)
