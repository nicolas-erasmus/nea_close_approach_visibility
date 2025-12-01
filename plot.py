import requests
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from astroquery.jplhorizons import Horizons
from astropy.time import Time
import astropy.units as u


# -----------------------------------------------------
# UTIL: Convert JPL close-approach date → ISO format
# -----------------------------------------------------
def parse_jpl_date(cd_string):
    """
    Converts '1948-Sep-02 08:48' → '1948-09-02 08:48:00'
    """
    try:
        dt = datetime.strptime(cd_string, "%Y-%b-%d %H:%M")
        return dt.isoformat()
    except Exception:
        print(f"[ERROR] Could not parse date: {cd_string}")
        return None


# -----------------------------------------------------
# 1. Query JPL SBDB for *future* close approaches
# -----------------------------------------------------
def get_future_close_approaches(asteroid_name, limit=3):
    url = "https://ssd-api.jpl.nasa.gov/sbdb.api"
    params = {"sstr": asteroid_name, "ca-data": True, "ca-body": "Earth"}

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    if "ca_data" not in data:
        return []

    now = datetime.utcnow()

    future = []
    for ca in data["ca_data"]:
        dt = parse_jpl_date(ca["cd"])
        if dt is None:
            continue

        if datetime.fromisoformat(dt) > now:
            future.append(ca)

    # Sort by date and take first N
    future_sorted = sorted(future, key=lambda x: parse_jpl_date(x["cd"]))
    return future_sorted[:limit]


# -----------------------------------------------------
# 2. Horizons ephemeris (±20 days around CA)
# -----------------------------------------------------
def get_ephemeris(asteroid_name, ca_date_string):
    iso = parse_jpl_date(ca_date_string)
    if iso is None:
        return None, None

    t0 = Time(iso)

    if t0.datetime.year < 1600 or t0.datetime.year > 2500:
        print(f"[WARNING] Date {iso} outside Horizons range — skipping")
        return None, None

    # Create a *proper* Horizons time range query (not TLIST)
    start = (t0 - 20 * u.day).isot
    stop  = (t0 + 20 * u.day).isot

    obj = Horizons(
        id=asteroid_name,
        location="M28",
        epochs={
            "start": start,
            "stop": stop,
            "step": "1h"       # 1-hour sampling, adjust if needed
        }
    )

    try:
        eph = obj.ephemerides()
        # Convert returned times → astropy Time array for plotting
        times = Time(eph['datetime_jd'], format='jd')
        return times, eph

    except Exception as e:
        print(f"[ERROR] Horizons failed: {e}")
        return None, None


# -----------------------------------------------------
# 3. Plot Vmag, phase angle, airmass
# -----------------------------------------------------
def plot_ephemeris(asteroid, ca, times, eph):
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax[0].plot(times.datetime, eph["V"], lw=2)
    ax[0].set_ylabel("Vmag")
    ax[0].set_title(f"{asteroid} – Close Approach {ca['cd']}")

    ax[1].plot(times.datetime, eph["alpha"], lw=2)
    ax[1].set_ylabel("Phase angle (deg)")

    ax[2].plot(times.datetime, eph["airmass"], lw=2)
    ax[2].set_ylabel("Airmass")
    ax[2].set_xlabel("Date (UTC)")
    ax[2].set_ylim(0, 5)

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------
# 4. Main runner
# -----------------------------------------------------
def process_asteroids(asteroid_list):
    for asteroid in asteroid_list:
        print(f"\n============== {asteroid} ==============")

        ca_list = get_future_close_approaches(asteroid)

        if not ca_list:
            print("No future close approaches.")
            continue

        for i, ca in enumerate(ca_list, 1):
            print(f"\nFuture CA {i}: {ca['cd']} (dist={ca['dist']})")

            times, eph = get_ephemeris(asteroid, ca["cd"])
            if times is None:
                continue

            plot_ephemeris(asteroid, ca, times, eph)


# -----------------------------------------------------
# Example
# -----------------------------------------------------
asteroids = ["2024 YR4", "2025 VW"]

process_asteroids(asteroids)
