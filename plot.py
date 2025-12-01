import requests
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from astroquery.jplhorizons import Horizons
from astropy.time import Time
from datetime import datetime, timedelta
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
def get_future_close_approaches(asteroid_name, limit=1):
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
    start = (t0 - 6 * u.day).isot
    stop  = (t0 + 6 * u.day).isot

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

    # ---- Extract useful arrays ----
    dt = times.datetime
    V = eph["V"]
    alpha = eph["alpha"]
    airmass = eph["airmass"]

    ca_iso = parse_jpl_date(ca["cd"])
    t_ca = datetime.fromisoformat(ca_iso)

    # ---- Figure ----
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # --------------------------------------------------
    # 1. Vmag plot with red highlights for V < 17
    # --------------------------------------------------
    ax[0].plot(dt, V, color="black", lw=1.8, label="V mag")
    ax[0].fill_between(dt, V, where=(V < 17), color="red", alpha=0.4,
                       label="V < 17 mag")
    ax[0].set_ylabel("Vmag")
    ax[0].set_ylim(np.min(V)-1, np.max(V)+1)
    ax[0].invert_yaxis()     # brighter = higher up
    ax[0].legend(loc="upper right")
    ax[0].set_title(
        f"{asteroid} — Close Approach {ca['cd']}  (JD={eph['datetime_jd'][np.argmin(V)]:.2f})"
    )

    # --------------------------------------------------
    # 2. Phase angle
    # --------------------------------------------------
    ax[1].plot(dt, alpha, lw=2)
    ax[1].set_ylabel("Phase angle (deg)")

    # --------------------------------------------------
    # 3. Airmass
    # --------------------------------------------------
    ax[2].plot(dt, airmass, lw=2)
    ax[2].set_ylabel("Airmass")
    ax[2].set_xlabel("Date (UTC)")
    ax[2].set_ylim(1, 2)

    # --------------------------------------------------
    # Add close-approach vertical line on all panels
    # --------------------------------------------------
    for a in ax:
        a.axvline(t_ca, color="blue", ls="--", lw=1.5, alpha=0.9)

    # --------------------------------------------------
    # Night-time shading (18:00 → next day 04:00 UT)
    # --------------------------------------------------
    # Determine unique days in the plot range
    days = sorted(set([d.date() for d in dt]))

    for day in days:
        start_night = datetime.combine(day, datetime.min.time()).replace(hour=18)
        end_night = start_night + timedelta(hours=10)  # 18:00 → 04:00

        for a in ax:
            a.axvspan(start_night, end_night,
                      color="grey", alpha=0.15, linewidth=0)

    plt.tight_layout()
    plt.savefig(f"{asteroid}_plot.png", dpi=200, bbox_inches="tight")
    plt.close()



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

            min_v_mag = np.min(eph["V"])
            if min_v_mag < 17.5:
                plot_ephemeris(asteroid, ca, times, eph)
            else:
                print(f"skippping, brighters is {min_v_mag}")    


# -----------------------------------------------------
# Example
# -----------------------------------------------------
# asteroids = [
#     "2020 GE3",
#     "2023 VR5",
#     "2025 KR4",
#     "2023 KH4",
#     "2023 KZ1",
#     "2023 BM4",
#     "2021 KN2",
#     "2018 GE",
#     "2011 LT17",
#     "2003 LN6",
#     "2025 WC4",
#     "2015 LM24",
#     "1997 NC1",
#     "2007 ML24",
#     "2023 YO1",
#     "2007 AA2",
#     "2025 PN7",
#     "2025 MB90",
#     "2020 OM",
#     "2020 UR1",
#     "2015 BF",
#     "2025 OW",
#     "2024 RM10",
#     "2000 YV137",
#     "2019 NY2",
#     "2016 BV14",
#     "2013 QC11",
#     "2025 AL2",
#     "2025 DU7",
#     "2025 FY11",
#     "2023 RL",
#     "2005 PJ2",
#     "2025 QM9",
#     "2006 BC10",
#     "2017 BP31",
#     "2007 EK"
# ]

asteroids = [
    "2017 BP31",
    "2006 BC10",
    "2025 AL2",
    "2000 YV137",
    "2025 MB90",
    "1997 NC1",
    "2025 WC4"
]

process_asteroids(asteroids)
