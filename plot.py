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
       
    # --------------------------------------------------
    # Larger fonts everywhere (2× bigger)
    # --------------------------------------------------
    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24
    })

    # ---- Extract arrays ----
    dt = times.datetime
    V = eph["V"]
    alpha = eph["alpha"]
    airmass = eph["airmass"]

    ra_rate = eph["RA_rate"]
    dec_rate = eph["DEC_rate"]
    dec = eph["DEC"]
    ra_proj = ra_rate * np.cos(np.deg2rad(dec))
    sky_rate = np.sqrt(ra_proj**2 + dec_rate**2)/60/60 #arcsec/sec#

    ca_iso = parse_jpl_date(ca["cd"])
    t_ca = datetime.fromisoformat(ca_iso)

    # ---- Figure (now 4 rows) ----
    fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # --------------------------------------------------
    # 1. Vmag
    # --------------------------------------------------
    ax[0].plot(dt, V, color="black", lw=2, label="V mag")
    # ax[0].fill_between(dt, V, where=(V < 17), color="red", alpha=0.35)
    ax[0].set_ylabel("Vmag")
    ax[0].set_ylim(np.min(V)-1, np.max(V)+1)
    ax[0].invert_yaxis()
    ax[0].set_title(f"{asteroid} — Close Approach {ca['cd']}")

    # --------------------------------------------------
    # 2. Phase angle
    # --------------------------------------------------
    ax[1].plot(dt, alpha, lw=2)
    ax[1].set_ylabel("Phase (deg)")

    # --------------------------------------------------
    # 3. Airmass
    # --------------------------------------------------
    ax[2].plot(dt, airmass, lw=2)
    ax[2].set_ylim(1, 2)
    ax[2].set_ylabel("Airmass")

    # --------------------------------------------------
    # 4. Sky rate
    # --------------------------------------------------
    ax[3].plot(dt, sky_rate, lw=2)
    ax[3].set_ylabel("Sky rate\n(arcsec/sec)")

    # --------------------------------------------------
    # Close-approach vertical line
    # --------------------------------------------------
    for a in ax:
        a.axvline(t_ca, color="blue", ls="--", lw=2, alpha=0.8)

    # --------------------------------------------------
    # Night shading 18:00→04:00 UT
    # --------------------------------------------------
    days = sorted(set([d.date() for d in dt]))
    for day in days:
        start_night = datetime.combine(day, datetime.min.time()).replace(hour=18)
        end_night = start_night + timedelta(hours=10)

        for a in ax:
            a.axvspan(start_night, end_night,
                      color="grey", alpha=0.12, linewidth=0)

    # --------------------------------------------------
    # Format x-axis → NO YEAR
    # --------------------------------------------------
    import matplotlib.dates as mdates

    ax[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%b %d")   # e.g. “Jan 12 03:00”
    )
    ax[-1].set_xlabel("UTC")

    plt.tight_layout()
    # plt.show()
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
            if min_v_mag < 17:
                plot_ephemeris(asteroid, ca, times, eph)
            else:
                print(f"skippping, brighters is {min_v_mag}")    


# -----------------------------------------------------
# Example
# -----------------------------------------------------

asteroids = [
    "2017 BP31",
    "2006 BC10",
    "2025 AL2",
    "2000 YV137",
    "2025 MB90",
    "1997 NC1",
    "2025 WC4"
] # Trimester II 2026

# asteroids = [
#     "1999 AO10",
#
# ] # February 2026
#
# asteroids = [
#     "2013 GM3"
# ] # April 2026

process_asteroids(asteroids)
