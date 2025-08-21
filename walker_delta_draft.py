import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta
from astropy import units as u

# -----------------------------
# USER PARAMETERS
# -----------------------------
start_time_str = "2024-10-21 00:00:00"   # UTC start
sim_duration_s = 24 * 3600               # simulate 1 day
dt = 10                                   # time step [s] (use 1–10 s depending on detail)
elevation_mask_deg = 0.0                  # only record passes above this elevation

# Walker-Delta constellation parameters
N = 20      # total satellites
P = 5       # orbital planes
F = 1       # phasing parameter (0..P-1 usually; general definition uses F mod P)
i_deg = 12  # inclination [deg] (set 10-15 as you wish)
h_km = 600  # altitude [km] (set 500-1000)
raan0_deg = 0.0  # reference RAAN
argp_deg = 0.0   # argument of perigee (circular orbit, arbitrary)

# Thailand focus – your ground station (Chiang Mai by default)
lat_gs_deg = 18.852706
lon_gs_deg = 98.958425
h_gs_m = 351.0

# -----------------------------
# CONSTANTS
# -----------------------------
Re_m = 6378137.0             # WGS-84 equatorial radius [m]
mu_earth = 3.986004418e14    # Earth GM [m^3/s^2]
omega_earth = 7.2921150e-5   # rad/s (sidereal)
deg2rad = np.pi/180.0
rad2deg = 180.0/np.pi

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def gmst_angle_rad(t_astropy):
    """
    Compute GMST angle (radians) for rotating ECI->ECEF using a simple IAU 1982-like approximation.
    Good enough for pass planning. For high fidelity, use astropy ERFA sidereal_time.
    """
    # Use astropy's sidereal_time directly for better accuracy if available
    gmst = t_astropy.sidereal_time('mean', 'greenwich').to(u.rad).value
    return gmst

def R3(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c,  s, 0],
                     [-s,  c, 0],
                     [ 0,  0, 1]])

def R1(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0,  0],
                     [0, c,  s],
                     [0,-s,  c]])

def perifocal_to_eci(a_m, i_rad, raan_rad, argp_rad, M_rad):
    """
    Circular orbit propagation:
    E = M; true anomaly ν = M for e=0
    r_pf = [a*cosν, a*sinν, 0]
    v_pf = [-n*a*sinν, n*a*cosν, 0], with n = sqrt(mu/a^3)
    Then rotate by R3(raan)*R1(i)*R3(argp)
    """
    n = np.sqrt(mu_earth / a_m**3)
    nu = M_rad  # circular orbit
    r_pf = np.array([a_m*np.cos(nu), a_m*np.sin(nu), 0.0])
    v_pf = n * np.array([-a_m*np.sin(nu), a_m*np.cos(nu), 0.0])

    Q = R3(raan_rad) @ R1(i_rad) @ R3(argp_rad)
    r_eci = Q @ r_pf
    v_eci = Q @ v_pf
    return r_eci, v_eci

def eci_to_ecef(r_eci, t_astropy):
    theta = gmst_angle_rad(t_astropy)
    return R3(theta) @ r_eci

def ecef_to_llh(ecef):
    x, y, z = ecef
    a = Re_m
    f = 1/298.257223563
    e2 = f*(2-f)
    lon = np.arctan2(y, x)
    r = np.hypot(x, y)
    lat = np.arctan2(z, r*(1 - e2))  # Bowring iteration (1 pass refine)
    for _ in range(2):
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2*sin_lat**2)
        lat = np.arctan2(z + e2*N*sin_lat, r)
    sin_lat = np.sin(lat)
    N = a / np.sqrt(1 - e2*sin_lat**2)
    h = r/np.cos(lat) - N
    return lat, lon, h

def enu_rotation(lat_rad, lon_rad):
    sL, cL = np.sin(lat_rad), np.cos(lat_rad)
    sλ, cλ = np.sin(lon_rad), np.cos(lon_rad)
    # ECEF->ENU rotation
    R = np.array([[-sλ,         cλ,        0],
                  [-sL*cλ, -sL*sλ,    cL],
                  [ cL*cλ,  cL*sλ,    sL]])
    return R

def az_el_slant(r_ecef_sat, r_ecef_gs):
    dr = r_ecef_sat - r_ecef_gs
    lat_gs = lat_gs_deg*deg2rad
    lon_gs = lon_gs_deg*deg2rad
    R = enu_rotation(lat_gs, lon_gs)
    enu = R @ dr
    e, n, u = enu
    az = np.arctan2(e, n) * rad2deg
    if az < 0: az += 360.0
    rng = np.linalg.norm(enu)
    el = np.arcsin(u / rng) * rad2deg
    return az, el, rng

def gs_ecef(lat_deg, lon_deg, h_m):
    lat = lat_deg*deg2rad
    lon = lon_deg*deg2rad
    a = Re_m
    f = 1/298.257223563
    e2 = f*(2-f)
    N = a/np.sqrt(1 - e2*np.sin(lat)**2)
    x = (N + h_m)*np.cos(lat)*np.cos(lon)
    y = (N + h_m)*np.cos(lat)*np.sin(lon)
    z = (N*(1 - e2) + h_m)*np.sin(lat)
    return np.array([x, y, z])

# -----------------------------
# WALKER-DELTA GENERATION
# -----------------------------
def walker_delta_params(N, P, F):
    """
    Returns tuples (plane_id, sat_in_plane, RAAN, M0) for all satellites.
    RAAN_k = RAAN0 + k*360/P
    Within each plane, satellites are spaced by 360/S (S = N/P) in mean anomaly.
    Plane-to-plane phase shift: ΔM_plane = F * 360/N per plane (classic Walker-Delta).
    """
    S = N // P
    sats = []
    for k in range(P):
        raan_k = raan0_deg + (360.0/P)*k
        dM_plane = (F * 360.0 / N) * k
        for s in range(S):
            M0 = (360.0/S)*s + dM_plane
            sats.append((k, s, raan_k, M0))
    return sats

# -----------------------------
# MAIN SIMULATION
# -----------------------------
def simulate_walker_delta():
    a_m = (Re_m + h_km*1000.0)
    i_rad = i_deg * deg2rad
    argp_rad = argp_deg * deg2rad

    sats = walker_delta_params(N, P, F)
    S = N // P
    n_mean = np.sqrt(mu_earth / a_m**3)      # mean motion [rad/s]
    T_orbit = 2*np.pi / n_mean               # period [s]

    t0 = Time(start_time_str, scale='utc')
    times = np.arange(0, sim_duration_s+1, dt)
    r_gs_ecef = gs_ecef(lat_gs_deg, lon_gs_deg, h_gs_m)

    rows = []
    for tsec in times:
        t = t0 + TimeDelta(tsec, format='sec')
        for (plane, s_in_plane, raan_deg, M0_deg) in sats:
            M_rad = (M0_deg*deg2rad + n_mean*tsec) % (2*np.pi)
            raan_rad = raan_deg * deg2rad
            r_eci, _ = perifocal_to_eci(a_m, i_rad, raan_rad, argp_rad, M_rad)
            r_ecef = eci_to_ecef(r_eci, t)

            az, el, slant = az_el_slant(r_ecef, r_gs_ecef)
            if el >= elevation_mask_deg:
                lat_sat, lon_sat, h_sat = ecef_to_llh(r_ecef)
                # Store UTC and local (UTC+7)
                t_local = (t + TimeDelta(7*u.hour)).iso
                rows.append({
                    "Time (UTC)": t.iso,
                    "Time (UTC+7)": t_local,
                    "t_sec": tsec,
                    "Plane": plane,
                    "SatInPlane": s_in_plane,
                    "SatID": f"P{plane:02d}-S{s_in_plane:02d}",
                    "RAAN(deg)": raan_deg,
                    "M0(deg)": M0_deg,
                    "Azimuth (deg)": az,
                    "Elevation (deg)": el,
                    "Slant Range (m)": slant,
                    "Sat Lat (deg)": lat_sat*rad2deg,
                    "Sat Lon (deg)": lon_sat*rad2deg,
                    "Sat Alt (m)": h_sat
                })

    df = pd.DataFrame(rows).sort_values(["t_sec","Plane","SatInPlane"]).reset_index(drop=True)
    outname = f"walker_delta_N{N}_P{P}_F{F}_i{i_deg}_h{h_km}km_{start_time_str[:10]}.csv"
    df.to_csv(outname, index=False)
    print(f"Saved {len(df)} visibility rows to {outname}")
    print(df.head())
    return df

if __name__ == "__main__":
    df = simulate_walker_delta()
