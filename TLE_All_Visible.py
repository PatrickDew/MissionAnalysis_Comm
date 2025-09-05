import numpy as np
from sgp4.api import Satrec, SGP4_ERRORS
from astropy.time import Time, TimeDelta
from astropy.coordinates import TEME, ITRS, CartesianDifferential, CartesianRepresentation
from astropy import units as u
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def ecef_to_enu(lat, lon, dx, dy, dz):
    """Convert ECEF vector to ENU coordinates."""
    lat = np.radians(lat)
    lon = np.radians(lon)
    
    R = np.array([
        [-np.sin(lon), np.cos(lon), 0],
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    ])
    
    enu = R @ np.array([dx, dy, dz])
    return enu

def compute_azimuth_elevation(lat, lon, ecef_sat, ecef_gs):
    """Compute azimuth and elevation from ground station to satellite."""
    dx, dy, dz = ecef_sat - ecef_gs
    e, n, u = ecef_to_enu(lat, lon, dx, dy, dz)
    azimuth = np.degrees(np.arctan2(e, n))
    elevation = np.degrees(np.arcsin(u / np.sqrt(e**2 + n**2 + u**2)))
    return azimuth, elevation

def generate_sun_synchronous_orbit(start_time_str, num_points, tle_line1, tle_line2):
    """Generate sun-synchronous orbit in ECEF coordinates using SGP4."""
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)
    orbit = []
    times = []
    start_time = Time(start_time_str, scale="utc")  # Start time in UTC
    julian_start = start_time.jd
    seconds_interval = 1  # Time step in seconds

    # Generate time steps at 1-second intervals
    for i in range(num_points):
        # Compute the Julian Date for the current time step
        current_jd = julian_start + i * (seconds_interval / (24 * 3600))  # 1 second in Julian Date units
        t = Time(current_jd, format='jd', scale='utc')

        # Propagate the orbit
        error_code, teme_p, teme_v = satellite.sgp4(t.jd1, t.jd2)
        if error_code != 0:
            raise RuntimeError(SGP4_ERRORS[error_code])

        # Convert to TEME coordinates
        teme_p = CartesianRepresentation(teme_p * u.km)
        teme_v = CartesianDifferential(teme_v * u.km/u.s)
        teme = TEME(teme_p.with_differentials(teme_v), obstime=t)

        # Convert to ITRS (ECEF) coordinates
        itrs_geo = teme.transform_to(ITRS(obstime=t))
        
        # Extract position in ECEF
        x, y, z = itrs_geo.cartesian.xyz.to(u.m).value
        orbit.append((x, y, z))
        times.append(t.iso)  # Store UTC time in ISO format

    return np.array(orbit), times

def ecef_to_lat_lon(ecef):
    """Convert ECEF coordinates to latitude and longitude."""
    x, y, z = ecef
    a = 6378137  # Semi-major axis (meters)
    e = 8.1819190842622e-2  # Eccentricity

    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * (1 - e**2))
    lat = np.arctan2(z + e**2 * a * np.sin(theta)**3, p - e**2 * a * np.cos(theta)**3)

    lon = np.degrees(lon)
    lat = np.degrees(lat)
    
    return lat, lon

def compute_satellite_altitude(ecef_sat, Re):
    """Calculate the altitude of the satellite above Earth's surface from ECEF coordinates."""
    x, y, z = ecef_sat
    distance_from_center = np.sqrt(x**2 + y**2 + z**2)
    altitude = distance_from_center - Re
    return altitude

def compute_slant_length(elevation, Re, h, h_S):
    """Calculate the slant length from elevation angle."""
    elevation_rad = np.radians(elevation)
    
    term1 = -(Re + h) * np.sin(elevation_rad)
    term2 = (Re + h)**2 * np.sin(elevation_rad)**2 + (h_S**2 - h**2) + 2 * Re * (h_S - h)
    slant_length_positive = term1 + np.sqrt(term2)
    slant_length_negative = term1 - np.sqrt(term2)
    
    # Typically, we use the positive slant length
    return slant_length_positive

# Parameters
num_points = 86400 #1days
start_time_str = "2025-08-21 00:00:00"  # Start time in UTC
lat_gs = 18.852706   # Latitude of ground station (degrees)
lon_gs = 98.958425   # Longitude of ground station (degrees)
Re = 6371000    # Earth's radius in meters
h = 351          # Altitude of ground station in meters

# Ground station ECEF coordinates
ecef_gs_x = (Re + h) * np.cos(np.radians(lat_gs)) * np.cos(np.radians(lon_gs))
ecef_gs_y = (Re + h) * np.cos(np.radians(lat_gs)) * np.sin(np.radians(lon_gs))
ecef_gs_z = (Re + h) * np.sin(np.radians(lat_gs))
ecef_gs = np.array([ecef_gs_x, ecef_gs_y, ecef_gs_z])

# Example TLE for a sun-synchronous satellite
tle_line1 = '1 56310U 23057C   25232.55885133  .00006108  00000-0  46501-3 0  9998'
tle_line2 = '2 56310   9.9870 233.1280 0000779 271.7418  88.2707 14.96411414127694'

# Generate orbit data
orbit_data, times = generate_sun_synchronous_orbit(start_time_str, num_points, tle_line1, tle_line2)

# Calculate azimuth, elevation, and slant length for each point in the orbit and filter based on elevation
local_times = []
utc_times = []
elevations = []
slant_lengths = []

for utc_time, ecef_sat in zip(times, orbit_data):
    azimuth, elevation = compute_azimuth_elevation(lat_gs, lon_gs, ecef_sat, ecef_gs)
    
    # Filter based on elevation angle
    if -90 <= elevation <= 90:
        h_S = compute_satellite_altitude(ecef_sat, Re)
        slant_length = compute_slant_length(elevation, Re, h, h_S)

        # Convert UTC time to local time (UTC+7)
        local_time = Time(utc_time, scale='utc') + TimeDelta(7 * u.hour)
        
        local_times.append(local_time.iso)
        utc_times.append(utc_time)
        elevations.append(elevation)
        slant_lengths.append(slant_length)

# Create a DataFrame to store filtered results
data = {
    'Time (UTC)': utc_times,
    'Time (UTC+7)': local_times,
    'Elevation (degrees)': elevations,
    'Slant Length (meters)': slant_lengths
}

df = pd.DataFrame(data)

# Save the filtered DataFrame to a CSV file
df.to_csv('TELEOS2_2025_08_21.csv', index=False)

print(df.head())  # Print the first few rows of the filtered DataFrame