# Satellite Visibility Simulation 🚀

Simulating how satellites appear from a ground station in **Chiang Mai, Thailand (18.852706° N, 98.958425° E, 351 m)**.  
This project includes both **synthetic constellation modeling** and **real TLE-based orbit propagation**.  

✨ This project connects orbital mechanics with ground station visibility analysis, providing a foundation for constellation design and satellite tracking studies.

## Overview
- [Walker-Delta Simulation](#walker-delta-simulation)
- [TLE Propagation](#tle-propagation)
- [Requirements](#requirements)

## Walker-Delta Simulation
**File:** `walker_delta_draft_positive_visible.py.py`  

Implements a configurable **Walker-Delta constellation** generator.  


## TLE Propagation
**File:** `TLE_All_Visible.py`  

Uses real **Two-Line Elements (TLEs)** and propagates satellite motion with the **SGP4 model**.  


## Requirements
Install dependencies with:
```bash
pip install numpy pandas astropy sgp4 matplotlib
```


## How to Run
Run either script directly:
```bash
python walker_delta_draft_positive_visible.py.py
python TLE_All_Visible.py
```
