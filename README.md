# R-Mode Gravitational Wave Tracking Prototype
### A Physics-Informed Kalman Filter Based on Li et al. (2026)

This repository contains a student research project that connects the analytical
r-mode gravitational-wave model from:

**Li, X., Abbassi, S., Upadhyaya, V., Zhang, X., & Valluri, S. R. (2026).  
_The role of r-modes in pulsar spin-down, pulsar timing, and gravitational waves._  
Journal of High Energy Astrophysics, 49, 100446.**

with a practical computational method for tracking weak continuous gravitational-wave signals.

---

## 🔭 Project Overview

R-modes in rapidly rotating neutron stars can generate continuous gravitational waves.  
Li et al. (2026) provide an analytical formula that links the gravitational-wave frequency  
to the star’s spin frequency and Keplerian breakup frequency.

This project uses that formula as the **prediction model** inside a **Kalman Filter** to track
a simulated gravitational-wave frequency signal over 30 days of noisy “LIGO-like” measurements.

The tracker converges cleanly to the true signal and demonstrates how physics-informed
models can reduce the search space for continuous-wave (CW) detection.

---

## 📁 Repository Contents

- **r_mode_ml_pipeline.py**  
  Full Python implementation of the r-mode frequency model, simplified spin-down model,
  Kalman Filter setup, simulation loop, and plotting.

- **r_mode_kalman_tracking.png**  
  Output figure showing:
  - top panel: true r-mode frequency, noisy measurements, Kalman estimate  
  - bottom panel: derivative tracking

- **R_Mode_Kalman_StudentClean.pdf**  
  A complete student research report explaining the project in simple and clear language.

---

## 🧠 Methods Summary

### 1. Analytical R-Mode Frequency Model  
The equation  
![fgw](eq1_fgw_student.png)  
is implemented as `r_mode_gw_frequency(...)`.

### 2. Simplified Spin-Down Model  
A linear approximation  
![nut](eq2_nut_student.png)  
is used to evolve the star’s spin frequency over short timescales.

### 3. Kalman Filter State  
State vector: (f_GW, f_dot_GW)  
Measurement: noisy f_GW only  
Time step: 6 hours  
Duration: 30 days

### 4. Simulation  
At each time step:
- update spin frequency  
- compute analytic r-mode f_GW  
- add Gaussian noise  
- update Kalman filter estimate  

---

## 📊 Results

The Kalman Filter successfully recovers the true gravitational-wave frequency curve from noisy measurements.  
It also tracks the derivative, even with a simple spin-down model.

The output figure is included as `r_mode_kalman_tracking.png`.

---

## 🔧 Next Steps

Planned future improvements:

- Replace linear spin-down with the non-linear system of equations from Li et al. (2026)
- Add realistic detector noise models (glitches, non-Gaussian noise)
- Apply the method to real LIGO O3/O4 data
- Try neural-network extensions for event classification

---

## 👤 Author

**Aashrith Raj Tatipamula**  
Student Researcher  
Supervised by Prof. S. R. Valluri

