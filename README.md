# R-Mode Gravitational Wave Detection — Model-Informed Kalman Tracking Prototype

This repository contains a computational prototype for tracking continuous gravitational waves produced by unstable r-modes in rapidly rotating neutron stars. It extends the analytical model introduced in Valluri & Sambasivam (2017) by embedding the gravitational-wave frequency relation directly into a Kalman Filter.

## Features
- Physics-informed prediction using the analytical f_GW model
- Kalman tracking of frequency and spin-down under heavy noise
- Realistic LIGO-style measurement simulation
- Modular design for future ML or numerical ODE integration extensions

## Files
- `r_mode_ml_pipeline.py` — Full implementation  
- `r_mode_kalman_tracking.png` — Generated figure  
- `R-Mode_Gravitational_Wave_Detection_ML_Pipeline_Prototype.pdf` — Project overview

## How It Works
1. Spin evolution ν(t) is simulated using a simplified model.
2. The analytical r-mode GW frequency equation is applied to obtain f_GW(t).
3. Noisy measurements are generated to mimic LIGO detector output.
4. A Kalman Filter tracks the underlying physical signal.

## Future Extensions
- Numerical integration of full r-mode torque equations
- RNN-based classification of candidate CW signals
- Integration with real LIGO O3/O4 data
