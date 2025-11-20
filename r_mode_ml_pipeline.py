import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# --- 1. R-Mode Gravitational Wave Frequency Model (Template Generator) ---

def r_mode_gw_frequency(nu, nu_k, A, B):
    """
    Calculates the R-mode gravitational wave frequency (f_GW) based on the 
    pulsar's spin frequency (nu) using the Paper's analytical model (Eq. 11).

    f_GW = A*nu - B/nu^2 * nu_k^2

    Parameters:
    - nu (float): Pulsar spin frequency (Hz).
    - nu_k (float): Keplerian breakup frequency (Hz).
    - A (float): GR and rotational correction parameter (1.39 <= A <= 1.57).
    - B (float): Centrifugal flattening and non-linear rotational effects parameter (0 <= B <= 0.195).

    Returns:
    - float: R-mode gravitational wave frequency (Hz).
    """
    if nu == 0:
        return 0.0
    return A * nu - B * (nu_k**2) / (nu**2)

def r_mode_spin_down_model(nu, nu_0, nu_dot_0, t, l_coeff):
    """
    A simplified model for the spin frequency evolution nu(t) based on the 
    Paper's generalized spin-down model (Eq. 7, focusing on the r-mode term).
    
    The full model is complex, involving coupled differential equations. For this 
    demonstration, we use a simplified, time-dependent decay model where the 
    spin-down rate (nu_dot) is dominated by the r-mode term (nu^7).
    
    nu_dot = -l_coeff * nu^7
    
    This can be integrated to give a time-dependent solution for nu(t).
    For simplicity, we will use a linear approximation for short time scales 
    around the initial spin-down rate, or a simple exponential decay for 
    demonstration purposes, as the full analytical solution involves complex 
    functions (Lambert-Tsallis).

    Here, we will use a simple linear decay for the demonstration:
    nu(t) = nu_0 + nu_dot_0 * t
    
    In a real application, the full numerical solution of the differential 
    equations (Eqs. 20 and 21 from the paper) would be used.
    """
    # Linear approximation for short time scales
    return nu_0 + nu_dot_0 * t

# --- 2. Kalman Filter for R-Mode Frequency Tracking ---

def setup_kalman_filter(initial_f_gw, initial_f_dot, process_noise_std, measurement_noise_std):
    """
    Sets up a Kalman Filter to track the R-mode GW frequency (f_GW) and its derivative (f_dot).
    
    State vector x = [f_GW, f_dot]^T
    """
    # State transition matrix (F): Assumes constant acceleration (or constant nu_dot)
    # x_k = F * x_{k-1} + w_k
    dt = 1.0 # Time step (seconds) - can be adjusted
    F = np.array([[1., dt],
                  [0., 1.]])

    # Measurement function (H): We only measure the frequency f_GW
    # z_k = H * x_k + v_k
    H = np.array([[1., 0.]])

    # Initial state estimate (x): [f_GW, f_dot]^T
    x = np.array([[initial_f_gw],
                  [initial_f_dot]])

    # Initial covariance matrix (P): High uncertainty initially
    P = np.diag([1000., 10.])

    # Measurement noise covariance (R): Based on detector sensitivity (e.g., 1 Hz^2)
    R = np.array([[measurement_noise_std**2]])

    # Process noise covariance (Q): Models the uncertainty in the model (e.g., small changes in nu_dot)
    Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise_std**2)

    # Initialize the Kalman Filter
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = x
    kf.F = F
    kf.H = H
    kf.P = P
    kf.R = R
    kf.Q = Q
    
    return kf

def simulate_and_track(kf, total_time, dt, true_nu_0, true_nu_dot_0, nu_k, A, B, measurement_noise_std):
    """
    Simulates the true R-mode frequency evolution and noisy measurements, 
    then applies the Kalman filter to track the signal.
    """
    time_steps = int(total_time / dt)
    time = np.linspace(0, total_time, time_steps)
    
    # Storage for results
    true_f_gw = []
    measurements = []
    kf_estimates = []
    
    # Initial true spin frequency (nu)
    nu_t = true_nu_0
    
    for t in time:
        # 1. True Signal Evolution (using the Paper's model)
        # Update the true spin frequency (nu) based on the spin-down rate
        # For simplicity, we use the linear approximation for nu(t)
        nu_t = r_mode_spin_down_model(nu_t, true_nu_0, true_nu_dot_0, t, l_coeff=1e-4)
        
        # Calculate the true R-mode GW frequency
        f_gw_true = r_mode_gw_frequency(nu_t, nu_k, A, B)
        true_f_gw.append(f_gw_true)
        
        # 2. Simulate Noisy Measurement (LIGO data)
        # Add Gaussian noise to the true frequency
        measurement = f_gw_true + np.random.normal(0, measurement_noise_std)
        measurements.append(measurement)
        
        # 3. Kalman Filter Prediction and Update
        kf.predict()
        kf.update(np.array([[measurement]]))
        kf_estimates.append(kf.x.copy())

    kf_estimates = np.array(kf_estimates)
    
    return time, np.array(true_f_gw), np.array(measurements), kf_estimates

# --- 3. Demonstration and Visualization ---

if __name__ == "__main__":
    # --- Parameters from the Paper/Astrophysical Context ---
    # Pulsar spin frequency (nu) - e.g., a rapidly spinning neutron star
    NU_0 = 100.0 # Hz
    # Spin-down rate (nu_dot) - typical for a young pulsar
    NU_DOT_0 = -1e-10 # Hz/s (very small, but non-zero)
    # Keplerian breakup frequency (nu_k) - typical value
    NU_K = 500.0 # Hz
    # R-mode model parameters (A and B) - using mid-range values
    A_PARAM = 1.45
    B_PARAM = 0.10
    
    # --- Simulation Parameters ---
    TOTAL_TIME = 3600 * 24 * 30 # 30 days of observation (seconds)
    DT = 3600 * 6 # Time step (6 hours)
    
    # --- Noise Parameters ---
    # Process Noise: Uncertainty in the spin-down model (how much nu_dot changes)
    PROCESS_NOISE_STD = 1e-12 # Standard deviation of the acceleration noise (Hz/s^2)
    # Measurement Noise: Uncertainty in the LIGO measurement (Hz)
    MEASUREMENT_NOISE_STD = 0.5 # Hz (Simulating a noisy, wide-band search)

    # --- Initial Kalman Filter Setup ---
    # Initial estimate of f_GW and f_dot (using the Paper's model for the initial state)
    initial_f_gw = r_mode_gw_frequency(NU_0, NU_K, A_PARAM, B_PARAM)
    # Initial f_dot is approximated from the spin-down rate: f_dot ~ A * nu_dot
    initial_f_dot = A_PARAM * NU_DOT_0 
    
    kf = setup_kalman_filter(initial_f_gw, initial_f_dot, PROCESS_NOISE_STD, MEASUREMENT_NOISE_STD)

    # --- Run Simulation and Tracking ---
    time, true_f_gw, measurements, kf_estimates = simulate_and_track(
        kf, TOTAL_TIME, DT, NU_0, NU_DOT_0, NU_K, A_PARAM, B_PARAM, MEASUREMENT_NOISE_STD
    )

    # --- Plot Results ---
    plt.figure(figsize=(12, 6))
    
    # Convert time to days for better visualization
    time_days = time / (3600 * 24)
    
    # Plot 1: Frequency Tracking
    plt.subplot(2, 1, 1)
    plt.plot(time_days, true_f_gw, 'k--', label='True $f_{GW}$ (Paper Model)')
    plt.plot(time_days, measurements, 'r.', label='Noisy Measurements (LIGO Data)', alpha=0.5)
    plt.plot(time_days, kf_estimates[:, 0], 'b-', label='Kalman Filter Estimate')
    plt.title('R-Mode Gravitational Wave Frequency Tracking (30 Days)')
    plt.xlabel('Time (Days)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Frequency Derivative Tracking (Spin-down)
    plt.subplot(2, 1, 2)
    true_f_dot = A_PARAM * NU_DOT_0 * np.ones_like(time_days) # True f_dot is constant in this linear model
    plt.plot(time_days, true_f_dot, 'k--', label='True $\dot{f}_{GW}$ (Paper Model)')
    plt.plot(time_days, kf_estimates[:, 1], 'g-', label='Kalman Filter $\dot{f}_{GW}$ Estimate')
    plt.title('R-Mode Frequency Derivative Tracking ($\dot{f}_{GW}$)')
    plt.xlabel('Time (Days)')
    plt.ylabel('$\dot{f}_{GW}$ (Hz/s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('r_mode_kalman_tracking.png')
    plt.close()
    
    print("Pipeline execution complete.")
    print("Output files: r_mode_ml_pipeline.py, r_mode_kalman_tracking.png")
