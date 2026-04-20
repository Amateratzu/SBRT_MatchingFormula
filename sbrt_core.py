## Libraries
import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.optimize import minimize

# Part 1: Monte Carlo Simulation for Margin Calculation
### MonteCarlo: Gold Standard
def generate_patient_errors(tau, sigma, n):
    """
    Returns:
    tuple: (systematic_error, random_errors)
           - systematic_error is a 1D array of size 3 (x, y, z).
           - random_errors is a 2D array of size (n, 3), representing n fractions in 3D.
    """
    systematic_error = np.random.normal(loc=0.0, scale=tau, size=3)
    random_errors = np.random.normal(loc=0.0, scale=sigma, size=(n, 3))
    
    return systematic_error, random_errors

def generate_ctv_surface_grid(radius=30.0, num_points=512):
    """
    Generates an evenly spaced grid of points on the surface of a sphere (CTV).
    The paper mentions that 512 points are sufficient for accuracy.
    """
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    
    return np.vstack((x, y, z)).T * radius

def calculate_accumulated_dose(points, radius, sys_err, rand_errs, margin, sigma_p):
    """
    Returns:
    float: The minimum accumulated dose to any point on the CTV surface 
           (as a fraction of the prescribed dose, e.g., 0.95 means 95%).
    """
    normals = points / radius
    n_fractions = len(rand_errs)
    total_dose = np.zeros(len(points))
    
    for i in range(n_fractions):    
        displacement = sys_err + rand_errs[i]
        outward_shifts = np.dot(normals, displacement)
        distance_to_edge = margin - outward_shifts
        fraction_dose = norm.cdf(distance_to_edge, loc=0.0, scale=sigma_p)
        total_dose += fraction_dose
    normalized_total_dose = total_dose / n_fractions
    min_ctv_dose = np.min(normalized_total_dose)
    
    return min_ctv_dose

def margin_objective_function(margin, n_patients, n_fractions, tau, sigma, sigma_p, ctv_grid, radius):
    """
    Returns:
    float: (actual_coverage - target_coverage). The root finder seeks to make this 0.
    """
    target_dose = 0.95     # b = 0.95 (95% of prescribed dose)
    target_coverage = 0.90 # a = 0.90 (90% of patients)
    success_count = 0
    
    # Simulate the population
    for i in range(n_patients):
        sys_err, rand_errs = generate_patient_errors(tau, sigma, n_fractions)       
        min_dose = calculate_accumulated_dose(ctv_grid, radius, sys_err, rand_errs, margin, sigma_p)
        if min_dose >= target_dose:
            success_count += 1
    actual_coverage = success_count / float(n_patients)
    
    return actual_coverage - target_coverage

def find_optimal_margin_mc(n_patients, n_fractions, tau, sigma, sigma_p, ctv_grid, radius):
    """
    Returns:
    float: The optimal margin width.
    """
    # Pre-generate the patient population to keep the objective function deterministic
    patient_errors = []
    for _ in range(n_patients):
        sys_err, rand_errs = generate_patient_errors(tau, sigma, n_fractions)
        patient_errors.append((sys_err, rand_errs))
    def margin_objective(test_margin):
        success_count = 0    
        for sys_err, rand_errs in patient_errors:
            min_dose = calculate_accumulated_dose(ctv_grid, radius, sys_err, rand_errs, test_margin, sigma_p)
            if min_dose >= 0.95:
                success_count += 1
        actual_coverage = success_count / float(n_patients)
        
        return actual_coverage - 0.90
    try:
        optimal_margin = brentq(margin_objective, a=0.1, b=15.0, xtol=0.01)
    except ValueError:
        print("Error: Could not find the margin in the [0.1, 15.0] range.")
        return None
        
    return optimal_margin

# Part 2: Analytical Margin Recipes
### AVH
def calculate_avh_margin(n, tau, sigma, sigma_p, b=0.95):
    """
    Calculates the Adjusted van Herk (AVH) margin for hypofractionated radiotherapy.
    Returns the margin to the prescription dose line (e.g., 95% or 80%).
    """
    
    # Step 1: Effective errors
    tau_prime = math.sqrt(tau**2 + (sigma**2) / n)
    sigma_prime = math.sqrt(((n - 1) / n) * (sigma**2))
    
    # Step 2: Recipe coefficients
    c_tau = 2.5
    
    if b == 0.95:
        c_sigma = 1.645
    elif b == 0.80:
        c_sigma = 0.84
    else:
        c_sigma = norm.ppf(b) 
        
    # Step 3: Compute margin to the 50% dose line
    m_50 = c_tau * tau_prime + c_sigma * math.sqrt(sigma_prime**2 + sigma_p**2)
    
    # Step 4: Adjust to the prescription dose line (95% or 80%) as per Section 3.1
    m_avh = m_50 - (c_sigma * sigma_p)
    
    return m_avh

### SDE
def calculate_fractionlet_margins(m, sigma_p, I=250):
    """
    Calculates the margin m_i for each fraction-let to model the blurred dose distribution.
    This corresponds to Equation (1) in the Herschtal (2012) paper.
    
    Parameters:
    m       (float): The current estimated margin (distance to 50% dose line).
    sigma_p (float): Penumbral width standard deviation.
    I       (int)  : Number of fraction-lets (the paper uses 250 for accuracy).
    
    Returns:
    numpy.ndarray: An array containing the margins (m_i) for each fraction-let.
    """
    
    # Step 1: Create an array of indices from 1 to I
    i_values = np.arange(1, I + 1)
    
    # Step 2: Calculate the probabilities centered around 0.5 to avoid bias
    # The (i - 0.5)/I ensures values are perfectly symmetric around the 50% mark
    probabilities = (i_values - 0.5) / I
    
    # Step 3: Compute the inverse cumulative Gaussian (norm.ppf) for each probability
    # loc is the mean (our margin m) and scale is the standard deviation (sigma_p)
    m_i_array = norm.ppf(probabilities, loc=m, scale=sigma_p)
    
    return m_i_array


def precompute_M_star(n, tau, sigma, num_patients=10000, radius=30.0, a=0.90):
    """
    Simulates the unblurred (sharp edge) dose distribution for a patient cohort.
    This replaces the large DP lookup tables mentioned in the paper.
    
    Parameters:
    n, tau, sigma: Fractionation, systematic and random errors.
    num_patients : Cohort size (10000 is enough for SDE approximation).
    a            : The population coverage requirement (e.g., 90% or 0.90).
    
    Returns:
    numpy.ndarray: Array of length 'n' containing the minimum margin required 
                   for 90% of patients to receive k/n of the dose (M_star).
    """
    
    # 1. Generate CTV surface points and their outward normal vectors
    points = generate_ctv_surface_grid(radius, num_points=512)
    normals = points / radius  # Shape: (512, 3)
    
    # 2. Simulate random and systematic errors for the cohort
    sys_err = np.random.normal(0, tau, (num_patients, 3))
    rand_errs = np.random.normal(0, sigma, (num_patients, n, 3))
    total_errs = sys_err[:, None, :] + rand_errs  # Shape: (num_patients, n, 3)
    
    # 3. Calculate outward shifts for all patients, fractions, and points
    # np.einsum provides a blazingly fast tensor dot product
    shifts = np.einsum('pd, jfd -> jfp', normals, total_errs)  # Shape: (num_patients, n, 512)
    
    # 4. Sort shifts along the fraction axis to find the worst-case fractions
    sorted_shifts = np.sort(shifts, axis=1)  # Shape: (num_patients, n, 512)
    
    # 5. Find the maximum shift over all points (worst point on the CTV)
    # This represents the minimum margin required for the entire CTV to get k/n dose
    M_jk = np.max(sorted_shifts, axis=2)  # Shape: (num_patients, n)
    
    # 6. Extract the 'a' percentile (e.g., 90th) across all simulated patients
    M_star = np.percentile(M_jk, a * 100, axis=0)  # Shape: (n,)
    
    return M_star

def calculate_sde_margin(n, M_star, sigma_p, b=0.95, I=250):
    """
    Calculates the Sharp Dose Edge (SDE) margin using root finding (Equation 2).
    
    Parameters:
    n       (int)   : Number of fractions per patient.
    M_star  (array) : Precomputed unblurred margins array (length n).
    sigma_p (float) : Penumbral width standard deviation.
    b       (float) : Minimum dose requirement (e.g., 0.95).
    I       (int)   : Number of fraction-lets (default 250).
    
    Returns:
    float: The final SDE margin adjusted to the prescription dose line.
    """
    
    # Define the objective function for the root finding algorithm
    def dose_difference(m_test):
        # 1. Calculate margins for all fraction-lets using Equation (1)
        m_i = calculate_fractionlet_margins(m_test, sigma_p, I)
        
        # 2. Compare fraction-lets against the unblurred margins (M_star)
        # H acts as the Heaviside step function from the paper.
        # It creates a matrix of 1s and 0s indicating if a fraction-let 
        # contributes to a specific dose level.
        H = (m_i[:, None] >= M_star[None, :]).astype(int)
        
        # 3. Aggregate the total dose received by the target population (90%)
        calculated_dose = np.sum(H) / (n * I)
        
        # We want the calculated dose to perfectly match our requirement 'b'
        return calculated_dose - b
        
    # Find the root (the margin m_test where dose_difference is 0)
    # We search in a safe range, from 0.0 mm up to an exaggerated 20.0 mm
    m_50 = brentq(dose_difference, 0.0, 40.0)
    
    # Adjust to the prescription dose line (95% or 80%) as per Section 3.1
    if b == 0.95:
        c_sigma = 1.645
    elif b == 0.80:
        c_sigma = 0.84
    else:
        c_sigma = norm.ppf(b)
        
    m_sde = m_50 - (c_sigma * sigma_p)
    
    return m_sde

### SDE2
def inverse_logit(x):
    """
    Calculates the inverse logit function: e^x / (1 + e^x).
    It acts as a smooth transition switch between 0 and 1.
    """
    return np.exp(x) / (1 + np.exp(x))

def calculate_sde2_margin(m_a, m_s, sigma, sigma_p, beta_0, beta_1):
    """
    Calculates the composite SDE2 margin using Equation (5) from Herschtal (2012).
    
    Parameters:
    m_a     (float or numpy.ndarray): AVH margin (lower limit).
    m_s     (float or numpy.ndarray): SDE margin (upper limit).
    sigma   (float or numpy.ndarray): Random error standard deviation.
    sigma_p (float or numpy.ndarray): Penumbral width standard deviation.
    beta_0  (float)                 : Optimized parameter 0.
    beta_1  (float)                 : Optimized parameter 1.
    
    Returns:
    float or numpy.ndarray: The composite SDE2 margin.
    """
    
    # 1. Calculate the natural log of the ratio (sigma / sigma_p)
    log_ratio = np.log(sigma / sigma_p)
    
    # 2. Calculate the interpolation weight using the inverse logit function
    weight = inverse_logit(beta_0 + beta_1 * log_ratio)
    
    # 3. Calculate the composite margin interpolating between m_a and m_s
    m_c = m_a + (m_s - m_a) * weight
    
    return m_c

def optimize_sde2_parameters(m_m_values, m_a_values, m_s_values, sigma_values, sigma_p_values):
    """
    Optimizes beta_0 and beta_1 to minimize the relative squared error 
    between the SDE2 composite margin and the Monte Carlo margin (m_m).
    
    Parameters:
    m_m_values     (numpy.ndarray): Array of Monte Carlo margins (m_M).
    m_a_values     (numpy.ndarray): Array of AVH margins (m_A).
    m_s_values     (numpy.ndarray): Array of SDE margins (m_S).
    sigma_values   (numpy.ndarray): Array of sigma values.
    sigma_p_values (numpy.ndarray): Array of sigma_p values.
    
    Returns:
    tuple: The optimal parameters (beta_0, beta_1).
    """
    
    # Define the objective function to minimize (Section 2.4 of the paper)
    def objective_function(betas):
        beta_0, beta_1 = betas
        
        # Calculate composite margins for all scenarios with the current betas
        m_c = calculate_sde2_margin(m_a_values, m_s_values, sigma_values, sigma_p_values, beta_0, beta_1)
        
        # Calculate the sum of squared relative errors
        relative_errors = (m_c - m_m_values) / m_m_values
        sum_squared_errors = np.sum(relative_errors**2)
        
        return sum_squared_errors
        
    # Initial guess for beta_0 and beta_1
    initial_guess = [0.0, 1.0]
    
    # Run the optimization using the Nelder-Mead method (robust for this kind of problem)
    result = minimize(objective_function, initial_guess, method='Nelder-Mead')
    
    if result.success:
        print("Optimization successful!")
        print(f"Optimal beta_0: {result.x[0]:.4f}")
        print(f"Optimal beta_1: {result.x[1]:.4f}")
        return result.x[0], result.x[1]
    else:
        print("Optimization failed:", result.message)
        return None, None
    
import numpy as np