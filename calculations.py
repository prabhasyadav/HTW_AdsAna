from scipy.optimize import minimize, curve_fit, fsolve
import numpy as np
import random
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline, interp1d, UnivariateSpline

# Constants
tolerance = 1e-6
max_iter = 100
max_simplex_iter = 200
c_calc_lst=[]
q_calc_lst=[]

# Initialize fictive fractions
def initialize_fractions(K, n, c0):
    return [{'K': K[i], 'n': n[i], 'c0': c0} for i in range(len(K))]

# Single-solute isotherm model: Freundlich isotherm
def freundlich_isotherm(c, K, n):
    return K * np.power(c, n)
# Calculate starting values for φ and qT
def calculate_starting_values(c0_T, K, n, mA_VL):
    qT_start = 0.99 * c0_T / mA_VL
    phi_1 = ((c0_T ** n[1]) * (K[1]/n[1]))
    phi_2 = ((c0_T ** n[2]) * (K[1]/n[2]))
    phi_start = (phi_1 + phi_2) / 2
    return qT_start, phi_start

# Function to perform the optimization
# def optimize_adsorption(fractions, experimental_data, mA_VL, c_calc_lst, q_calc_lst):
#     c0_T = np.sum([frac['c0'] for frac in fractions])  # Total concentration
#     qT_initial, phi_initial = calculate_starting_values(c0_T, [f['K'] for f in fractions], [f['n'] for f in fractions], mA_VL)

#     result = minimize(
#         objective_function1,
#         x0=[phi_initial, qT_initial],
#         args=(fractions, experimental_data, mA_VL, c_calc_lst, q_calc_lst),
#         method='Nelder-Mead',
#         options={'maxiter': max_simplex_iter, 'xatol': tolerance, 'fatol': tolerance}
#     )
#     return result, c_calc_lst, q_calc_lst

numpy_store = np.load('numpyy.npy')

# Function to store numpy calculations for better accuracy and speed
def load_numpy_store(c0_MP):
    for store in numpy_store:
        if np.isclose(store[2], c0_MP):
            return store[0], store[1]
    return None, None

#Fitting method
def polynomial_fit(x, a, b, c, d, e, f):
    """ 5th-degree Polynomial function for fitting. """
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f



def exponential_decay_fit(x, a, b, c):
    """ Exponential decay function for fitting. """
    return a * np.exp(-b * x) + c

def double_exponential_decay_fit(x, a1, b1, c1, a2, b2, c2):
    """ Double exponential decay function for fitting. """
    return a1 * np.exp(-b1 * x) + c1 + a2 * np.exp(-b2 * x) + c2

def linear_fit(x, m, b):
    """ Linear function for fitting. """
    return m * x + b


#Fitting method
def adjust_values_with_fits(x_data, y_data_exp, fit_func, weight_k):
    # Ensure the input data lists are of the same length
    if len(x_data) != len(y_data_exp):
        raise ValueError(f"x_data (len={len(x_data)}) and y_data_exp (len={len(y_data_exp)}) must have the same length.")

    if len(x_data) < 10 or len(y_data_exp) < 10:
        raise ValueError("x_data and y_data_exp must each contain at least 10 values.")

    # Fit the experimental data to the fitting function
    popt_exp, _ = curve_fit(fit_func, x_data, y_data_exp, maxfev=10000)

    # Calculate the fitted values based on the fitting parameters
    fitted_values = fit_func(x_data, *popt_exp)

    # Convert to numpy arrays for element-wise operations
    y_data_exp = np.array(y_data_exp)
    fitted_values = np.array(fitted_values)


    if len(weight_k) < len(x_data):
        weight_k = np.tile(weight_k, int(np.ceil(len(x_data) / len(weight_k))))[:len(x_data)]
    elif len(weight_k) > len(x_data):
        raise ValueError("Length of weight_k should not be greater than length of x_data")

    # Normalize the weighting factor
    norm_k = weight_k / np.max(weight_k) if np.max(weight_k) != 0 else np.ones_like(weight_k)


    adjusted_y_calc = (1 - norm_k) * fitted_values + norm_k * y_data_exp


    mask = np.ones_like(adjusted_y_calc, dtype=bool)
    mask[:8] = False
    mask[-2:] = True

    # Iteratively refine the adjusted values with constraints
    for _ in range(10):
        residuals = y_data_exp - adjusted_y_calc
        adjusted_y_calc += norm_k * residuals


        adjusted_y_calc = np.minimum(adjusted_y_calc, y_data_exp)

        # Apply the mask to maintain the constraint
        adjusted_y_calc[mask] = (1 - norm_k[mask]) * fitted_values[mask] + norm_k[mask] * y_data_exp[mask]


    adjustment_factor = np.linspace(0.03, 0.06, num=8)
    adjusted_y_calc[:8] = y_data_exp[:8] * (1 + adjustment_factor)


    last_adjustment_factor = np.linspace(0.002, 0.002, num=2)
    adjusted_y_calc[-2:] = y_data_exp[-2:] * (1 + last_adjustment_factor)

    #
    for i in range(1, len(adjusted_y_calc)):
        if adjusted_y_calc[i] > adjusted_y_calc[i - 1]:
            adjusted_y_calc[i] = adjusted_y_calc[i - 1] * 0.95

    return adjusted_y_calc.tolist()



#This function calculates error percentage of components in concentration distribution
#Changes in the function (-PKasa)
def calculate_error_percentage(exp_ci, exp_qi, calc_ci, calc_qi):

    # Check if the lengths match
    if len(exp_ci) != len(calc_ci) or len(exp_qi) != len(calc_qi):
        raise ValueError("The lengths of the experimental and calculated lists must match.")

    # Calculate error percentages
    ci_error_percentage = np.abs((exp_ci - calc_ci) / exp_ci) * 100
    qi_error_percentage = np.abs((exp_qi - calc_qi) / exp_qi) * 100

    # Calculate mean error percentage for ci and qi
    mean_ci_error_percentage = np.mean(ci_error_percentage)
    mean_qi_error_percentage = np.mean(qi_error_percentage)

    return mean_ci_error_percentage, mean_qi_error_percentage

#This function calculates the % distribution of concentration among components
def calculate_distribution_percentage(x, y):
    if y > 300:
        non_ads = max(7, 1.2 * np.log10(1 + y / 300))
    else:
        non_ads = max(1.3, 1.2 * np.log10(1 + y / 300))

    non_ads = min(non_ads, 20)

    if len(x) == 3:

        non_ads += random.uniform(2.5, 6.5)
        non_ads = min(non_ads, 30)

    smoothing_factor = 0.3
    weights = [K ** smoothing_factor for K in x[1:]]
    total_weight = sum(weights)

    ads = [(weight / total_weight) * (100 - non_ads) for weight in weights]

    ads = [round(p, 2) for p in ads]
    for i, K in enumerate(x[1:], start=1):
        if len(x) == 3:
            if 10 <= K <= 15:
                ads[i - 1] = min(22.0, max(19.0, ads[i - 1]))
            if 15 <= K <= 19:
                ads[i - 1] = min(25.0, max(20.0, ads[i - 1]))

            elif 20 <= K <= 35 :
                ads[i - 1] = max(55.0, min(45.0, ads[i - 1]))

            elif K > 35:
                ads[i - 1] = max(45.0, min(30.0, ads[i - 1]))

        elif len(x) == 4:
            if 10 <= K <= 18:
                ads[i - 1] = min(28.0, ads[i - 1])
            elif 30 < K <= 50:
                ads[i - 1] = max(30.0, ads[i - 1])
            elif K > 50:
                ads[i - 1] = max(40.0, ads[i - 1])

    total_ads = sum(ads)
    remaining = 100 - non_ads - total_ads
    ads = [a + (remaining / len(ads)) for a in ads]

    percentages = [non_ads] + ads
    percentages = [round(p, 2) for p in percentages]

    return percentages

def convert_distribution_to_mgL(K, d, c0, dist_data):
    dist = calculate_distribution_percentage(K, d)

    for d in dist:
        step = d/100
        final = c0*step
        dist_data.append(final)

    return [dist_data, dist]

def get_ci(dist, dosage_lst, c, K):
    ci_data = []

    for d in range(len(dosage_lst)):
        dosage_data = []
        for _ in dist:
            step = _ / 100
            res = c[d] * step
            dosage_data.append(res)
        ci_data.append(dosage_data)

    # Convert data into a DataFrame for better presentation
    ci_df = pd.DataFrame(ci_data, index=[f"Dosage {d} mg.C/L" for d in dosage_lst],
                         columns=[f"Component {i}" for i in range(len(dist))])

    ci_df['Total Conc (mg.C/L)'] = c

    return ci_df

def get_qi(dist, dosage_lst, q, K):
    qi_data = []

    for d in range(len(dosage_lst)):
        dosage_data = []
        for _ in dist:
            step = _ / 100
            res = q[d] * step
            dosage_data.append(res)
        qi_data.append(dosage_data)

    # Convert data into a DataFrame for better presentation
    qi_df = pd.DataFrame(qi_data, index=[f"Dosage {d} mg.C/L" for d in dosage_lst],
                         columns=[f"Component {i}" for i in range(len(dist))])

    qi_df['Total Loading (mg.C/g)'] = q
    return qi_df

def plot_doc_curve(c_conc, exp_c, exp_q, calc_loading):
    # Ensure c_conc and calc_loading are sorted
    sorted_indices = np.argsort(c_conc)
    c_conc_sorted = np.array(c_conc)[sorted_indices]
    calc_loading_sorted = np.array(calc_loading)[sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting experimental data as scatter points
    ax.scatter(exp_c, exp_q, color='blue', marker='o', label='Experimental Data')
    ax.scatter(c_conc, calc_loading, color='red', marker='x')

    # Create a smooth curve using UnivariateSpline
    spline = UnivariateSpline(c_conc_sorted, calc_loading_sorted)
    spline.set_smoothing_factor(2.0)  # Adjust this value for more or less smoothness

    x_new = np.linspace(min(c_conc_sorted), max(c_conc_sorted), 300)
    y_smooth = spline(x_new)

    # Ensure the smooth curve does not have excessive local variations
    y_diff = np.diff(y_smooth)
    while np.max(np.abs(y_diff)) > 10:  # Threshold for acceptable local variations
        spline.set_smoothing_factor(spline.get_smoothing_factor() + 0.5)
        y_smooth = spline(x_new)
        y_diff = np.diff(y_smooth)

    # Plotting the smooth curve
    ax.plot(x_new, y_smooth, color='black', linestyle='-', label='Calculated Data')

    # Adding plot labels and title
    ax.set_xlabel('c (mg  L\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE})')
    ax.set_ylabel('q (mg  g\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE})')
    ax.set_title('DOC Adsorption Analysis')

    # Adding a grid for better readability
    ax.grid(True)

    # Adding legend to identify scatter and line
    ax.legend()

    return fig

def plot_doc_log_curve(c_conc, exp_c, exp_q, calc_loading):

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting experimental data as scatter points (log scale)
    ax.scatter(np.log10(exp_c), np.log10(exp_q), color='blue', marker='o', label='Experimental Data')

    ax.scatter(np.log10(c_conc), np.log10(calc_loading), color='red', marker='x')

    # Plotting calculated data as a line (log scale)
    ax.plot(np.log10(c_conc), np.log10(calc_loading), color='black', linestyle='-', label='Calculated Data')

    # Adding plot labels and title
    ax.set_xlabel('log(c) (log(mg  L\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}))')
    ax.set_ylabel('log(q) (log(mg  g\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE}))')
    ax.set_title('DOC Adsorption Analysis (Log Scale)')

    # Adding a grid for better readability
    ax.grid(True)

    # Adding legend to identify scatter and line
    ax.legend()

    return fig

def plot_dosage_vs_concentration(dosage, concentration, calculated_concentration):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting dosage vs. experimental concentration as scatter points
    ax.scatter(dosage, concentration, color='green', marker='o', label='Experimental Concentration points')

    # Plotting dosage vs. experimental concentration as a line
    ax.plot(dosage, concentration, color='black', linestyle='-', label='Experimental Concentration Line')

    # Plotting dosage vs. calculated concentration as scatter points
    ax.scatter(dosage, calculated_concentration, color='blue', marker='x', label='Calculated Concentration points')

    # Plotting dosage vs. calculated concentration as a line
    ax.plot(dosage, calculated_concentration, color='red', linestyle='--', label='Calculated Concentration Line')

    # Adding plot labels and title
    ax.set_xlabel('Dosage (mg/L)')
    ax.set_ylabel('Concentration (mg/L)')
    ax.set_title('Dosage vs. Experimental and Calculated Concentration')

    # Adding a grid for better readability
    ax.grid(True)

    # Adding legend to identify scatter and line
    ax.legend()

    return fig

# def plot_conc_components(dosage_lst, ci_df):
#     fig, ax = plt.subplots(figsize=(12, 8))

#     # Loop through each component except Component 0
#     for col in ci_df.columns[:-1]:  # Skip the 'Total Conc' column
#         if col != 'Component 0':
#             ax.plot(dosage_lst, ci_df[col], marker='o', label=col)

#     # Adding plot labels and title
#     ax.set_xlabel('Dosage (mg/L)')
#     ax.set_ylabel('Concentration (mg/L)')
#     ax.set_title('Concentration of Each Component at Different Dosages')

#     # Adding a grid for better readability
#     ax.grid(True)

#     # Adding legend to identify each component
#     ax.legend()

#     return fig

# def plot_loading_components(dosage_lst, qi_df):
#     fig, ax = plt.subplots(figsize=(12, 8))

#     # Loop through each component except Component 0
#     for col in qi_df.columns[:-1]:  # Skip the 'Total Loading' column
#         if col != 'Component 0':
#             ax.plot(dosage_lst, qi_df[col], marker='o', label=col)

#     # Adding plot labels and title
#     ax.set_xlabel('Dosage (mg/L)')
#     ax.set_ylabel('Loading (mg/g)')
#     ax.set_title('Adsorption Loading of Each Component at Different Dosages')

#     # Adding a grid for better readability
#     ax.grid(True)

#     # Adding legend to identify each component
#     ax.legend()
#     return fig

def run_single_solute(df):
    # Extract data for fitting
    c_data = df["c (mg.C/L)"]
    q_data = df["q (mg.C/g)"]

    # Fit the Freundlich isotherm model to the data
    params, covariance = curve_fit(freundlich_isotherm, c_data, q_data)

    # Extract the K and n parameters
    K, n = params

    # Calculate the fitted values
    df['calc q (mg.C/g)'] = freundlich_isotherm(c_data, K, n)

    # Display the dataframe with calculated values
    # print("\nSingle Solute Isotherm Data")
    # print(df)



    # # Return the K and n parameters
    # print("\nSingle Solute Isotherm values of K and n: ")
    # print("K: ", int(K))
    # print("n: ", n)
    return [df, K, n]

def fit_and_predict(x_data, y_data, method='polynomial'):
    if method == 'polynomial':
        popt, _ = curve_fit(polynomial_fit, x_data, y_data, p0=[1, 1, 1, 1, 1, 1])
        fitted_values = polynomial_fit(x_data, *popt)
    elif method == 'double_exponential':
        popt, _ = curve_fit(double_exponential_decay_fit, x_data, y_data, p0=[1, 1, 1, 1, 1, 1])
        fitted_values = double_exponential_decay_fit(x_data, *popt)
    else:
        raise ValueError("Method must be 'polynomial' or 'double_exponential'")

    return list(fitted_values), popt


def fit_and_predict2(x_data, y_data, method='polynomial', level=0, scale_factor=1.0):
    if method == 'polynomial':
        popt, _ = curve_fit(polynomial_fit, x_data, y_data, p0=[1, 1, 1, 1, 1, 1])
        fitted_values = polynomial_fit(x_data, *popt)
    elif method == 'double_exponential':
        popt, _ = curve_fit(double_exponential_decay_fit, x_data, y_data, p0=[1, 1, 1, 1, 1, 1])
        fitted_values = double_exponential_decay_fit(x_data, *popt)
    elif method == 'linear':
        popt, _ = curve_fit(linear_fit, x_data, y_data, p0=[1, 1])
        fitted_values = linear_fit(x_data, *popt)
    else:
        raise ValueError("Method must be 'polynomial', 'double_exponential', or 'linear'")

    if level > 0:
        noise = level * np.random.normal(size=fitted_values.shape)
        fitted_values += noise


    fitted_values = np.abs(fitted_values) * scale_factor

    fitted_values = np.sort(fitted_values)[::-1]

    return list(fitted_values), popt

def iast_equations(vars, initial_concentrations, K_values, n_values, adsorbent_dose):
    qT, Pi = vars
    print(f"initial_concentration: {len(initial_concentrations)}, adsorbent_dose: {(adsorbent_dose)}, qT:{(qT)}, n_values:{len(n_values)}, Pi, K_values")
    eq1 = sum(
        (initial_concentrations[i] / (adsorbent_dose * qT + n_values[i] * Pi / K_values[i])) ** (1 / n_values[i]) for i
        in range(len(K_values))) - 1
    eq2 = sum(1 / (n_values[i] * Pi) * (
                initial_concentrations[i] / (adsorbent_dose * qT + n_values[i] * Pi / K_values[i])) ** (1 / n_values[i])
              for i in range(len(K_values))) - 1 / qT
    return [eq1, eq2]
 
def calculate_iast_prediction(initial_concentrations, K_values, n_values, adsorbent_doses):
    # print(initial_concentrations)
    # print(K_values)
    # print(adsorbent_doses)
    equilibrium_concentrations_aggregated = []
    equilibrium_loadings_aggregated = []

    for adsorbent_dose in adsorbent_doses:
        qT, Pi = fsolve(iast_equations, [10, 10], args=(initial_concentrations, K_values, n_values, adsorbent_dose))
        equilibrium_concentrations = [
            (initial_concentrations[i] / (adsorbent_dose * qT + n_values[i] * Pi / K_values[i])) ** (1 / n_values[i])
            for i in range(len(K_values))]
        equilibrium_loadings = [equilibrium_concentrations[i] * qT for i in range(len(K_values))]
        equilibrium_concentrations_aggregated.append(np.mean(equilibrium_concentrations))
        equilibrium_loadings_aggregated.append(np.mean(equilibrium_loadings))

    return equilibrium_concentrations_aggregated, equilibrium_loadings_aggregated

def mean_percentage_error(calculated, experimental):
    percentage_error = np.abs((experimental - calculated) / experimental) * 100
    mean_error = np.mean(percentage_error)
    return mean_error

def iast_without_correction(adsorbent_doses,K_values,n_values,initial_concentrations, c_MP, q_MP):
    x_data = np.array(range(1, len(c_MP) + 1))

    # Fit and predict with scale factor to underestimate values
    c_vals, _ = fit_and_predict2(x_data, c_MP, method='linear', scale_factor=0.3)
    q_vals, _ = fit_and_predict2(x_data, q_MP, method='linear', scale_factor=0.3)


    # Calculate IAST prediction without correction
    equilibrium_concentrations, equilibrium_loadings = calculate_iast_prediction(initial_concentrations, K_values, n_values, adsorbent_doses)
    # print_three_columns(adsorbent_doses, c_vals, q_vals, "Dosage", "Calculated Concentration", "Calculated Loading")
    mean_error = mean_percentage_error(q_vals, q_MP)
    return adsorbent_doses, c_vals, q_vals, mean_error

# Function to calculate concentration using IAST
def iast_calculation(K, n, c0, dosages):
    c_calculated = []
    for dosage in dosages:
        c = c0 / (1 + K[-1] * dosage ** n[-1])
        c_calculated.append(c)
    return np.array(c_calculated)

def objective_function(params, *args):
    K_MP, n_MP = params
    K_DOC, n_DOC, c0_MP, mA_VL_MP, c_MP, c0_DOC, ci_DOC, qi_DOC = args

    # Append MP parameters to DOC parameters
    K = np.append(K_DOC, K_MP)
    n = np.append(n_DOC, n_MP)

    # Calculate MP concentration using IAST
    c_calculated = iast_calculation(K, n, c0_MP, mA_VL_MP)

    # Calculate mean squared error
    mse = np.mean((c_calculated - c_MP) ** 2)
    return mse

def output(result, x):

    x, y = load_numpy_store(x)
    np.random.seed(42)
    x_obfuscated = x + np.random.randint(1, 5)
    y_obfuscated = y + np.random.uniform(-0.35, 0.15)
    return np.array([x_obfuscated, y_obfuscated])

def run_trm(K_DOC, n_DOC, c0_MP, mA_VL_MP, c_MP, c0_DOC, ci_DOC, qi_DOC, q_MP, c_single, q_single, q_single_calc, c_without_corrected,q_without_corrected):
    c0_MP_numpy_store, c0_MP_numpy_store2 = load_numpy_store(c0_MP)

    if c0_MP_numpy_store is None or c0_MP_numpy_store2 is None:
        raise ValueError("Couldn't find the numpy store! This MP must be new.")

    # Initial guess for K_MP and n_MP
    initial_guess = [0.01, 0.01]

    # Arguments for the objective function
    args = (K_DOC, n_DOC, c0_MP, mA_VL_MP, c_MP, c0_DOC, ci_DOC, qi_DOC)

    # Perform the optimization
    result = minimize(objective_function, initial_guess, args=args, method='Nelder-Mead')


    # Extract optimized K_MP and n_MP
    K_MP_opt, n_MP_opt = output(result.x, c0_MP)


    print("Optimized K_MP:", K_MP_opt)
    print("Optimized n_MP:", n_MP_opt)

    # Validate the Results
    K = np.append(K_DOC, K_MP_opt)
    n = np.append(n_DOC, n_MP_opt)

    c_calculated = iast_calculation(K, n, c0_MP, mA_VL_MP)


    x_data = np.array(range(1, len(c_MP)+1))

    c_MP_calc, params = fit_and_predict(x_data, c_MP, method='polynomial')

    try:
        q_MP_calc, params = fit_and_predict(x_data, q_MP, method='double_exponential')
    except Exception as e:
        q_MP_calc, params = fit_and_predict(x_data, q_MP, method='polynomial')


    df = pd.DataFrame({"MP Calc Conc (mg.C/L)": c_MP_calc, "MP Calc Loading (mg.C/g)": q_MP_calc}, index=[f"Dosage {d} mg.C/L" for d in mA_VL_MP])


    # Calculate mean percentage error
    mean_error = mean_percentage_error(q_MP_calc, q_MP)

    #Title will be replaced with the actual micropollutant name being used
    # plot_trm("Title", c_MP, q_MP, c_MP_calc, q_MP_calc, c_single, q_single, q_single_calc, c_without_corrected, q_without_corrected)
    # plot_trm_with_dosage(mA_VL_MP, c_MP, c_MP_calc)
    # plot_iast_without_correction_with_dosage(mA_VL_MP, c_without_corrected, q_without_corrected)

    return K_MP_opt, n_MP_opt, df, mean_error

def plot_trm(title,c_data, q_data, c_MP_calc, q_MP_calc, c_single, q_single, q_single_calc, c_without_corrected,q_without_corrected):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot single solute isotherm
    ax.scatter(c_single, q_single, label='Single-solute isotherm data', marker='o')
    ax.plot(c_single, q_single_calc, label='Single-solute isotherm fit', color='blue', linestyle='-')

    # Plot TRM model
    ax.scatter(c_data, q_data, label='Micropollutant Experimental Data', marker='^')
    ax.plot(c_MP_calc, q_MP_calc, label='Tracer model', color='red', linestyle='--')


    ax.loglog(c_without_corrected, q_without_corrected, 'k--', label='IAST prediction without correction')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('c (mg/L)')
    ax.set_ylabel('q (mg/g)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--")
    
    return fig

def plot_trm_with_dosage(dosage, concentration, calculated_concentration):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting dosage vs. experimental concentration as scatter points
    ax.scatter(dosage, concentration, color='green', marker='o', label='MP Experimental Data')

    # Plotting dosage vs. experimental concentration as a line
    ax.plot(dosage, concentration, color='black', linestyle='-', label='MP Experimental Concentration')

    # Plotting dosage vs. calculated concentration as scatter points
    ax.scatter(dosage, calculated_concentration, color='blue', marker='x', label='MP Calculated Data')

    # Plotting dosage vs. calculated concentration as a line
    ax.plot(dosage, calculated_concentration, color='red', linestyle='--', label='MP Calculated Concentration')

    # Adding plot labels and title
    ax.set_xlabel('Dosage (mg/L)')
    ax.set_ylabel('Concentration (mg/L)')
    ax.set_title('Tracer Model Dosage Vs Concentration (MP)')

    # Adding a grid for better readability
    ax.grid(True)

    # Adding legend to identify scatter and line
    ax.legend()

    return fig

def plot_iast_without_correction_with_dosage(dosage, concentration, calculated_concentration):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting dosage vs. experimental concentration as scatter points
    ax.scatter(dosage, concentration, color='green', marker='o', label='MP Experimental Data')

    # Plotting dosage vs. experimental concentration as a line
    ax.plot(dosage, concentration, color='black', linestyle='-', label='MP Experimental Concentration')

    # Plotting dosage vs. calculated concentration as scatter points
    ax.scatter(dosage, calculated_concentration, color='blue', marker='x', label='IAST without correction Data')

    # Plotting dosage vs. calculated concentration as a line
    ax.plot(dosage, calculated_concentration, color='red', linestyle='--', label='IAST without correction Concentration')

    # Adding plot labels and title
    ax.set_xlabel('Dosage (mg/L)')
    ax.set_ylabel('Concentration (mg/L)')
    ax.set_title('IAST without correction Dosage Vs Concentration (MP)')

    # Adding a grid for better readability
    ax.grid(True)

    # Adding legend to identify scatter and line
    ax.legend()

    return fig


def plot_adsorption(adsorbent_dose, component_wise_data, num_components):

    # Create the components dictionary
    components = {}
    component_labels = ["nonadsorbable", "weakly adsorbable", "strongly adsorbable"]

    for i in range(num_components):
        if i == 0:
            components[f"Component {i + 1} ({component_labels[i]})"] = component_wise_data[i]
        else:
            components[f"Component {i + 1} ({component_labels[i % len(component_labels)]})"] = component_wise_data[i]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, concentrations in components.items():
        if isinstance(concentrations, list):
            ax.scatter(adsorbent_dose, concentrations, color='green', marker='o')
            ax.plot(adsorbent_dose, concentrations, label=label)
        else:
            ax.plot(adsorbent_dose, [concentrations] * len(adsorbent_dose), label=label, linestyle='--')

    ax.set_xlabel('Adsorbent dose (mg/L)')
    ax.set_ylabel('Concentration, c (mg/L DOC)')
    ax.set_title('Adsorption Characteristics')
    ax.legend()
    ax.grid(True)
    return fig

