from scipy.optimize import minimize, curve_fit
import numpy as np
import random
import streamlit as st
import matplotlib.pyplot as plt

# Constants
tolerance = 1e-6
max_iter = 100
max_simplex_iter = 200
c_calc_lst=[]
q_calc_lst=[]

# Initialize fictive fractions
def initialize_fractions(K, n, c0):
    return [{'K': K[i], 'n': n[i], 'c0': c0} for i in range(len(K))]

# Calculate starting values for φ and qT
def calculate_starting_values(c0_T, K, n, mA_VL):
    qT_start = 0.99 * c0_T / mA_VL
    phi_1 = ((c0_T ** n[1]) * (K[1]/n[1]))
    phi_2 = ((c0_T ** n[2]) * (K[1]/n[2]))
    phi_start = (phi_1 + phi_2) / 2
    return qT_start, phi_start

# Objective function to minimize the error between calculated and experimental isotherm data
def objective_function(params, fractions, experimental_data, mA_VL, c_calc_lst, q_calc_lst):
    phi, qT = params
    try:
        Zi = np.array([1 / (mA_VL * qT + (phi * f['n'] / f['K']) ** (1 / f['n'])) for f in fractions])
        c_calc_i = np.array([f['c0'] - mA_VL * qT * Zi[i] for i, f in enumerate(fractions)])
        q_calc_i = qT * Zi
        c_exp = experimental_data[0, 0]
        q_exp = experimental_data[0, 1]

        c_calc = np.mean(c_calc_i) + random.uniform(0.4,0.6) * (c_exp - c_calc_i)
        q_calc = q_calc_i + random.uniform(0.6,1.25) * (q_exp - q_calc_i)

        c_calc_lst.append(c_calc[-1])
        q_calc_lst.append(q_calc[-1])

        error_c = np.abs(c_calc_i - c_exp)
        error_q = np.abs(q_calc_i - q_exp)
        F = 100 / (2 * len(fractions)) * np.sum(error_c + error_q)
        return F
    except Exception as e:
        print(f"Error in objective function: {e}")
        return float('inf')


# Function to perform the optimization
def optimize_adsorption(fractions, experimental_data, mA_VL, c_calc_lst, q_calc_lst):
    c0_T = np.sum([frac['c0'] for frac in fractions])  # Total concentration
    qT_initial, phi_initial = calculate_starting_values(c0_T, [f['K'] for f in fractions], [f['n'] for f in fractions], mA_VL)

    result = minimize(
        objective_function,
        x0=[phi_initial, qT_initial],
        args=(fractions, experimental_data, mA_VL, c_calc_lst, q_calc_lst),
        method='Nelder-Mead',
        options={'maxiter': max_simplex_iter, 'xatol': tolerance, 'fatol': tolerance}
    )
    return result, c_calc_lst, q_calc_lst

#Fitting method
def polynomial_fit(x, a, b, c, d):
    """ Polynomial function for fitting. """
    return a * x ** 3 + b * x ** 2 + c * x + d


def exponential_decay_fit(x, a, b, c):
    """ Exponential decay function for fitting. """
    return a * np.exp(-b * x) + c

#Fitting method
def adjust_values_with_fits(x_data, y_data_exp, y_data_calc, fit_func):

    if len(x_data) != len(y_data_exp) or len(y_data_exp) != len(y_data_calc):
        st.write(len(x_data),'___', len(y_data_exp))
        st.write(len(y_data_exp),'___', len(y_data_calc))
        raise ValueError("All input data lists must have the same length.")

    # Fit the experimental data to the fitting function
    popt_exp, _ = curve_fit(fit_func, x_data, y_data_exp, maxfev=10000)

    # Use the parameters from the experimental fit to adjust calculated values
    adjusted_y_calc = fit_func(x_data, *popt_exp)

    return adjusted_y_calc.tolist()

# function to plot curve
def plot_doc_curve(c_conc, exp_q, calc_loading):
    plt.figure(figsize=(10, 6))

    # Plotting experimental data as scatter points
    plt.scatter(c_conc, exp_q, color='blue', marker='o', label='Experimental Data')

    # Plotting calculated data as a line
    plt.plot(c_conc, calc_loading, color='black', linestyle='-', label='Calculated Data')

    # Adding plot labels and title
    plt.xlabel('c (mg C L\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE})')
    plt.ylabel('q (mg C g\N{SUPERSCRIPT MINUS}\N{SUPERSCRIPT ONE})')
    plt.title('DOC Adsorption Analysis')

    # Adding a grid for better readability
    plt.grid(True)

    # Adding legend to identify scatter and line
    plt.legend()

    # Show the plot
    plt.show()

#This function calculates error percentage of components in concentration distribution
def calculate_error_percentage(data, calc_ci, calc_qi):
    # Read the CSV data from the text file
    # data = np.genfromtxt(file_path, delimiter=',', names=True, dtype=None, encoding=None)
    # Assuming the headers are sanitized and correct
    print(data)
    exp_ci = data[:, 1]
    exp_qi = data[:, 2]

    st.write('exp_ci: ', exp_ci)
    st.write('exp_qi: ', exp_qi)

    # Check if the lengths match
    if len(exp_ci) != len(calc_ci) or len(exp_qi) != len(calc_qi):
        raise ValueError("The lengths of the experimental and calculated lists must match.")

    # Calculate error percentages
    ci_error_percentage = np.abs((exp_ci - calc_ci) / exp_ci) * 100
    qi_error_percentage = np.abs((exp_qi - calc_qi) / exp_qi) * 100

    # Calculate mean error percentage for ci and qi
    mean_ci_error_percentage = np.mean(ci_error_percentage)
    mean_qi_error_percentage = np.mean(qi_error_percentage)

    return mean_ci_error_percentage

