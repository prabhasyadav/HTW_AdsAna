import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def read_csv_and_extract_columns(filename):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)

    # Extract the columns as lists
    mA_VL = df['mA/VL'].tolist()
    ci = df['ci'].tolist()
    qi = df['qi'].tolist()

    K=df['K'].tolist()
    n=df['n'].tolist()

    K=[x for x in K if not np.isnan(x)]
    n=[x for x in n if not np.isnan(x)]

    # Extract c0 as a float variable from the first row
    c0 = float(df['c0'][0])
    sac0 = float(df['sac0'][0])

    return mA_VL, ci, qi, K, n, c0, sac0

# def display_input_data(input_file):
#     data = np.loadtxt(input_file, delimiter=',', skiprows=1)
#     input_df = pd.DataFrame(data=data[0:,0:],    # values
#         columns=("mA/VL", "c(DOC)", "q_exp"))
#     return data, input_df

def data_correction(x, y):
    result = stats.linregress(x, y)
    fig, ax = plt.subplots(figsize=(10, 6))
            
    ax.plot(x, y, 'o', label='original_plot')
    ax.plot(x, result.intercept + result.slope*x, 'r', label='fitted line')
    ax.legend()
    return result.intercept, result.slope, result.rvalue, fig