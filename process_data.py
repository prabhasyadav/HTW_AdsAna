import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import streamlit as st

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

def data_correction(x, y):
    result = stats.linregress(x, y)
    fig, ax = plt.subplots(figsize=(10, 6))
            
    ax.plot(x, y, 'o', label='original_plot')
    ax.plot(x, result.intercept + result.slope*x, 'r', label='fitted line')
    ax.legend()
    return result.intercept, result.slope, result.rvalue, fig

def download_input_csv():
    dosage_lst = st.session_state['dosage_lst']
    c_exp_lst = st.session_state['c_exp_lst']
    q_exp_lst = st.session_state['q_exp_lst']
    k = st.session_state['K']
    n = st.session_state['n']
    c0 = st.session_state['c0']
    sac0 = st.session_state['sac0']
    input_data_df = pd.DataFrame({'mA/VL':pd.Series(dosage_lst), 'ci':pd.Series(c_exp_lst), 'qi':pd.Series(q_exp_lst), 'K':pd.Series(k), 'n':pd.Series(n), 'c0':pd.Series(c0), 'sac0':pd.Series(sac0)})
    return input_data_df.to_csv(index=False).encode('utf-8')
