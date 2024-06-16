import pandas as pd
import numpy as np

def display_input_data(input_file):
    data = np.loadtxt(input_file, delimiter=',', skiprows=1)
    input_df = pd.DataFrame(data=data[0:,0:],    # values
        columns=("mA/VL", "c(DOC)", "q_exp"))
    return data, input_df