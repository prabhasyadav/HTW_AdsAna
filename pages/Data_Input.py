import streamlit as st
# from streamlit_theme import st_theme
import pandas as pd
import numpy as np
from process_data import display_input_data
from calculations import *
st.set_page_config(page_title="Data Input")

# theme = st_theme()

st.write("# Data Input")

tab1, tab2 = st.tabs(["Input Data", "View Data"])

with tab1:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Initial Concentrations")
        iso1_c0 = st.number_input("Isotherm 1 c0")
        iso2_c0 = st.number_input("Isotherm 2 c0")

        st.subheader("Experimental Data Files")
        input_files = st.file_uploader("Upload a file", accept_multiple_files = True)

    with col2:
        st.subheader("Adsorption Parameters")
        adsorbable_num = st.selectbox("Number of Adsorption Components", [1,2,3,4,5,6])
                
        ads_df = pd.DataFrame({"K":[0], "n":[0]}, columns=("K", "n"), index=range(adsorbable_num))
        ads_df = ads_df.astype({'K':float, 'n':float})
        ads_df = st.data_editor(ads_df,
        column_config={
            "K": st.column_config.NumberColumn("K", help="About K parameter"),
            "n": st.column_config.NumberColumn("n", help="About n parameter")},
        use_container_width=True)
        
        
    st.divider()
    submit_btn = st.button("Submit")

    if submit_btn: #process input data and store to session state
        for key in st.session_state.keys(): #clear all session states
            del st.session_state[key]
        disp_col1, disp_col2 = st.columns(2, gap="large")
        input_data_dfs = []
        input_data_arrs = []
        
        for file_index, input_file in enumerate(input_files):
            index = file_index + 1
            data_arr, data_df = display_input_data(input_file)
            st.session_state[f'data_file{index}'] = data_arr
            input_data_dfs.append(data_df)
            input_data_arrs.append(data_arr)
            st.session_state[f'c_calc_lst{index}']=[]
            st.session_state[f'q_calc_lst{index}']=[]

        with disp_col1:
            st.dataframe(input_data_dfs[0], use_container_width=True)
        if len(input_data_dfs)==2:
            with disp_col2:
                st.dataframe(input_data_dfs[1], use_container_width=True)
        st.write(input_data_dfs)

        st.session_state['K'] = ads_df['K'].to_numpy()
        st.session_state['n'] = ads_df['n'].to_numpy()

        st.session_state['iso1_c0'] = iso1_c0
        st.session_state['iso2_c0'] = iso2_c0

        for index, data_arr in enumerate(input_data_arrs):
            arr_index = index + 1
            for dosage in data_arr:
                mA_VL = dosage[0]
                c_exp = dosage[1]
                q_exp = dosage[2]

                if f'c_exp_lst{arr_index}' not in st.session_state:
                    st.session_state[f'c_exp_lst{arr_index}'] = []
                if f'q_exp_lst{arr_index}' not in st.session_state:
                    st.session_state[f'q_exp_lst{arr_index}'] = []
                if f'dosage_lst{arr_index}' not in st.session_state:
                    st.session_state[f'dosage_lst{arr_index}'] = []
                st.session_state[f'c_exp_lst{arr_index}'].append(c_exp)
                st.session_state[f'q_exp_lst{arr_index}'].append(q_exp)
                st.session_state[f'dosage_lst{arr_index}'].append(mA_VL)

                fractions = initialize_fractions(st.session_state['K'], st.session_state['n'], st.session_state['iso1_c0'])
                data_point = np.array([[c_exp, q_exp]])

                optimized_params, c_calc_lst_opt, q_calc_lst_opt = optimize_adsorption(fractions, data_point, mA_VL, st.session_state[f'c_calc_lst{arr_index}'], st.session_state[f'q_calc_lst{arr_index}'])
                st.session_state[f'c_calc_lst{arr_index}']=c_calc_lst_opt
                st.session_state[f'q_calc_lst{arr_index}']=q_calc_lst_opt
            st.write(len(st.session_state[f'c_calc_lst{arr_index}']))
        
        for key in st.session_state.keys():
            st.write(key)
            st.write(st.session_state[key])
        # for dosage in experimental_data1:
        # mA_VL = dosage[0]
        # c_exp = dosage[1]
        # q_exp = dosage[2]

        # c_exp_lst1.append(c_exp)
        # q_exp_lst1.append(q_exp)
        # dosage_lst.append(mA_VL)

        # fractions = initialize_fractions(K, n, c01)
        # data_point = np.array([[c_exp, q_exp]])

        # optimized_params = optimize_adsorption(fractions, data_point, mA_VL)



with tab2:
    st.write("View Input Data, Visualization...")

st.markdown("""
<style>
	.stTabs [data-baseweb="tab-list"] {
		background-color: #527a8a;
        padding: 0px 10px 0px 10px;
        border-radius: 10px
    }

    .stTabs [data-baseweb="tab"] p{
        font-size: 1rem;
        color: white;
        font-weight: 700;
    }

	.stTabs [aria-selected="true"] p{
        font-size: 1.1rem;
        font-weight: 700;
        text-shadow: black 2px 2px 5px;
	}
</style>""", unsafe_allow_html=True
)

#     st.markdown("""
# <style>
#     [data-testid="stSidebar"]{
#         background-color: #f5f5dc;
#         color: coral;
#         margin: 0.5rem;
#         border-radius: 10px
#     }

#     [data-testid="stSidebar"] span, [data-testid="stSidebar"] label{
#         color: #082e5a;
#     }

#     h1{
#         color: #f5f5dc
#     }

# 	.stTabs [data-baseweb="tab-list"] {
# 		background-color: #527a8a;
#         padding: 0px 10px 0px 10px;
#         border-radius: 10px
#     }

#     .stTabs [data-baseweb="tab"] p{
#         font-size: 1rem;
#         color: white;
#         font-weight: 700;
#     }

# 	.stTabs [aria-selected="true"] p{
#         font-size: 1.1rem;
#         font-weight: 700;
#         text-shadow: black 2px 2px 5px;
# 	}

# </style>""", unsafe_allow_html=True)
    # st.write(theme.get("base"))
    # if theme.get("base") == "light":
    #     st.markdown(
    #         """<style>
    #             h1{
    #                 color: black
    #             }
    #         </style>""", unsafe_allow_html=True
    #     )
