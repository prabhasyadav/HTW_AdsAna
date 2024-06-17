import streamlit as st
# from streamlit_theme import st_theme
import pandas as pd
import numpy as np
from process_data import *
from calculations import *
st.set_page_config(page_title="Data Input")

# theme = st_theme()

st.write("# Data Input")

tab1, tab2 = st.tabs(["Input Data", "View Current Data"])

with tab1:

    st.subheader("Experimental Data Files")
    input_file = st.file_uploader("Upload a file", accept_multiple_files = False)
    K=[]
    n=[]
    c0=0
    mA_VL = []
    ci = []
    qi = [] 
    if input_file:
        try:
            mA_VL, ci, qi, K, n, c0 = read_csv_and_extract_columns(input_file)    
        except Exception as e:
            print("Error in reading file:", e)
            st.error('Invalid File Format')
    st.divider()

    col1, col2 = st.columns(2, gap="large")
    with col1:    
        st.subheader("Initial Concentrations")
        c0 = st.number_input("c0 value", value=c0)

        st.subheader("Adsorption Parameters")
        # adsorbable_num = st.selectbox("Number of Adsorption Components", [1,2,3,4,5,6])
                
        ads_df = pd.DataFrame({"K":K, "n":n}, columns=("K", "n"))
        ads_df = ads_df.astype({'K':float, 'n':float})
        ads_df = st.data_editor(ads_df,
        column_config={
            "K": st.column_config.NumberColumn("K", help="About K parameter"),
            "n": st.column_config.NumberColumn("n", help="About n parameter")},
        use_container_width=True)

        

    with col2:
        
        st.subheader("Isotherm Data")
        isotherm_df = pd.DataFrame({"mA/VL":mA_VL, "ci":ci, "qi":qi}, columns=("mA/VL", "ci", "qi"))
        isotherm_df = isotherm_df.astype({'mA/VL':float, 'ci':float, 'qi':float})
        isotherm_df = st.data_editor(isotherm_df,
        column_config={
            "mA/VL": st.column_config.NumberColumn("mA/VL", help="About mA/VL parameter"),
            "ci": st.column_config.NumberColumn("ci", help="About ci parameter"),
            "qi": st.column_config.NumberColumn("qi", help="About qi parameter")},
        use_container_width=True)
    
    st.divider()
    submit_btn = st.button("Submit")

    if submit_btn: #process input data and store to session state
        try:
            for key in st.session_state.keys(): #clear all session states
                del st.session_state[key]

            st.session_state['K'] = ads_df['K'].to_numpy()
            st.session_state['n'] = ads_df['n'].to_numpy()

            st.session_state['dosage_lst'] = isotherm_df['mA/VL'].to_numpy()
            st.session_state['c_exp_lst'] = isotherm_df['ci'].to_numpy()
            st.session_state['q_exp_lst'] = isotherm_df['qi'].to_numpy()

            st.session_state['c0'] = c0

            st.session_state['c_calc_lst']=[]
            st.session_state['q_calc_lst']=[]

            st.success("Data input successfully.")
        except Exception as e:
            st.error("Something went wrong while loading the input data. Please try again.")
        



with tab2:
    if 'dosage_lst' not in st.session_state.keys():
        st.info("No input data added yet.")
    else:
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            st.subheader("Isotherm Data")
            iso_input_df = pd.DataFrame({"mA/VL":st.session_state['dosage_lst'], "Concentration":st.session_state['c_exp_lst'], "Adsorption":st.session_state['q_exp_lst']})
            st.dataframe(iso_input_df, use_container_width=True)
        with col2:
            col2_1, col2_2 = st.columns(2, gap="small")
            with col2_1:
                st.subheader('c0 Value : ')#
            with col2_2:
                st.html('''<style>
                    h3.value_text{
                            padding-top: 0.75rem;
                            font: bold;
                            font-size: 1.5rem;
                            color: white;
                    }
                    </style>''')
                st.markdown(f'''
                    <h3 class="value_text">{st.session_state['c0']}</h3>''', unsafe_allow_html=True)
            st.divider()
            st.subheader("Adsorption Components")
            ads_input_df = pd.DataFrame({"K": st.session_state['K'], "n":st.session_state['n']})
            st.dataframe(ads_input_df, use_container_width=True)

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
