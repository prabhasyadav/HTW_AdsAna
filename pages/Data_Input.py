import streamlit as st
# from streamlit_theme import st_theme
import pandas as pd
import numpy as np
from process_data import *
from calculations import *
st.set_page_config(page_title="Data Input")

# theme = st_theme()
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

st.write("# Data Input")

tab1, tab2 = st.tabs(["Input Data", "View Current Data"])

with tab1:

    st.subheader("Experimental Data Files")
    input_file = st.file_uploader("Upload a file", accept_multiple_files = False)
    K=[]
    n=[]
    c0=0
    sac0=0
    mA_VL = []
    ci = []
    qi = [] 
    if input_file:
        try:
            mA_VL, ci, qi, K, n, c0, sac0 = read_csv_and_extract_columns(input_file)    
        except Exception as e:
            print("Error in reading file:", e)
            st.error('Invalid File Format')
    st.divider()

    col1, col2 = st.columns(2, gap="large")
    with col1:    
        st.subheader("Initial Concentrations")
        c0 = st.number_input("c0 value", value=c0)
        sac0 = st.number_input("SAC0", value=sac0)

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
            st.session_state['sac0'] = sac0

            st.session_state['c_calc_lst']=[]
            st.session_state['q_calc_lst']=[]
            
            correction_btn = st.button("Data Correction")

            st.success("Data input successfully.")
        except Exception as e:
            st.error(f"Something went wrong while loading the input data. Please try again. {e}")

    if 'dosage_lst' in st.session_state.keys():
        with st.popover("Data Correction"):
            a0, a1, r, fig = data_correction(st.session_state['q_exp_lst'], st.session_state['c_exp_lst'])
            st.markdown(f"""
                <h5>a0: {a0}</h5>
                <h5>a1: {a1}</h5>
                <h5>r: {r}</h5>
            """, unsafe_allow_html=True)
            st.pyplot(fig)
            apply_corr_btn = st.button("Apply Correction")
            if apply_corr_btn:
                corr_c_exp_lst = a0 + a1 * st.session_state['q_exp_lst']
                corr_c0 = a0 + a1 * st.session_state['sac0']
                st.session_state['corr_c_exp_lst'] = corr_c_exp_lst
                st.session_state['corr_c0'] = corr_c0



with tab2:
    if 'dosage_lst' not in st.session_state.keys():
        st.info("No input data added yet.")
    else:
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            st.subheader("Isotherm Data")
            if 'corr_c_exp_lst' in st.session_state.keys():
                iso_input_df = pd.DataFrame({"mA/VL":st.session_state['dosage_lst'], "DOC":st.session_state['c_exp_lst'], "SAC":st.session_state['q_exp_lst'], "Corrected DOC": st.session_state['corr_c_exp_lst']})
            else:
                iso_input_df = pd.DataFrame({"mA/VL":st.session_state['dosage_lst'], "DOC":st.session_state['c_exp_lst'], "SAC":st.session_state['q_exp_lst']})
            st.dataframe(iso_input_df, use_container_width=True)
        with col2:
            st.html('''<style>
                    p.value_text{
                            padding-top: 0.25rem;
                            font-size: 1.5rem;
                            color: white;
                    }
                    </style>''')
            st.markdown(f'''
                <p class="value_text">c0 Value : {round(st.session_state['c0'], 2)}</p>''', unsafe_allow_html=True)
            st.markdown(f'''
                <p class="value_text">SAC0 Value : {round(st.session_state['sac0'], 2)}</p>''', unsafe_allow_html=True)
            if 'corr_c0' in st.session_state.keys():
                st.markdown(f'''
                    <p class="value_text">Corrected c0 : {round(st.session_state['corr_c0'], 2)}</p>''', unsafe_allow_html=True)
            st.divider()
            st.subheader("Adsorption Components")
            ads_input_df = pd.DataFrame({"K": st.session_state['K'], "n":st.session_state['n']})
            st.dataframe(ads_input_df, use_container_width=True)




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
