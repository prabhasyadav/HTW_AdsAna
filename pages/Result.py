import streamlit as st
from calculations import *
import pandas as pd
st.set_page_config(page_title="Result")

st.write("# Results")
tab1, tab2, tab3, tab4 = st.tabs(["Result", "Concentration Distribution", "Diagrams", "Download Result"])

if 'dosage_lst' not in st.session_state.keys():
    st.info('No input data. Please input data to process calculations.')
else:    
    with tab1:
        x_data = np.array(range(1, len(st.session_state['dosage_lst'])+1))
        c_calc_lst = st.session_state['c_calc_lst']
        q_calc_lst = st.session_state['q_calc_lst']
        c_calc = adjust_values_with_fits(x_data, st.session_state['c_exp_lst'], c_calc_lst, polynomial_fit) #calculated concentration
        q_calc = adjust_values_with_fits(x_data, st.session_state['q_exp_lst'], q_calc_lst, exponential_decay_fit) #calculated adsorption

        #---------Calculated Conc and Adsorption Result
        calc_df = pd.DataFrame({'Dosage':st.session_state['dosage_lst'], 'Calculated Concentration':c_calc, 'Calculated Adsorption':q_calc})
        st.dataframe(calc_df, use_container_width=True)

        #----------Error Percentage------------------
        error = calculate_error_percentage(st.session_state['c_exp_lst'], st.session_state['q_exp_lst'], c_calc, q_calc)
        col1, col2 = st.columns([1.5,4.5])
        with col1:
            st.subheader("Error % : ")
        with col2:
            st.markdown(f'''
            <h3 class=value_text>{error[0]}</h3>''', unsafe_allow_html=True)
            st.markdown('''
            <style>
                h3.value_text{
                    padding-top: 0.75rem;
                    font-size: 1.5rem;
                    color: white;
                }
            </style>''', unsafe_allow_html=True)
    with tab2:
        iso_dist = []
        x = convert_distribution_to_mgL(st.session_state['K'], st.session_state['dosage_lst'][-1], st.session_state['c0'], iso_dist)

        #show ci for all components at all dosage
        st.subheader("Calculated Concentration at all dosage")
        ci = get_ci(x[-1], st.session_state['dosage_lst'], c_calc, st.session_state['K'])
        st.dataframe(ci, use_container_width=True)

        # show qi for all components at all dosage
        st.subheader("Calculated Adsorption at all dosage")
        qi = get_qi(x[-1], st.session_state['dosage_lst'], q_calc, st.session_state['K'])
        st.dataframe(qi, use_container_width=True)

        st.subheader("Concentration Distribution")
        conc_dist = pd.DataFrame({"K": st.session_state['K'], "n": st.session_state['n'], "c0 %": x[1], "c0 (mg/L)": x[0]})
        st.dataframe(conc_dist, use_container_width=True)

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