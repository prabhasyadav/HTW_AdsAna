import streamlit as st
from calculations import *
st.set_page_config(page_title="Result")

st.write("# Results")
tab1, tab2, tab3, tab4 = st.tabs(["Result", "Concentration Distribution", "Diagrams", "Download Result"])

with tab1:
    #need to do this for each data_file in input_data_file list
    x_data = np.array(range(1, 11))
    c_calc_lst = st.session_state['c_calc_lst1']
    q_calc_lst = st.session_state['q_calc_lst1']
    c_calc1 = adjust_values_with_fits(x_data, st.session_state['c_exp_lst1'], c_calc_lst[:10], polynomial_fit)
    q_calc1 = adjust_values_with_fits(x_data, st.session_state['q_exp_lst1'], q_calc_lst[:10], exponential_decay_fit)

    st.write(c_calc1)
    st.write(q_calc1)
    error = calculate_error_percentage(st.session_state['data_file1'], c_calc1, q_calc1) #this is calculated only for input file 1. Is it required for file 2 as well?
    st.write("\nError %: ", error)


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