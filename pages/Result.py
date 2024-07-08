import streamlit as st
from calculations import *
import pandas as pd
st.set_page_config(page_title="Result")

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
st.write("# Results")
st.session_state['corr_val_choice'] = False
if 'corr_c0' in st.session_state.keys():
    with st.sidebar:
        use_corr_val = st.checkbox("Use Corrected Values", value=False)
        st.session_state['corr_val_choice'] = use_corr_val

tab1, tab2, tab3, tab4 = st.tabs(["IAST Result", "Competetive Ads Result", "Diagrams", "Download Result"])

if 'dosage_lst' not in st.session_state.keys():
    st.info('No input data. Please input data to process calculations.')
else:          
    c_calc_lst = st.session_state['c_calc_lst']
    q_calc_lst = st.session_state['q_calc_lst']
    if st.session_state['corr_val_choice']:
        c_exp_lst = st.session_state['corr_c_exp_lst']
        c0 = st.session_state['corr_c0']
    else:
        c_exp_lst = st.session_state['c_exp_lst']
        c0 = st.session_state['c0']
    q_exp_lst = st.session_state['q_exp_lst']

    dosage_lst = st.session_state['dosage_lst']
    K = st.session_state['K']    
    n = st.session_state['n']
    with tab1:
        x_data = np.array(range(1, len(dosage_lst)+1))
        
        c_calc = adjust_values_with_fits(x_data, c_exp_lst, polynomial_fit, K) #calculated concentration
        q_calc = adjust_values_with_fits(x_data, q_exp_lst, exponential_decay_fit, K) #calculated adsorption

        #---------Calculated Conc and Adsorption Result
        st.subheader("Calculated Concentration and Adsorption")
        calc_df = pd.DataFrame({'Dosage':dosage_lst, 'Calculated Concentration':c_calc, 'Calculated Adsorption':q_calc})
        st.dataframe(calc_df, use_container_width=True)

        st.subheader("Concentration Distribution")
        iso_dist = []
        x = convert_distribution_to_mgL(K, dosage_lst[-1], c0, iso_dist)
        conc_dist = pd.DataFrame({"K": K, "n": n, "c0 %": x[1], "c0 (mg/L)": x[0]})
        st.dataframe(conc_dist, use_container_width=True)

        #----------Error Percentage------------------
        error = calculate_error_percentage(c_exp_lst, q_exp_lst, c_calc, q_calc)
        col1, col2 = st.columns([1.5,4.5])
        with col1:
            st.subheader("Error : ")
        with col2:
            st.markdown(f'''
            <h3 class=value_text>{round(error[0], 2)}%</h3>''', unsafe_allow_html=True)
            st.markdown('''
            <style>
                h3.value_text{
                    padding-top: 0.75rem;
                    font-size: 1.5rem;
                    color: white;
                }
            </style>''', unsafe_allow_html=True)
                #show ci for all components at all dosage
        st.subheader("Calculated Concentration at all dosage")
        ci = get_ci(x[-1], dosage_lst, c_calc, K)
        st.dataframe(ci, use_container_width=True)
        st.session_state['ci'] = ci

        # show qi for all components at all dosage
        st.subheader("Calculated Adsorption at all dosage")
        qi = get_qi(x[-1], dosage_lst, q_calc, K)
        st.dataframe(qi, use_container_width=True)

        

    with tab2:  
        mA_VL_mp = st.session_state['mA_VL_mp']
        c_mp = st.session_state['c_mp']
        q_mp = st.session_state['q_mp']
        
        mA_VL_ss = st.session_state['mA_VL_ss']  
        c_ss = st.session_state['c_ss']
        q_ss = st.session_state['q_ss']
        c0_mp = st.session_state['c0_mp']
        df_ss = pd.DataFrame({
            "Dosage": mA_VL_ss,
            "c": c_ss,
            "q": q_ss
        })    
        single_solute = run_single_solute(df_ss)
        df_single, K_single, n_single = single_solute
        c_single = df_single['c']
        q_single = df_single['q']
        q_single_calc = df_single['q_calculated']

        st.markdown("""<h4>Single Solute Isotherm Data</h4>""", unsafe_allow_html=True)
        st.dataframe(df_single)

        st.markdown(f"""<h5>K: {round(K_single, 4)}</h5>""", unsafe_allow_html=True)
        st.markdown(f"""<h5>n: {round(n_single, 4)}</h5>""", unsafe_allow_html=True)
        st.divider()

        K_values = K + [K_single]
        K_values[0] = 0.001
        n_values = n + [n_single]
        n_values[0] = 0.001

        initial_concentrations = [x[0][0], x[0][1], x[0][2], c0_mp]
        
        adsorbent_doses, c_without_corrected, q_without_corrected, mean_error = iast_without_correction(mA_VL_mp, K_values, n_values, initial_concentrations, c_mp, q_mp)
        iast_wo_corr_df = pd.DataFrame({'Dosage':adsorbent_doses, 'Calc Concentration': c_without_corrected, 'Calc Loading': q_without_corrected})
        st.session_state['iast_wo_corr_df'] = iast_wo_corr_df
        st.markdown("""<h4>IAST Without Correction</h4>""", unsafe_allow_html=True)
        st.dataframe(iast_wo_corr_df)
        st.markdown(f"""<h5>Mean Error: {round(mean_error, 2)}%</h5>""", unsafe_allow_html=True)

        st.divider()

        K_MP_opt, n_MP_opt, trm_df, trm_mean_error = run_trm(K, n, c0_mp, mA_VL_mp, c_mp, c0, ci, qi, q_mp, c_single, q_single, q_single_calc, c_without_corrected, q_without_corrected)
        st.session_state['trm_df'] = trm_df
        st.markdown("""<h4>TRM Model</h4>""", unsafe_allow_html=True)
        st.markdown(f"""<h5>Optimized K (Micro-Pollutant): {round(K_MP_opt, 4)}</h5>""", unsafe_allow_html=True)
        st.markdown(f"""<h5>Optimized n (Micro-Pollutant): {round(n_MP_opt, 4)}</h5>""", unsafe_allow_html=True)
        st.dataframe(trm_df)
        st.markdown(f"""<h5>Mean Error: {round(trm_mean_error,2)}%</h5>""", unsafe_allow_html=True)

    with tab3:
        trm_df = st.session_state['trm_df']
        c_mp_calc = trm_df['MP Calculated Concentration']
        q_mp_calc = trm_df['MP Calculated Loading']
        with st.expander("Doc Curve Plot"):
            doc_curve_fig = plot_doc_curve(c_calc, c_exp_lst, q_exp_lst, q_calc)
            st.pyplot(doc_curve_fig)

        with st.expander("Doc Log Curve Plot "):
            doc_log_curve_fig = plot_doc_log_curve(c_calc, c_exp_lst, q_exp_lst, q_calc)
            st.pyplot(doc_log_curve_fig)

        with st.expander("Dosage vs Concentration"):
            dos_con_fig = plot_dosage_vs_concentration(dosage_lst, c_exp_lst, c_calc)
            st.pyplot(dos_con_fig)

        with st.expander("Adsorption Plot"):
            component_wise_data = []
            ci = st.session_state['ci']
            for i in range(len(K)):
                if i==0:
                    component = ci[f'Component 0'].iloc[0]
                    component_wise_data.append(component)
                else:
                    component = ci[f'Component {i}'].tolist()
                    component_wise_data.append(component)
        
            adsorption_plot_fig = plot_adsorption(dosage_lst, component_wise_data, len(K))
            st.pyplot(adsorption_plot_fig)

        with st.expander("TRM Plot"):
            
            trm_plot_fig = plot_trm("TRM Model", c_mp, q_mp, c_mp_calc, q_mp_calc, c_single, q_single, q_single_calc, c_without_corrected, q_without_corrected)
            st.pyplot(trm_plot_fig)


        with st.expander("TRM Plot with Dosage"):
            trm_dos_plot = plot_trm_with_dosage(mA_VL_mp, c_mp, c_mp_calc)
            st.pyplot(trm_dos_plot)


        with st.expander("IAST Plot Without Correction"):
            iast_dos_plot = plot_iast_without_correction_with_dosage(mA_VL_mp, c_without_corrected, q_without_corrected) 
            st.pyplot(iast_dos_plot)

        # with st.expander("Component Concentration at Different Dosages"):
        #     ci = get_ci(x[-1], dosage_lst, c_calc, K)
        #     conc_comp_fig = plot_conc_components(dosage_lst, ci)
        #     st.pyplot(conc_comp_fig)

        # with st.expander("Component Adsorption Loading at Different Dosages"):
        #     qi = get_qi(x[-1], dosage_lst, q_calc, K)
        #     loading_comp_fig = plot_loading_components(dosage_lst, qi)
        #     st.pyplot(loading_comp_fig)



