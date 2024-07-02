import streamlit as st

st.set_page_config(
    page_title="Adsorption Analysis (Ver. Alpha)",
    page_icon="👋"
)

st.write("# Adsorption Analysis Application")

st.markdown(
    """
        <p class='note'>Please note that this is an Alpha versiona and still under development</p>
        <h2>Current Features:</h2>
        <ul>
        <li>Data Input with File Upload and Editing</li>
        <li>Data Correction</li>
        <li>Result Generation</li>
        <li>Result Visualization</li>
        </ul>    
        <h2>Step-by-Step Guide:</h2>
        <h3>Data Input:</h3>
        <ul>
        <li>Go to <strong>Data Input</strong> page and then <strong>Input Data</strong> tab.</li>
        <li>You can edit the preexisting input data or upload a new CSV data file.</li>
        <li>Press the <strong>Submit</strong> button to load the input data.</li>
        </ul>    
        <h3>Data Correction:</h3>
        <ul>
        <li>After loading the input data, click on <strong>Data Correction</strong> to open a pop-up with data correction parameters.</li>
        <li>To apply data correction, click on the <strong>Apply Data Correction</strong> button on the pop-up.</li>
        </ul>
        <h3>View Input Data:</h3>
        <ul>
        <li>Go to <strong>View Current Data</strong> tab on <strong>Data Input</strong> page to view the current input data along with any corrections applied.</li>
        <li>Click on <strong>Download as CSV</strong> button to download the input data as a CSV file.</li>
        </ul>
        <h3>Result Generation:</h3>
        <ul>
        <li>Go to <strong>Results</strong> page to view the calculated results.</li>
        <li><strong>Result</strong> tab contains calculated concentration and adsorption, concentration distribution and calculated error.</li>
        <li><strong>Concentration Distribution</strong> tab contains calculated concentration and adsorption of the components as different dosages.</li>
        <li>To generate results using corrected values, check the <strong>Use Corrected Values</strong> option in the sidebar.</li>
        </ul>
        <h3>Result Visualization:</h3>
        <ul>
        <li>Go to <strong>Diagrams</strong> tab on <strong>Results</strong> page to view the result plottings.</li>
        <li>Click on the expander with the plotting titles to view the respective plots.</li>
        </ul>""",
    unsafe_allow_html = True
)

st.markdown("""<style>
.note {
            color: #ff9800;
            font-weight: bold;
            font-style: italic;
        }
</style>
""", unsafe_allow_html=True)
