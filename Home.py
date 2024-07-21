import streamlit as st

st.set_page_config(
    page_title="Ideal Adsorption Solution Theory (IAST) Application (Ver. Alpha)",
)

st.write("# IAST Applications")

st.markdown(
    """
        <p class='note'>Please note that this is an Alpha versiona and still under development</p>
        <h2>Current Features:</h2>
        <ul>
        <li>Data Input with File Upload and Editing</li>
        <li>Competetive Adsorption Analysis with TRM Model</li>
        <li>Result Generation</li>
        <li>Result Visualization</li>
        </ul>    
        <h2>Step-by-Step Guide:</h2>
        <h3>Data Input:</h3>
        <ul>
        <li>Go to <strong>Data Input</strong> page and then <strong>Iso Input Data</strong> tab.</li>
        <li>You can edit the preexisting input data or upload a new CSV data file.</li>
        <li>Press the <strong>Submit</strong> button to load the input data.</li>
        </ul>    
        <h3>Competetive Adsorption Analysis:</h3>
        <ul>
        <li>Go to <strong>Data Input</strong> page and then <strong>Competetive Ads Data Input</strong> tab.</li>
        <li>You can upload a new CSV data files for Micropollutant and Single solute data and edit the loaded data.</li>
        <li>Press the <strong>Submit</strong> button to load the input data.</li>
        </ul>
        <h3>View Input Data:</h3>
        <ul>
        <li>Go to <strong>View DOC Data</strong> or <strong>View Competetive Ads Data</strong> tab on <strong>Data Input</strong> page to view the current input data along.</li>
        <li>Click on the <strong>Download</strong> button to download the respective input data as a CSV file.</li>
        </ul>
        <h3>Result Generation:</h3>
        <ul>
        <li>Go to <strong>Results</strong> page to view the calculated results.</li>
        <li><strong>IAST Result</strong> tab contains calculated concentration and adsorption, concentration distribution and calculated error. Additionally, it also presents calculated concentration and adsorption of the components as different dosages</li>
        <li><strong>Competetive Ads Result</strong> tab contains the IAST calculation w.r.t to the provided Multipollutant and Single solute data and corrected data using TRM model.</li>
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
