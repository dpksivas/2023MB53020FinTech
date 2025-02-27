import streamlit as st
import pandas as pd
import duckdb as dd
from streamlit import expander

from modules.nav import Navbar

Navbar()

st.subheader('🛠️ Machine Learning Models 🕹️')
st.write('Analysis of Selected Assets with Machine Learning Models')

a2db = dd.connect("a2db.db", read_only=True)

tab_titles = ['🌲 Decision Tree Model', '🏝️ Random Forest Model', '🎮 XGBoost Model', '📲 ECM Forecast Model']
tabs = st.tabs(tab_titles)

# Add content to the Data Preprocessing tab
with tabs[0]:
    st.subheader('Decision Tree Model')
    st.write('Decision Tree Model')

with tabs[1]:
    st.subheader('🏝️ Random Forest Model')
    st.write('🏝️ Random Forest Model')

with tabs[2]:
    st.subheader('XGBoost Model')
    st.write('XGBoost Model')

with tabs[3]:
    st.subheader('Garch Model')
    st.write('Garch Model')

with tabs[4]:
    st.subheader('ECM Forecast Model')
    st.write('ECM Forecast Model')
