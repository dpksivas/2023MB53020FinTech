import streamlit as st
import pandas as pd
import duckdb as dd
from streamlit import expander


from modules.nav import Navbar

Navbar()

st.subheader('🏯Portfolio Analysis of Selected Assets')
st.write('Portfolio Analysis of Selected Assets')

a2db = dd.connect("a2db.db", read_only=True)

tab_titles = ['🙊 Empyrial Analysis', '🏃Momentum Analysis', '🤹 Portfolio Optimization', '⚖️ Monte Carlo Analysis' ]
tabs = st.tabs(tab_titles)

# Add content to the Data Preprocessing tab
with tabs[0]:
    st.subheader('🙊 Empyrial Analysis')
    st.write('Empyrial empowers portfolio management by bringing the best of performance and risk analysis in an easy-to-understand, flexible and powerful framework')

with tabs[1]:
    st.subheader('Momentum Analysis')
    st.write('Momentum Analysis')

with tabs[2]:
    st.subheader('Portfolio Optimization')
    st.write('Portfolio Optimization')

with tabs[3]:
    st.subheader('Monte Carlo Analysis')
    st.write('Monte Carlo Analysis')
