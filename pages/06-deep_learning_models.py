import streamlit as st
import pandas as pd
import duckdb as dd
from streamlit import expander

from modules.nav import Navbar

Navbar()

st.subheader('🌊 Analysis of Selected Assets with Deep Learning Models')
st.write('Analysis of Selected Assets with Deep Learning Models')

a2db = dd.connect("a2db.db", read_only=True)

tab_titles = ['💽 LSTM Model', '🤖 GRU Model', '🧬 Hybrid ARIMA-LSTM', '🌈 Temporal Fusion Transformer', '🧙‍♂️ Prophet', '⚛️ Deep Q-Networks Model','🪐 Proximal Policy Optimization' ]
tabs = st.tabs(tab_titles)

# Add content to the Data Preprocessing tab
with tabs[0]:
    st.subheader('LSTM Model Model')
    st.write('LSTM Model Model')

with tabs[1]:
    st.subheader('GRU Model')
    st.write('GRU Model')

with tabs[2]:
    st.subheader('Hybrid ARIMA-LSTM')
    st.write('Hybrid ARIMA-LSTM')

with tabs[3]:
    st.subheader('Temporal Fusion Transformer')
    st.write('Temporal Fusion Transformer')

with tabs[4]:
    st.subheader('Prophet Model')
    st.write('Prophet Model')

with tabs[5]:
    st.subheader('Deep Q-Networks Model')
    st.write('Deep Q-Networks Model')

with tabs[6]:
    st.subheader('Proximal Policy Optimization')
    st.write('Proximal Policy Optimization')
