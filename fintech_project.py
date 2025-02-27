import pandas as pd
import duckdb as dd
from modules.nav import Navbar
import streamlit as st
from modules.a2init_session_variables import *
import modules.a2about_html_content as a2about

st.subheader("Comparative Forecasting of Crypto & Equity Assets")
st.text("MBA FinTech Project - by Sivasankaran K(2023MB53020)")
Navbar()

a2db = dd.connect("a2db.db", read_only=True)

a2init_session_variables()

st.html(a2about.a2about_html_str)
