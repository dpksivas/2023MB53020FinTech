import streamlit as st

from modules.nav import Navbar, populate_ag_grid_tbl
import duckdb as dd
import pandas as pd
import modules.a2init_session_variables

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

Navbar()

st.title('🛕 Assets Meta Data')
st.write('It is Assets Meta Master page - where master data are read from CSV files esp Equity Assets, Crypto Assets, Equity Indices & Crypto ETFs')

tab_titles = ['💸 Equities', '💷 Mutual Funds', '🗃️ Indices', '₿ Crypto Assets', '🗄️ Crypto ETFs']
tabs = st.tabs(tab_titles)

a2db = dd.connect("a2db.db", read_only=True)

# Add content to the Data Preprocessing tab
with tabs[0]:
    st.header('💸 Equity Meta Data')
    st.write('You can find the meta data of Indian Equities - NIFTY500 companies')
    a2equity_meta_df = pd.DataFrame(a2db.sql("select * from a2equity_meta").df())
    a2equity_meta_df.rename(columns={"trendlyne_durability_score": "durability_score", "trendlyne_valuation_score": "valuation_score"}, inplace=True)
    a2equity_meta_df.rename(columns={"trendlyne_momentum_score": "momentum_score"},inplace=True)
    populate_ag_grid_tbl(a2equity_meta_df)

# Add content to the Model Evaluation tab
with tabs[1]:
    st.header('💷 Mutual Funds Meta Data')
    st.write('Top Ten Mutual funds by AUM and Top Five High Growth Mutual Funds')
    a2mfi_meta_df = pd.DataFrame(a2db.sql("select asset_code, asset_name  from a2mfi_meta where asset_code is not null").df())
    st.table(a2mfi_meta_df.set_index(a2mfi_meta_df.columns[0]))
    #populate_ag_grid_tbl(a2mfi_meta_df)

# Add content to the Model Evaluation tab
with tabs[2]:
    st.header('🗃️ Indices Meta Data')
    st.write('Indices Meta Data as Input to analysis')
    a2index_meta_df = pd.DataFrame(a2db.sql("select asset_code, asset_name  from a2index_meta").df())
    #populate_ag_grid_tbl(a2index_meta_df)
    st.table(a2index_meta_df.set_index(a2index_meta_df.columns[0]))

# Add content to the Model Training tab
with tabs[3]:
    st.header('₿ Crypto Meta Data')
    st.write('Top Ten Crypto Asset Meta Data on Market Capitalization')
    a2crypto_meta_df = pd.DataFrame(a2db.sql("select asset_code, asset_name from a2crypto_meta").df())
    st.table(a2crypto_meta_df.set_index(a2crypto_meta_df.columns[0]))

# Add content to the Results Visualization tab
with tabs[4]:
    st.header('🗄️ Crypto ETF Meta Data')
    st.write('Top Five Crypto ETF Meta Data as Input to analysis')
    a2crypto_etf_meta_df = pd.DataFrame(a2db.sql("select asset_code, asset_name  from a2crypto_etf_meta").df())
    st.table(a2crypto_etf_meta_df.set_index(a2crypto_etf_meta_df.columns[0]))
