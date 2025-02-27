import streamlit as st
from modules.nav import Navbar, populate_ag_grid_tbl
import duckdb as dd
import pandas as pd
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

Navbar()
st.title('🌃 Assets Timeseries Data')
st.write('Assets Timeseries Data - basis for forecasting analysis')

tab_titles = ['💸 Equities Timeseries','💷 Mutual Funds NAV Timeseries',  '🗃️ Indices Timeseries', '₿ Crypto Timeseries', '🗄️ Crypto ETFs Timeseries']
tabs = st.tabs(tab_titles)

a2db = dd.connect("a2db.db", read_only=True)

# Add content to the Data Preprocessing tab
with tabs[0]:
    st.header('💸 Equities Timeseries')
    st.write('You can find the timeseries data of Indian Equities - NIFTY500 companies')
    a2equity_ts_df = pd.DataFrame(a2db.sql("select * from a2master_data_vw where asset_type='equity'").df())
    st.dataframe(a2equity_ts_df, hide_index=True, use_container_width=True)
    #populate_ag_grid_tbl(a2equity_ts_df)

# Add content to the Data Preprocessing tab
with tabs[1]:
    st.header('💷 Mutual Funds NAV Timeseries')
    st.write('You can find the timeseries data of Top Mutual Funds')
    a2mfi_ts_df = pd.DataFrame(a2db.sql("select * from a2master_data_vw where asset_type='mutual_fund'").df())
    st.dataframe(a2mfi_ts_df, hide_index=True, use_container_width=True)
    #populate_ag_grid_tbl(a2mfi_ts_df)

# Add content to the Model Evaluation tab
with tabs[2]:
    st.header('🗃️ Indices Timeseries')
    st.write('Indices Meta Data as Input to analysis')
    a2index_ts_df = pd.DataFrame(a2db.sql("select * from a2master_data_vw where asset_type='index'").df())
    st.dataframe(a2index_ts_df,hide_index=True,  use_container_width=True)
    #populate_ag_grid_tbl(a2index_ts_df)

# Add content to the Model Training tab
with tabs[3]:
    st.header('₿ Crypto Timeseries')
    st.write('Top Ten Crypto Asset Timeseries Data')
    a2crypto_ts_df = pd.DataFrame(a2db.sql("select * from a2master_data_vw where asset_type='crypto'").df())
    st.dataframe(a2crypto_ts_df, hide_index=True, use_container_width=True)
    #populate_ag_grid_tbl(a2crypto_ts_df)

# Add content to the Results Visualization tab
with tabs[4]:
    st.header('🗄️ Crypto ETF Timeseries')
    st.write('Top Five Crypto ETF Data as Input to analysis')
    a2crypto_etf_ts_df = pd.DataFrame(a2db.sql("select * from a2master_data_vw where asset_type='crypto_etf'").df())
    st.dataframe(a2crypto_etf_ts_df, hide_index=True, use_container_width=True)
    #populate_ag_grid_tbl(a2crypto_etf_ts_df)

