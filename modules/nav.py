import streamlit as st
import pandas as pd
import duckdb as dd
import requests, urllib
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder

a2db = dd.connect("a2db.db", read_only=True)
#st.set_page_config(layout="wide")

st.set_page_config(
    page_title="Analysis of Crypto and Equity Assets",
    page_icon="🌈",   # Emoji or icon
    layout="wide",    # Makes app full width
    initial_sidebar_state="expanded"  # Sidebar opens by default
)
def Navbar():
    with st.sidebar:
        st.page_link('fintech_project.py', label='About ', icon='🔥')
        st.page_link('pages/01-a2assets_meta_data.py', label='Meta Data Details', icon='🛡️')
        st.page_link('pages/02-a2assets_timeseries_data.py', label='Assets Timeseries Data', icon='📈')
        st.page_link('pages/03-a2asset_selections.py', label='Asset Selection', icon='🛣️')
        st.page_link('pages/04-statistical_models.py', label='Statistical Models', icon='🗿')
        st.page_link('pages/05-machine_learning_models.py', label='Machine Learning Models', icon='🤖')
        st.page_link('pages/06-deep_learning_models.py', label='Deep Learning Models', icon='🐦‍🔥')
        st.page_link('pages/07-portfolio_analysis.py', label='Portfolio Analysis', icon='💰')

@st.cache_data
def display_meta_data(a2asset_type: str, asset_code: str):
    if a2asset_type == "Equity_Asset":
        a2equity_meta_df = a2db.sql("select * from a2equity_meta where asset_code='{0}'".format(asset_code)).df()
        a2equity_meta_df.rename(
            columns={"trendlyne_durability_score": "durability_score", "trendlyne_valuation_score": "valuation_score"},
            inplace=True)
        a2equity_meta_df.rename(columns={"trendlyne_momentum_score": "momentum_score"}, inplace=True)
        st.table(a2equity_meta_df.T)
    else:
        tmp_str = asset_code[:6]
        url_str = "https://finance.yahoo.com/quote/{0}/".format(tmp_str)
        st.write(url_str)
    return 1

def populate_ag_grid_tbl(input_df: pd.DataFrame):
    gb = GridOptionsBuilder.from_dataframe(input_df)

    gb.configure_default_column(
        cellStyle={'border': 'none'},
        #cellStyle={'color': 'black', 'fontWeight': 'regular', 'border': 'none'},
        headerCellStyle={'background': 'white', 'text-align': 'center'}
    )
    gb.configure_grid_options(
        rowStyle={'backgroundColor': 'white'},
        domLayout='autoHeight'
    )

    for column in input_df.columns:
        gb.configure_column(column, autoSizeColumns=True)

    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="count", editable=False, autoSizeColumns=True)
    gridOptions = gb.build()
    return AgGrid(input_df, gridOptions=gridOptions, enable_enterprise_modules=True)

