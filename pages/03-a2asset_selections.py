import string

import pandas as pd
from pandas import DataFrame
from streamlit import expander
from modules.nav import *
from modules.a2fin_packages import *

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from modules.a2html_strings import *
from modules.a2stat_functions import *
from modules.a2init_session_variables import *
from statsmodels.tsa.holtwinters import ExponentialSmoothing

Navbar()
a2init_session_variables()

st.subheader('🛣️ Select your Primary Equity Asset for analysis against Basket of Other Assets')
st.write('The Selection will give you fair idea of Forecasting performance of your selected assets against the basket of other assets ')

a2db = dd.connect("a2db.db", read_only=True)

st.markdown("""<style> span[data-baseweb="tag"] {background-color: green !important;}</style>""", unsafe_allow_html=True)

# Fetch Equity Assets
a2data: DataFrame = pd.DataFrame(a2db.sql("SELECT asset_code FROM a2equity_meta").df())
a2data_lst = a2data["asset_code"].tolist()
a2data_lst.insert(0, "<Select>")

a2equity_primary_df = pd.DataFrame(a2data_lst, columns=['asset_code'])
a2equity_df = a2equity_primary_df.copy()

a2data: DataFrame = pd.DataFrame(a2db.sql("SELECT asset_code FROM a2crypto_meta").df())
a2data_lst = a2data["asset_code"].tolist()
a2data_lst.insert(0, "<Select>")

a2crypto_primary_df = pd.DataFrame(a2data_lst, columns=['asset_code'])
a2crypto_df = a2crypto_primary_df.copy()

a2data: DataFrame = pd.DataFrame(a2db.sql("SELECT asset_code FROM a2crypto_etf_meta").df())
a2data_lst = a2data["asset_code"].tolist()
a2data_lst.insert(0, "<Select>")
a2crypto_etf_df = pd.DataFrame(a2data_lst, columns=['asset_code'])

a2data: DataFrame = pd.DataFrame(a2db.sql("SELECT asset_code FROM a2index_meta").df())
a2data_lst = a2data["asset_code"].tolist()
a2data_lst.insert(0, "<Select>")
a2index_df = pd.DataFrame(a2data_lst, columns=['asset_code'])

a2data: DataFrame = pd.DataFrame(a2db.sql("SELECT asset_name || '~' || asset_code asset_code FROM a2mfi_meta where asset_code not null").df())
a2data_lst = a2data["asset_code"].tolist()
a2data_lst.insert(0, "<Select>")
a2mfi_df = pd.DataFrame(a2data_lst, columns=['asset_code'])

#a2eq_cry_options = ["Equity_Assets", "Crypto_Assets"]
st.session_state.a2primary_asset_selected_tmp = []

with expander("Primary Asset Selection", expanded=True):

    a2options1 = ["Equity_Assets", "Crypto_Assets"]
    # Main logic with improved readability
    st.session_state.a2primary_asset_selected = st.radio(
        "Primary Asset",
        a2options1,
        horizontal=True,
        index=a2options1.index(st.session_state.a2primary_asset_selected_tmp[0]) if st.session_state.a2primary_asset_selected_tmp else 0)

    st.session_state.a2primary_asset_selected_tmp = [st.session_state.a2primary_asset_selected]

    a2tmp_col1, a2tmp_col2 = st.columns(2)
    default_index = 0
    if st.session_state.a2primary_asset_selected == "Equity_Assets":
        with a2tmp_col1:
            a2options = a2equity_primary_df["asset_code"]
            a2options = a2options.tolist()

            # Set default selection from session state or use the first option
            if st.session_state.a2primary_asset_tmp_e:
                default_index_t1 = a2options.index(st.session_state.a2primary_asset_tmp_e[0])
            else:
                default_index_t1 = 0  # Default to the first option if no selection exists

            st.session_state.a2primary_asset = st.selectbox('Please Select Your Primary Equity Asset for Analysis', a2options,
                                                            index=default_index_t1)
            st.session_state.a2primary_asset_tmp_e = [st.session_state.a2primary_asset]
            st.session_state.a2primary_asset_tmp_c = []
    else:
        with a2tmp_col1:
            a2options = a2crypto_primary_df["asset_code"]
            a2options = a2options.tolist()

            # Set default selection from session state or use the first option
            if st.session_state.a2primary_asset_tmp_c:
                default_index_t2 = a2options.index(st.session_state.a2primary_asset_tmp_c[0])
            else:
                default_index_t2 = 0  # Default to the first option if no selection exists

            st.session_state.a2primary_asset = st.selectbox('Please Select Your Primary Equity Asset for Analysis', a2options,
                                                            index=default_index_t2)
            st.session_state.a2primary_asset_tmp_c = [st.session_state.a2primary_asset]
            st.session_state.a2primary_asset_tmp_e = []

st.info("Please select other assets for analysing against the primary asset selected")
a2equity_column, a2mfi_column = st.columns(2)
with a2equity_column:
    with expander("Equity Assets", expanded=True):
        # Set default selection from session state or use the first option
        if st.session_state.a2equity_list_tmp:
            default_list_e = st.session_state.a2equity_list_tmp
        else:
            default_list_e = []

        st.session_state.a2equity_list = st.multiselect("Select #3 Assets", a2equity_df, default=default_list_e, key="a2equity_sel")

        if len(st.session_state.a2equity_list) > 3:
            st.error("Please select only 3 equity assets for analysis")

        st.session_state.a2equity_list_tmp = st.session_state.a2equity_list

with a2mfi_column:
    if st.session_state.a2mfi_list_tmp:
        default_list_mfi = st.session_state.a2mfi_list
    else:
        default_list_mfi = None  # Default to the first option if no selection exists

    with expander("MFI Assets", expanded=True):
        st.session_state.a2mfi_list = st.multiselect("Select #3 Assets", a2mfi_df, default=default_list_mfi, key="a2mfi_sel")
        if len(st.session_state.a2mfi_list) > 3:
            st.error("Please select only 3 Mutual Fund assets for analysis")
        st.session_state.a2mfi_list_tmp = st.session_state.a2mfi_list

a2crypto_column, a2crypto_etf_column, a2index_column = st.columns(3)
with a2crypto_column:
    if st.session_state.a2crypto_list_tmp:
        default_list_crypto = st.session_state.a2crypto_list_tmp
    else:
        default_list_crypto = None  # Default to the first option if no selection exists

    with expander("Crypto Assets", expanded=True):
        st.session_state.a2crypto_list = st.multiselect("Select #3 Crypto Assets", a2crypto_df, default=default_list_crypto, key="a2crypto_sel")
        if len(st.session_state.a2crypto_list) > 3:
            st.error("Please select only 3 Crypto assets for analysis")
        st.session_state.a2crypto_list_tmp = st.session_state.a2crypto_list

with a2crypto_etf_column:
    if st.session_state.a2crypto_etf_list_tmp:
        default_list_crypto_etf = st.session_state.a2crypto_etf_list_tmp
    else:
        default_list_crypto_etf = None  # Default to the first option if no selection exists

    with expander("Crypto ETF Assets", expanded=True):
        st.session_state.a2crypto_etf_list = st.multiselect("Select #3 Assets", a2crypto_etf_df, default=default_list_crypto_etf, key="a2crypto_etf_sel")
        if len(st.session_state.a2crypto_etf_list) > 3:
            st.error("Please select only 3 Crypto ETFs for analysis")
        st.session_state.a2crypto_etf_list_tmp = st.session_state.a2crypto_etf_list

with a2index_column:
    if st.session_state.a2index_list_tmp:
        default_list_index = st.session_state.a2index_list_tmp
    else:
        default_list_index = None  # Default to the first option if no selection exists

    with expander("Indices", expanded=True):
        st.session_state.a2index_list = st.multiselect("Select #3 Indices", a2index_df, default=default_list_index, key="a2index_sel")
        if len(st.session_state.a2index_list) > 3:
            st.error("Please select only 3 Indices for analysis")
        st.session_state.a2index_list_tmp = st.session_state.a2index_list

st.info(" ")
st.warning("Selected Asset for Analysis - {0}".format(st.session_state.a2primary_asset))
st.session_state.a2mfi_list_codes = [item.split('~')[1] for item in st.session_state.a2mfi_list]
st.session_state.a2selected_assets_list = []
st.session_state.a2selected_assets_list.insert(0, st.session_state.a2primary_asset)
st.session_state.a2selected_assets_list += (st.session_state.a2equity_list
                                            + st.session_state.a2mfi_list_codes
                                            + st.session_state.a2index_list + st.session_state.a2crypto_list + st.session_state.a2crypto_etf_list)
#to remove duplicates
st.session_state.a2selected_assets_list = list(set(st.session_state.a2selected_assets_list))
st.session_state.a2selected_assets_df = a2asset_code_names(st.session_state.a2selected_assets_list)

with expander("The Below assets are selected to analyze against Primary Asset - {0}".format(st.session_state.a2primary_asset),expanded=True):
    st.table(st.session_state.a2selected_assets_df)
    #populate_ag_grid_tbl(st.session_state.a2selected_assets_df)

with expander("Please refer the Primary Asset details below", expanded=True):
    if st.session_state.a2primary_asset_selected =="Equity_Assets" and st.session_state.a2primary_asset != "<Select>":
        a2primary_asset_meta_df = a2db.sql("select * from a2equity_meta where asset_code = '{0}'".format(st.session_state.a2primary_asset)).df()
        display_meta_data("Equity_Asset", st.session_state.a2primary_asset)
    if st.session_state.a2primary_asset_selected == "Crypto_Assets" and st.session_state.a2primary_asset != "<Select>":
        a2primary_asset_meta_df = a2db.sql("select * from a2crypto_meta where asset_code = '{0}'".format(st.session_state.a2primary_asset)).df()
        display_meta_data("Crypto_Asset", st.session_state.a2primary_asset)

# initializing the ts data for the session
# get timeseries data of the selected assets
tmp_str = st.session_state.a2primary_asset
tmp_lst1 = st.session_state.a2selected_assets_list
if not tmp_str == '<Select>':
    st.session_state.a2df_sel_equity_ts = a2selected_assets_df_ts(tmp_lst1,tmp_str, 'equity')
    st.session_state.a2df_sel_equity_ts_pivot = st.session_state.a2df_sel_equity_ts.pivot(index='date', columns='asset_code', values='close')

    st.session_state.a2df_sel_mfi_ts = a2selected_assets_df_ts(tmp_lst1,tmp_str, 'mutual_fund')
    st.session_state.a2df_sel_mfi_ts_pivot = st.session_state.a2df_sel_mfi_ts.pivot(index='date', columns='asset_code', values='close')

    st.session_state.a2df_sel_crypto_ts = a2selected_assets_df_ts(tmp_lst1,tmp_str, 'crypto')
    st.session_state.a2df_sel_crypto_ts_pivot = st.session_state.a2df_sel_crypto_ts.pivot(index='date', columns='asset_code', values='close')

    st.session_state.a2df_sel_crypto_etf_ts = a2selected_assets_df_ts(tmp_lst1,tmp_str, 'crypto_etf')
    st.session_state.a2df_sel_crypto_etf_ts_pivot = st.session_state.a2df_sel_crypto_etf_ts.pivot(index='date', columns='asset_code', values='close')

    st.session_state.a2df_sel_index_ts = a2selected_assets_df_ts(tmp_lst1,tmp_str, 'index')
    st.session_state.a2df_sel_index_ts_pivot = st.session_state.a2df_sel_index_ts.pivot(index='date', columns='asset_code', values='close')

    st.session_state.a2df_sel_primary_ts = a2selected_assets_df_ts(tmp_lst1,tmp_str, 'primary')
else:
    st.error("You need to select at least One Primary Asset to Proceed")

st.session_state.a2chart_equity_ts_close = a2plot_asset_prices(st.session_state.a2df_sel_equity_ts,'Equity')
st.session_state.a2chart_crypto_ts_close = a2plot_asset_prices(st.session_state.a2df_sel_crypto_ts,'Crypto')
st.session_state.a2chart_crypto_etf_ts_close = a2plot_asset_prices(st.session_state.a2df_sel_crypto_etf_ts,'Crypto ETF')
st.session_state.a2chart_mfi_ts_close = a2plot_asset_prices(st.session_state.a2df_sel_mfi_ts,'Mutual Fund')
st.session_state.a2chart_index_ts_close = a2plot_asset_prices(st.session_state.a2df_sel_index_ts,'Index')

ma_types = ['sma', 'cma', 'ema', 'ewma']

# List of datasets to iterate through
a2datasets = [
    st.session_state.a2df_sel_equity_ts,
    st.session_state.a2df_sel_mfi_ts,
    st.session_state.a2df_sel_crypto_ts,
    st.session_state.a2df_sel_crypto_etf_ts,
    st.session_state.a2df_sel_index_ts
]

tmp_df_eq = pd.DataFrame()
tmp_df_mfi = pd.DataFrame()
tmp_df_crypto = pd.DataFrame()
tmp_df_crypto_etf = pd.DataFrame()
tmp_df_index = pd.DataFrame()
tmp_df_1 = pd.DataFrame()

a2datasets1 = [
    ('eq', st.session_state.a2df_sel_equity_ts),
    ('mfi', st.session_state.a2df_sel_mfi_ts),
    ('crypto',st.session_state.a2df_sel_crypto_ts),
    ('crypto_etf',st.session_state.a2df_sel_crypto_etf_ts),
    ('index', st.session_state.a2df_sel_index_ts),
    ('primary', st.session_state.a2df_sel_primary_ts)
]

for asset_name, df in a2datasets1:
    if len(df) == 0: continue
    if asset_name == 'eq':
        tmp_df_eq = df.query("asset_code!= '{0}'".format(st.session_state.a2primary_asset))

    if asset_name == 'mfi':
        tmp_df_mfi = df.query("asset_code!= '{0}'".format(st.session_state.a2primary_asset))

    if asset_name == 'crypto':
        tmp_df_crypto = df.query("asset_code!= '{0}'".format(st.session_state.a2primary_asset))

    if asset_name == 'crypto_etf':
        tmp_df_crypto_etf = df.query("asset_code!= '{0}'".format(st.session_state.a2primary_asset))

    if asset_name == 'index':
        tmp_df_index = df.query("asset_code!= '{0}'".format(st.session_state.a2primary_asset))

tmp_df_primary = st.session_state.a2df_sel_primary_ts.copy()

st.session_state.a2merged_df = pd.concat([tmp_df_primary, tmp_df_eq, tmp_df_mfi,tmp_df_crypto,tmp_df_crypto_etf,tmp_df_index ])

st.session_state.a2df_sel_primary_ts_ses, st.session_state.a2df_sel_primary_ts_ses_metric,st.session_state.a2df_sel_primary_ts_ses_3alphas = a2train_and_forecast_ses(tmp_df_primary, 5)
st.session_state.a2df_sel_equity_ts_ses, st.session_state.a2df_sel_equity_ts_ses_metric,st.session_state.a2df_sel_equity_ts_ses_3alphas = a2train_and_forecast_ses(tmp_df_eq, 5)
st.session_state.a2df_sel_mfi_ts_ses,st.session_state.a2df_sel_mfi_ts_ses_metric,st.session_state.a2df_sel_mfi_ts_ses_3alphas = a2train_and_forecast_ses(tmp_df_mfi, 5)
st.session_state.a2df_sel_crypto_ts_ses, st.session_state.a2df_sel_crypto_ts_ses_metric, st.session_state.a2df_sel_crypto_ts_ses_3alphas = a2train_and_forecast_ses(tmp_df_crypto, 5)
st.session_state.a2df_sel_crypto_etf_ts_ses, st.session_state.a2df_sel_crypto_etf_ts_ses_metric,st.session_state.a2df_sel_crypto_etf_ts_ses_3alphas = a2train_and_forecast_ses(tmp_df_crypto_etf, 5)
st.session_state.a2df_sel_index_ts_ses, st.session_state.a2df_sel_index_ts_ses_metric, st.session_state.a2df_sel_index_ts_ses_3alphas = a2train_and_forecast_ses(tmp_df_index, 5)

tmp_df_eq = st.session_state.a2df_sel_equity_ts_ses.copy()
tmp_df_mfi = st.session_state.a2df_sel_mfi_ts_ses.copy()
tmp_df_crypto = st.session_state.a2df_sel_crypto_ts_ses.copy()
tmp_df_crypto_etf = st.session_state.a2df_sel_crypto_etf_ts_ses.copy()
tmp_df_index = st.session_state.a2df_sel_index_ts_ses.copy()
tmp_df_primary = st.session_state.a2df_sel_primary_ts_ses.copy()

st.session_state.a2df_sel_primary_ts_des, st.session_state.a2df_sel_primary_ts_des_metric,st.session_state.a2df_sel_primary_ts_des_3betas = a2train_and_forecast_des_short_term(tmp_df_primary, 5)
st.session_state.a2df_sel_equity_ts_des, st.session_state.a2df_sel_equity_ts_des_metric,st.session_state.a2df_sel_equity_ts_des_3betas = a2train_and_forecast_des_short_term(tmp_df_eq, 5)
st.session_state.a2df_sel_mfi_ts_des,st.session_state.a2df_sel_mfi_ts_des_metric,st.session_state.a2df_sel_mfi_ts_des_3betas = a2train_and_forecast_des_short_term(tmp_df_mfi,5)
st.session_state.a2df_sel_crypto_ts_des, st.session_state.a2df_sel_crypto_ts_des_metric, st.session_state.a2df_sel_crypto_ts_des_3betas = a2train_and_forecast_des_short_term(tmp_df_crypto,5)
st.session_state.a2df_sel_crypto_etf_ts_des, st.session_state.a2df_sel_crypto_etf_ts_des_metric,st.session_state.a2df_sel_crypto_etf_ts_des_3betas = a2train_and_forecast_des_short_term(tmp_df_crypto_etf, 5)
st.session_state.a2df_sel_index_ts_des, st.session_state.a2df_sel_index_ts_des_metric, st.session_state.a2df_sel_index_ts_des_3betas = a2train_and_forecast_des_short_term(tmp_df_index, 5)

tmp_df_primary = st.session_state.a2df_sel_primary_ts_des.copy()
tmp_df_eq = st.session_state.a2df_sel_equity_ts_des.copy()
tmp_df_mfi = st.session_state.a2df_sel_mfi_ts_des.copy()
tmp_df_crypto = st.session_state.a2df_sel_crypto_ts_des.copy()
tmp_df_crypto_etf = st.session_state.a2df_sel_crypto_etf_ts_des.copy()
tmp_df_index = st.session_state.a2df_sel_index_ts_des.copy()

st.session_state.a2df_sel_primary_ts_tes, st.session_state.a2df_sel_primary_ts_tes_metric,st.session_state.a2df_sel_primary_ts_tes_3gamma = a2train_and_forecast_tes_stable(tmp_df_primary, 5)
st.session_state.a2df_sel_equity_ts_tes, st.session_state.a2df_sel_equity_ts_tes_metric,st.session_state.a2df_sel_equity_ts_tes_3gamma = a2train_and_forecast_tes_stable(tmp_df_eq, 5)
st.session_state.a2df_sel_mfi_ts_tes,st.session_state.a2df_sel_mfi_ts_tes_metric,st.session_state.a2df_sel_mfi_ts_tes_3gamma = a2train_and_forecast_tes_stable(tmp_df_mfi, 5)
st.session_state.a2df_sel_crypto_ts_tes, st.session_state.a2df_sel_crypto_ts_tes_metric, st.session_state.a2df_sel_crypto_ts_tes_3gamma = a2train_and_forecast_tes_stable(tmp_df_crypto, 5)
st.session_state.a2df_sel_crypto_etf_ts_tes, st.session_state.a2df_sel_crypto_etf_ts_tes_metric,st.session_state.a2df_sel_crypto_etf_ts_tes_3gamma = a2train_and_forecast_tes_stable(tmp_df_crypto_etf)
st.session_state.a2df_sel_index_ts_tes, st.session_state.a2df_sel_index_ts_tes_metric, st.session_state.a2df_sel_index_ts_tes_3gamma = a2train_and_forecast_tes_stable(tmp_df_index,  5)

