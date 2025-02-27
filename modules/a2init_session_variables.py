import streamlit as st
import pandas as pd
import altair as alt


def a2init_session_variables():
    # Sample Data
    df = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'moving_avg': [10, 12, 15, 20, 25, 30, 28, 27, 29, 35],
        'asset_code': ['AAPL'] * 10
    })

    a2chart = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('moving_avg:Q', title='Moving Average'),
        color=alt.Color('asset_code:N', title='Asset'),
        tooltip=['date:T', 'asset_code:N', 'moving_avg:Q']
    ).properties(width=800, height=400)

    if "a2selected_assets_df" not in st.session_state:
        st.session_state.a2selected_assets_df = pd.DataFrame()

    # initialize session state variables
    if "a2primary_asset" not in st.session_state:
        st.session_state.a2primary_asset = []
        st.session_state.a2primary_asset_tmp = []
        st.session_state.a2primary_asset_tmp_e = []
        st.session_state.a2primary_asset_tmp_c = []

        st.session_state.a2primary_asset_selected = []
        st.session_state.a2primary_asset_selected_tmp = []
        st.session_state.a2primary_asset_selected_tmp = []

        st.session_state.a2index_list = []
        st.session_state.a2index_list_tmp = []

        st.session_state.a2crypto_etf_list = []
        st.session_state.a2crypto_etf_list_tmp = []

        st.session_state.a2crypto_list = []
        st.session_state.a2crypto_list_tmp = []

        st.session_state.a2mfi_list = []
        st.session_state.a2mfi_list_tmp = []

        st.session_state.a2equity_list = []
        st.session_state.a2equity_list_tmp = []

        st.session_state.a2selected_assets_list = []

        st.session_state.a2df_sel_equity_ts = pd.DataFrame()
        st.session_state.a2df_sel_mfi_ts = pd.DataFrame()
        st.session_state.a2df_sel_crypto_ts = pd.DataFrame()
        st.session_state.a2df_sel_crypto_etf_ts = pd.DataFrame()
        st.session_state.a2df_sel_index_ts = pd.DataFrame()
        st.session_state.a2df_sel_primary_ts = pd.DataFrame()

        st.session_state.a2chart_equity_ts_close = a2chart
        st.session_state.a2chart_crypto_ts_close = a2chart
        st.session_state.a2chart_crypto_etf_ts_close = a2chart
        st.session_state.a2chart_mfi_ts_close = a2chart
        st.session_state.a2chart_index_ts_close = a2chart

        st.session_state.a2df_sel_equity_ts_ses = pd.DataFrame()
        st.session_state.a2df_sel_mfi_ts_ses = pd.DataFrame()
        st.session_state.a2df_sel_crypto_ts_ses = pd.DataFrame()
        st.session_state.a2df_sel_crypto_etf_ts_ses = pd.DataFrame()
        st.session_state.a2df_sel_index_ts_ses = pd.DataFrame()
        st.session_state.a2df_sel_primary_ts_ses = pd.DataFrame()

        st.session_state.a2df_sel_equity_ts_ses_metric = pd.DataFrame()
        st.session_state.a2df_sel_mfi_ts_ses_metric = pd.DataFrame()
        st.session_state.a2df_sel_crypto_ts_ses_metric = pd.DataFrame()
        st.session_state.a2df_sel_crypto_etf_ts_ses_metric = pd.DataFrame()
        st.session_state.a2df_sel_index_ts_ses_metric = pd.DataFrame()
        st.session_state.a2df_sel_primary_ts_ses_metric = pd.DataFrame()

        st.session_state.a2df_sel_equity_ts_ses_3alphas = pd.DataFrame()
        st.session_state.a2df_sel_mfi_ts_ses_3alphas = pd.DataFrame()
        st.session_state.a2df_sel_crypto_ts_ses_3alphas = pd.DataFrame()
        st.session_state.a2df_sel_crypto_etf_ts_ses_3alphas = pd.DataFrame()
        st.session_state.a2df_sel_index_ts_ses_3alphas = pd.DataFrame()
        st.session_state.a2df_sel_primary_ts_ses_3alphas = pd.DataFrame()

        st.session_state.a2df_sel_equity_ts_des = pd.DataFrame()
        st.session_state.a2df_sel_mfi_ts_des = pd.DataFrame()
        st.session_state.a2df_sel_crypto_ts_des = pd.DataFrame()
        st.session_state.a2df_sel_crypto_etf_ts_des = pd.DataFrame()
        st.session_state.a2df_sel_index_ts_des = pd.DataFrame()
        st.session_state.a2df_sel_primary_ts_des = pd.DataFrame()

        st.session_state.a2df_sel_equity_ts_des_metric = pd.DataFrame()
        st.session_state.a2df_sel_mfi_ts_des_metric = pd.DataFrame()
        st.session_state.a2df_sel_crypto_ts_des_metric = pd.DataFrame()
        st.session_state.a2df_sel_crypto_etf_ts_des_metric = pd.DataFrame()
        st.session_state.a2df_sel_index_ts_des_metric = pd.DataFrame()
        st.session_state.a2df_sel_primary_ts_des_metric = pd.DataFrame()

        st.session_state.a2df_sel_equity_ts_des_3betas = pd.DataFrame()
        st.session_state.a2df_sel_mfi_ts_des_3betas = pd.DataFrame()
        st.session_state.a2df_sel_crypto_ts_des_3betas = pd.DataFrame()
        st.session_state.a2df_sel_crypto_etf_ts_des_3betas = pd.DataFrame()
        st.session_state.a2df_sel_index_ts_des_3betas = pd.DataFrame()
        st.session_state.a2df_sel_primary_ts_des_3betas = pd.DataFrame()

        st.session_state.a2df_sel_equity_ts_tes = pd.DataFrame()
        st.session_state.a2df_sel_mfi_ts_tes = pd.DataFrame()
        st.session_state.a2df_sel_crypto_ts_tes = pd.DataFrame()
        st.session_state.a2df_sel_crypto_etf_ts_tes = pd.DataFrame()
        st.session_state.a2df_sel_index_ts_tes = pd.DataFrame()
        st.session_state.a2df_sel_primary_ts_tes = pd.DataFrame()

        st.session_state.a2df_sel_equity_ts_tes_metric = pd.DataFrame()
        st.session_state.a2df_sel_mfi_ts_tes_metric = pd.DataFrame()
        st.session_state.a2df_sel_crypto_ts_tes_metric = pd.DataFrame()
        st.session_state.a2df_sel_crypto_etf_ts_tes_metric = pd.DataFrame()
        st.session_state.a2df_sel_index_ts_tes_metric = pd.DataFrame()
        st.session_state.a2df_sel_primary_ts_tes_metric = pd.DataFrame()

        st.session_state.a2df_sel_equity_ts_tes_3gammas = pd.DataFrame()
        st.session_state.a2df_sel_mfi_ts_tes_3gammas = pd.DataFrame()
        st.session_state.a2df_sel_crypto_ts_tes_3gammas = pd.DataFrame()
        st.session_state.a2df_sel_crypto_etf_ts_tes_3gammas = pd.DataFrame()
        st.session_state.a2df_sel_index_ts_tes_3gammas = pd.DataFrame()
        st.session_state.a2df_sel_primary_ts_tes_3gammas = pd.DataFrame()

        st.session_state.a2merged_df = pd.DataFrame()

    if "a2html_str_pearson" not in st.session_state:
        st.session_state.a2html_str_pearson = ' '
        st.session_state.a2html_str_eda = ' '
        st.session_state.a2html_str_adf = ' '
        st.session_state.a2html_str_covariance = ' '
        st.session_state.a2html_eigenvector_str = ' '
        st.session_state.a2html_log_trans_str = ' '

        st.session_state.a2html_sma_str = ' '
        st.session_state.a2html_cma_str = ' '
        st.session_state.a2html_ema_str = ' '
        st.session_state.a2html_str_daily_dist_percent = ' '
        st.session_state.a2html_str_se_smoothing = ' '
        st.session_state.a2html_str_de_smoothing = ' '
        st.session_state.a2html_str_outer_box = ' '
        st.session_state.a2html_str_exp_smoothing = ' '
        st.session_state.a2html_str_exp_smoothing_legends = ' '
        st.session_state.a2html_50days_ma_str = ' '

    st.session_state.a2datasets_ses = [
        ("Primary", st.session_state.a2df_sel_primary_ts_ses, st.session_state.a2df_sel_primary_ts_ses_metric,
         st.session_state.a2df_sel_primary_ts_ses_3alphas),
        ("Equities", st.session_state.a2df_sel_equity_ts_ses, st.session_state.a2df_sel_equity_ts_ses_metric,
         st.session_state.a2df_sel_equity_ts_ses_3alphas),
        ("Mutual Funds", st.session_state.a2df_sel_mfi_ts_ses, st.session_state.a2df_sel_mfi_ts_ses_metric,
         st.session_state.a2df_sel_mfi_ts_ses_3alphas),
        ("Cryptos", st.session_state.a2df_sel_crypto_ts_ses, st.session_state.a2df_sel_crypto_ts_ses_metric,
         st.session_state.a2df_sel_crypto_ts_ses_3alphas),
        ("Crypto ETFs", st.session_state.a2df_sel_crypto_etf_ts_ses, st.session_state.a2df_sel_crypto_etf_ts_ses_metric,
         st.session_state.a2df_sel_crypto_etf_ts_ses_3alphas),
        ("Index", st.session_state.a2df_sel_index_ts_ses, st.session_state.a2df_sel_index_ts_ses_metric,
         st.session_state.a2df_sel_index_ts_ses_3alphas)
    ]

    st.session_state.a2datasets_des = [
        ("Primary", st.session_state.a2df_sel_primary_ts_des, st.session_state.a2df_sel_primary_ts_des_metric,
         st.session_state.a2df_sel_primary_ts_des_3betas),
        ("Equities", st.session_state.a2df_sel_equity_ts_des, st.session_state.a2df_sel_equity_ts_des_metric,
         st.session_state.a2df_sel_equity_ts_des_3betas),
        ("Mutual Funds", st.session_state.a2df_sel_mfi_ts_des, st.session_state.a2df_sel_mfi_ts_des_metric,
         st.session_state.a2df_sel_mfi_ts_des_3betas),
        ("Cryptos", st.session_state.a2df_sel_crypto_ts_des, st.session_state.a2df_sel_crypto_ts_des_metric,
         st.session_state.a2df_sel_crypto_ts_des_3betas),
        ["Crypto ETFs", st.session_state.a2df_sel_crypto_etf_ts_des, st.session_state.a2df_sel_crypto_etf_ts_des_metric,
         st.session_state.a2df_sel_crypto_etf_ts_des_3betas],
        ("Index", st.session_state.a2df_sel_index_ts_des, st.session_state.a2df_sel_index_ts_des_metric,
         st.session_state.a2df_sel_index_ts_des_3betas)
    ]

    st.session_state.a2datasets_tes = [
        ("Primary", st.session_state.a2df_sel_primary_ts_tes, st.session_state.a2df_sel_primary_ts_tes_metric,
         st.session_state.a2df_sel_primary_ts_tes_3gammas),
        ("Equities", st.session_state.a2df_sel_equity_ts_tes, st.session_state.a2df_sel_equity_ts_tes_metric,
         st.session_state.a2df_sel_equity_ts_tes_3gammas),
        ("Mutual Funds", st.session_state.a2df_sel_mfi_ts_tes, st.session_state.a2df_sel_mfi_ts_tes_metric,
         st.session_state.a2df_sel_mfi_ts_tes_3gammas),
        ["Cryptos", st.session_state.a2df_sel_crypto_ts_tes, st.session_state.a2df_sel_crypto_ts_tes_metric,
         st.session_state.a2df_sel_crypto_ts_tes_3gammas],
        ("Crypto ETFs", st.session_state.a2df_sel_crypto_etf_ts_tes, st.session_state.a2df_sel_crypto_etf_ts_tes_metric,
         st.session_state.a2df_sel_crypto_etf_ts_tes_3gammas),
        ("Index", st.session_state.a2df_sel_index_ts_tes, st.session_state.a2df_sel_index_ts_tes_metric,
         st.session_state.a2df_sel_index_ts_tes_3gammas)
    ]
