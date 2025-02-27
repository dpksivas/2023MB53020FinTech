import time
import warnings

import pandas as pd
from streamlit import expander
from modules.nav import Navbar, display_meta_data
from modules.a2html_strings import *
from modules.a2stat_functions import *
from modules.a2stat_arima_sarima_garima import *
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import sklearn.metrics

from modules.a2init_session_variables import *
Navbar()
a2init_session_variables()

warnings.filterwarnings("ignore")

st.subheader('🦩Statistical Analysis of Selected Assets')
st.write('Statistical Analysis of Selected Assets')

a2db = dd.connect("a2db.db", read_only=True)

tab_titles = ['📜 Exploratory Data Analysis', '🎢 Moving Average', '📈 Exponential Smoothing', '🪿 ARIMA Model', '🦜 SARIMA Model', '🦤 GARIMA Model']
tabs = st.tabs(tab_titles)

with tabs[0]:
    st.subheader('Exploratory Data analysis')
    st.html(st.session_state.a2html_str_eda)
    st.info("Your Primary Asset Selected is - {0}".format(st.session_state.a2primary_asset))

    tab_titles = ['🎢 Trend Analysis', '🫧 Percentage Change Trend', '🛞 log transformation', ':sagittarius: Correlations', '⛓️ Covariances', '⛄ Stationarity Analysis']
    sub_tabs1 = st.tabs(tab_titles)

    with sub_tabs1[0]:
        st.warning("Trending of Asset Prices over the time period range")
        a2tmp_col1, a2tmp_col2 = st.columns(2, border=True)
        with a2tmp_col1:
            st.info("vs - Equities - Pricing Trend Analysis (INR) - Against Primary Asset")
            if not len(st.session_state.a2df_sel_equity_ts) == 0:
                a2place_holder1 = st.empty()
                with a2place_holder1.container():
                    st.altair_chart(st.session_state.a2chart_equity_ts_close, use_container_width=True)
                    time.sleep(1)
        with a2tmp_col2:
            st.info("vs - Cryptos - Pricing Trend Analysis (INR) - Against Primary Asset")
            if not len(st.session_state.a2df_sel_crypto_ts) == 0:
                a2place_holder2 = st.empty()
                with a2place_holder2.container():
                    st.altair_chart(st.session_state.a2chart_crypto_ts_close, use_container_width=True)
                    time.sleep(1)

        a2tmp_col3, a2tmp_col4 = st.columns(2, border=True)
        with a2tmp_col3:
            st.info("vs - MFI - Pricing Trend Analysis (INR) - Against Primary Asset")
            if not len(st.session_state.a2df_sel_mfi_ts) == 0:
                a2place_holder3 = st.empty()
                with a2place_holder3.container():
                    st.altair_chart(st.session_state.a2chart_mfi_ts_close, use_container_width=True)
                    time.sleep(1)
        with a2tmp_col4:
            st.info("vs - Index - Trend Analysis - Against Primary Asset")
            if not len(st.session_state.a2df_sel_index_ts) == 0:
                a2place_holder4 = st.empty()
                with a2place_holder4.container():
                    st.altair_chart(st.session_state.a2chart_index_ts_close, use_container_width=True)
                    time.sleep(1)

        a2tmp_col5, a2tmp_col6 = st.columns(2, border=True)
        with a2tmp_col5:
            st.info("vs - Crypto ETF - Trend Analysis - Against Primary Asset")
            if not len(st.session_state.a2df_sel_crypto_etf_ts) == 0:
                a2place_holder5 = st.empty()
                with a2place_holder5.container():
                    st.altair_chart(a2plot_asset_prices(st.session_state.a2df_sel_crypto_etf_ts,'Crypto ETF'), use_container_width=True)
                    time.sleep(1)

    with sub_tabs1[1]:
        st.warning("Daily Price changing Trending of Asset Prices")
        st.info("vs - Equities - Pricing Trend Analysis (INR) - Against Primary Asset")
        st.html(st.session_state.a2html_str_daily_dist_percent)
        if not len(st.session_state.a2df_sel_equity_ts) == 0 and not st.session_state.a2df_sel_equity_ts.empty:
            a2tmp_col7, a2tmp_col8 = st.columns(2, border=True)
            with a2tmp_col7:
                a2place_holder6 = st.empty()
                with a2place_holder6.container():
                    st.plotly_chart(a2plot_with_param(st.session_state.a2df_sel_equity_ts_pivot,
                        st.session_state.a2primary_asset,'p'), use_container_width=True, key=random.randint(1, 100000))
                    time.sleep(1)
            with a2tmp_col8:
                a2place_holder8 = st.empty()
                with a2place_holder8.container():
                    st.plotly_chart(a2plot_with_param(st.session_state.a2df_sel_equity_ts_pivot,
                        st.session_state.a2primary_asset,'d'), use_container_width=True, key=random.randint(1, 100000))

        st.info("vs - Cryptos - Pricing Trend Analysis (INR) - Against Primary Asset")
        if not len(st.session_state.a2df_sel_crypto_ts) == 0 and not st.session_state.a2df_sel_crypto_ts.empty:
            a2tmp_col9, a2tmp_col10 = st.columns(2, border=True)
            with a2tmp_col9:
                st.plotly_chart(a2plot_with_param(st.session_state.a2df_sel_crypto_ts_pivot,
                                                  st.session_state.a2primary_asset,'p'), use_container_width=True, key=random.randint(1, 100000))
            with a2tmp_col10:
                st.plotly_chart(a2plot_with_param(st.session_state.a2df_sel_crypto_ts_pivot,
                                                  st.session_state.a2primary_asset,'d'), use_container_width=True, key=random.randint(1, 100000))

        st.info("vs - MFI - Pricing Trend Analysis (INR) - Against Primary Asset")
        if not len(st.session_state.a2df_sel_mfi_ts) == 0 and not st.session_state.a2df_sel_mfi_ts.empty:
            a2tmp_col11, a2tmp_col12 = st.columns(2, border=True)
            with a2tmp_col11:
                st.plotly_chart(a2plot_with_param(st.session_state.a2df_sel_mfi_ts_pivot,
                    st.session_state.a2primary_asset,'p'),use_container_width=True, key=random.randint(1, 100000))
            with a2tmp_col12:
                st.plotly_chart(a2plot_with_param(st.session_state.a2df_sel_mfi_ts_pivot,
                    st.session_state.a2primary_asset,'d'), use_container_width=True, key=random.randint(1, 100000))

        st.info("vs - Index - Trend Analysis - Against Primary Asset")
        if not len(st.session_state.a2df_sel_index_ts) == 0:
            a2tmp_col13, a2tmp_col14 = st.columns(2, border=True)
            with a2tmp_col13:
                if not len(st.session_state.a2df_sel_index_ts_pivot) == 0:
                    st.plotly_chart(a2plot_with_param(st.session_state.a2df_sel_index_ts_pivot,
                        st.session_state.a2primary_asset,'p'), use_container_width=True, key=random.randint(1, 100000))
            with a2tmp_col14:
                if not len(st.session_state.a2df_sel_index_ts_pivot) == 0:
                    st.plotly_chart(a2plot_with_param(st.session_state.a2df_sel_index_ts_pivot,
                                                  st.session_state.a2primary_asset,'d'), use_container_width=True, key=random.randint(1, 100000))

        st.info("vs - Crypto ETF - Trend Analysis - Against Primary Asset")
        if not len(st.session_state.a2df_sel_crypto_etf_ts) == 0:
            a2tmp_col15, a2tmp_col16 = st.columns(2, border=True)
            with a2tmp_col15:
                st.plotly_chart(a2plot_with_param(st.session_state.a2df_sel_crypto_etf_ts_pivot,
                                                  st.session_state.a2primary_asset,'p'), use_container_width=True, key=random.randint(1, 100000))
            with a2tmp_col16:
                st.plotly_chart(a2plot_with_param(st.session_state.a2df_sel_crypto_etf_ts_pivot,
                                                  st.session_state.a2primary_asset,'d'), use_container_width=True, key=random.randint(1, 100000))

    with sub_tabs1[2]:
        with expander("Log Transformation & Assets Analysis", expanded=False):
            st.html(st.session_state.a2html_log_trans_str)

        # Equities log transformation
        if not len(st.session_state.a2df_sel_equity_ts) == 0:
            st.write("Log Transformation of Equities")
            st.plotly_chart(a2plot_with_param(st.session_state.a2df_sel_equity_ts_pivot,
                                              st.session_state.a2primary_asset, 'log'), use_container_width=True, key=random.randint(1, 100000))
        # MFI log transformation
        if not len(st.session_state.a2df_sel_mfi_ts) == 0:
            st.write("Log Transformation of Mutual Funds")
            st.plotly_chart(a2plot_with_param(st.session_state.a2df_sel_mfi_ts_pivot,
                                              st.session_state.a2primary_asset, 'log'), use_container_width=True, key=random.randint(1, 100000))

        # Crypto Log Transformation
        if not len(st.session_state.a2df_sel_crypto_ts) == 0:
            st.write("Log Transformation of Mutual Funds")
            st.plotly_chart(a2plot_with_param(st.session_state.a2df_sel_crypto_ts_pivot,
                                              st.session_state.a2primary_asset, 'log'), use_container_width=True, key=random.randint(1, 100000))

        # Crypto ETF Log Transformation
        if not len(st.session_state.a2df_sel_crypto_etf_ts) == 0:
            st.write("Log Transformation of Mutual Funds")
            st.plotly_chart(a2plot_with_param(st.session_state.a2df_sel_crypto_etf_ts_pivot,
                                              st.session_state.a2primary_asset, 'log'), use_container_width=True, key=random.randint(1, 100000))

        # Index ETF Log Transformation
        if not len(st.session_state.a2df_sel_index_ts) == 0:
            st.write("Log Transformation of Mutual Funds")
            st.plotly_chart(a2plot_with_param(st.session_state.a2df_sel_index_ts_pivot,
                                              st.session_state.a2primary_asset, 'log'), use_container_width=True, key=random.randint(1, 100000))

    # Correlation Tabs
    with sub_tabs1[3]:
        st.html(st.session_state.a2html_str_pearson)
        a2tmp_col1, a2tmp_col2 = st.columns(2)

        # Equity Correlations
        if not len(st.session_state.a2df_sel_equity_ts) == 0:
            with a2tmp_col1:
                st.info('Equities - Pearson Correlations')
                correlation_matrix = a2process_correlations(st.session_state.a2df_sel_equity_ts_pivot, st.session_state.a2primary_asset, 'pearson')
                fig20 = plot_correlation_heatmap_px(correlation_matrix, st.session_state.a2primary_asset,'pearson', 'greens' )
                st.plotly_chart(fig20, key=random.randint(1, 100000))
            with a2tmp_col2:
                st.info('Equities - Spearman Correlations')
                correlation_matrix = a2process_correlations(st.session_state.a2df_sel_equity_ts_pivot, st.session_state.a2primary_asset, 'spearman')
                fig21 = plot_correlation_heatmap_px(correlation_matrix, st.session_state.a2primary_asset,'spearman', 'oranges' )
                st.plotly_chart(fig21, use_container_width=True, key=random.randint(1, 100000))

        # MFI Correlations
        if not len(st.session_state.a2df_sel_mfi_ts) == 0:
            with a2tmp_col1:
                st.info('Mutual Funds - Pearson Correlations')
                correlation_matrix = a2process_correlations(st.session_state.a2df_sel_mfi_ts_pivot, st.session_state.a2primary_asset, 'pearson')
                fig22 = plot_correlation_heatmap_px(correlation_matrix, st.session_state.a2primary_asset,'pearson', 'RdBu' )
                st.plotly_chart(fig22, use_container_width=True, key=random.randint(1, 100000))
            with a2tmp_col2:
                st.info('Mutual Funds - Spearman Correlations')
                correlation_matrix = a2process_correlations(st.session_state.a2df_sel_mfi_ts_pivot, st.session_state.a2primary_asset, 'spearman')
                fig23 = plot_correlation_heatmap_px(correlation_matrix, st.session_state.a2primary_asset,'spearman', 'greens' )
                st.plotly_chart(fig23, use_container_width=True, key=random.randint(1, 100000))

        # Crypto Correlations
        if not len(st.session_state.a2df_sel_crypto_ts) == 0:
            with a2tmp_col1:
                st.info('Crypto - Pearson Correlations')
                correlation_matrix = a2process_correlations(st.session_state.a2df_sel_crypto_ts_pivot, st.session_state.a2primary_asset, 'pearson')
                fig24 = plot_correlation_heatmap_px(correlation_matrix, st.session_state.a2primary_asset,'pearson', 'purples' )
                st.plotly_chart(fig24, use_container_width=True, key=random.randint(1, 100000))
            with a2tmp_col2:
                st.info('Crypto - Spearman Correlations')
                correlation_matrix = a2process_correlations(st.session_state.a2df_sel_crypto_ts_pivot, st.session_state.a2primary_asset, 'spearman')
                fig25 = plot_correlation_heatmap_px(correlation_matrix, st.session_state.a2primary_asset,'spearman', 'greys' )
                st.plotly_chart(fig25, use_container_width=True, key=random.randint(1, 100000))

        # Crypto ETF Correlations
        if not len(st.session_state.a2df_sel_crypto_etf_ts) == 0:
            with a2tmp_col1:
                st.info('Crypto ETF - Pearson Correlations')
                correlation_matrix = a2process_correlations(st.session_state.a2df_sel_crypto_etf_ts_pivot, st.session_state.a2primary_asset, 'pearson')
                fig26 = plot_correlation_heatmap_px(correlation_matrix, st.session_state.a2primary_asset,'pearson', 'greens' )
                st.plotly_chart(fig26, use_container_width=True, key=random.randint(1, 100000))
            with a2tmp_col2:
                st.info('Crypto ETF - Spearman Correlations')
                correlation_matrix = a2process_correlations(st.session_state.a2df_sel_crypto_etf_ts_pivot, st.session_state.a2primary_asset, 'spearman')
                fig27 = plot_correlation_heatmap_px(correlation_matrix, st.session_state.a2primary_asset,'spearman', 'Cividis' )
                st.plotly_chart(fig27, use_container_width=True, key=random.randint(1, 100000))

        # Index Correlations
        if not len(st.session_state.a2df_sel_index_ts) == 0:
            with a2tmp_col1:
                st.info('Indexs - Pearson Correlations')
                correlation_matrix = a2process_correlations(st.session_state.a2df_sel_index_ts_pivot, st.session_state.a2primary_asset, 'pearson')
                fig28 = plot_correlation_heatmap_px(correlation_matrix, st.session_state.a2primary_asset,'pearson', 'purples' )
                st.plotly_chart(fig28, use_container_width=True, key=random.randint(1, 100000))
            with a2tmp_col2:
                st.info('Indexs - Spearman Correlations')
                correlation_matrix = a2process_correlations(st.session_state.a2df_sel_index_ts_pivot, st.session_state.a2primary_asset, 'spearman')
                fig29 = plot_correlation_heatmap_px(correlation_matrix, st.session_state.a2primary_asset,'spearman', 'oranges' )
                st.plotly_chart(fig29, use_container_width=True, key=random.randint(1, 100000))

    with sub_tabs1[4]:
        with expander("Expand here for Understanding Covariance & Its impact with different assets", expanded=False):
            st.html(st.session_state.a2html_str_covariance)
        # Equity Covariance
        with expander("Expand here for Understanding of Eigen Vectors and its significances", expanded=False):
            st.html(st.session_state.a2html_eigenvector_str)
        if not len(st.session_state.a2df_sel_equity_ts) == 0 and not st.session_state.a2df_sel_equity_ts.empty:
            st.info('vs - Equities Covariance & Eugene Values')
            a2tmp_col1, a2tmp_col2 = st.columns(2, border=True)
            with a2tmp_col1:
                fig30 = plot_covariance_heatmap(st.session_state.a2df_sel_equity_ts_pivot, st.session_state.a2primary_asset, 'greens')
                st.plotly_chart(fig30, use_container_width=True, key=random.randint(1, 100000))
            with a2tmp_col2:
                if st.session_state.a2df_sel_equity_ts_pivot.shape[1] >= 2:
                    fig31 = eigen_decomposition_from_covariance(st.session_state.a2df_sel_equity_ts_pivot, 'purples')
                    st.plotly_chart(fig31, use_container_width=True, key=random.randint(1, 100000))

        # Crypto Covariance
        if not len(st.session_state.a2df_sel_crypto_ts) == 0:
            st.info('vs - Cryptos Covariance & Eugene Values')
            a2tmp_col1, a2tmp_col2 = st.columns(2, border=True)
            with a2tmp_col1:
                fig32 = plot_covariance_heatmap(st.session_state.a2df_sel_crypto_ts_pivot, st.session_state.a2primary_asset, 'blues')
                st.plotly_chart(fig32, use_container_width=True, key=random.randint(1, 100000))
            with a2tmp_col2:
                if st.session_state.a2df_sel_crypto_ts_pivot.shape[1] >= 2:
                    fig33 = eigen_decomposition_from_covariance(st.session_state.a2df_sel_crypto_ts_pivot, 'reds')
                    st.plotly_chart(fig33, use_container_width=True, key=random.randint(1, 100000))

        # Crypto ETF Covariance
        if not len(st.session_state.a2df_sel_crypto_etf_ts) == 0:
            st.info('vs - Crypto ETFs Covariance & Eugene Values')
            a2tmp_col1, a2tmp_col2 = st.columns(2, border=True)
            with a2tmp_col1:
                fig34 = plot_covariance_heatmap(st.session_state.a2df_sel_crypto_etf_ts_pivot, st.session_state.a2primary_asset, 'oranges')
                st.plotly_chart(fig34, use_container_width=True, key=random.randint(1, 100000))
            with a2tmp_col2:
                if st.session_state.a2df_sel_crypto_etf_ts_pivot.shape[1] >= 2:
                    fig35 = eigen_decomposition_from_covariance(st.session_state.a2df_sel_crypto_etf_ts_pivot, 'greys')
                    st.plotly_chart(fig35, use_container_width=True, key=random.randint(1, 100000))

        # Index Covariance
        if not len(st.session_state.a2df_sel_index_ts) == 0:
            st.info('vs - Indices Covariance & Eugene Values')
            a2tmp_col1, a2tmp_col2 = st.columns(2, border=True)
            with a2tmp_col1:
                fig36 = plot_covariance_heatmap(st.session_state.a2df_sel_index_ts_pivot, st.session_state.a2primary_asset, 'purples')
                st.plotly_chart(fig36)
            with a2tmp_col2:
                if st.session_state.a2df_sel_index_ts_pivot.shape[1] >= 2:
                    fig37 = eigen_decomposition_from_covariance(st.session_state.a2df_sel_index_ts_pivot, 'blues')
                    st.plotly_chart(fig37)

    with sub_tabs1[5]:
        with expander("Expand here for Understanding Stationarity of the Assets over the Timeperiod", expanded=True):
            st.html(st.session_state.a2html_str_adf)
        a2tmp_col1, a2tmp_col2 = st.columns(2, border=True)
        # Equity ADF Test
        with a2tmp_col1:
            st.write("ADF Test for Equities")
            if not len(st.session_state.a2df_sel_equity_ts) == 0:
                st.write(a2adf_test_matrix(st.session_state.a2df_sel_equity_ts_pivot))
        # MFI Adf Test
        with a2tmp_col2:
            st.write("ADF Test for Mutual Funds")
            if not len(st.session_state.a2df_sel_mfi_ts) == 0:
                st.write(a2adf_test_matrix(st.session_state.a2df_sel_mfi_ts_pivot))
        # Crypto ADF Test
        with a2tmp_col1:
            st.write("ADF Test for Crypto")
            if not len(st.session_state.a2df_sel_crypto_ts) == 0:
                st.write(a2adf_test_matrix(st.session_state.a2df_sel_crypto_ts_pivot))
        # Crypto ETF ADF Test
        with a2tmp_col2:
            st.write("ADF Test for Crypto ETF")
            if not len(st.session_state.a2df_sel_crypto_ts) == 0:
                st.write(a2adf_test_matrix(st.session_state.a2df_sel_crypto_ts_pivot))
        # Index ADF Test
        with a2tmp_col1:
            st.write("ADF Test for Indexes")
            if not len(st.session_state.a2df_sel_index_ts) == 0:
                st.write(a2adf_test_matrix(st.session_state.a2df_sel_index_ts_pivot))

with tabs[1]:
    st.write('Moving Averages Selected Assets')
    with expander("Significance of No of Days Moving Average", expanded=False):
        st.html(st.session_state.a2html_50days_ma_str)

    a2tmp_col1, a2tmp_col2 = st.columns(2, border=True)

    with a2tmp_col1:
        window = st.select_slider("Select Moving Average Window",  # Title
        options=[15, 21, 30, 50, 100, 200],  # Allowed values
        format_func=lambda x: f"{x} Days",  # Display label with "Days"
        value=50 ) # Default selection

    with a2tmp_col2:
        with expander("Significance of Weightage", expanded=False):
            st.html(st.session_state.a2html_50days_ma_str)
        weight = st.select_slider("Select Weightage for Weighed Moving Average",  # Title
        options=[0.25, 0.5, 0.75],  # Allowed values
        format_func=lambda x: f"{x:.2f}",  # Display label with percentage
        value=0.5 )

    # Define Moving Average types and corresponding tab titles
    asset_type_tabs = ['💰 Primary Asset', '💸 Equities Moving Averages','💷 MFI Moving Averages','₿ Crypto Moving Averages','🗄️ Crypto ETF Moving Averages','🗃️ Index funcs Moving Averages']
    sub_tabs1 = st.tabs(asset_type_tabs)

    with sub_tabs1[0]:
        if not len(st.session_state.a2df_sel_primary_ts) == 0:
            a2process_all_assets_ma(st.session_state.a2df_sel_primary_ts, window)

    with sub_tabs1[1]:
        if not len(st.session_state.a2df_sel_equity_ts) == 0:
            tmp_df_x = st.session_state.a2df_sel_equity_ts.query("asset_code!= '{0}'".format(st.session_state.a2primary_asset))
            if not len(tmp_df_x) == 0:
                a2process_all_assets_ma(tmp_df_x, window)

    with sub_tabs1[2]:
        if not len(st.session_state.a2df_sel_mfi_ts) == 0:
            tmp_df_x = st.session_state.a2df_sel_mfi_ts.query("asset_code!= '{0}'".format(st.session_state.a2primary_asset))
            if not len(tmp_df_x) == 0:
                a2process_all_assets_ma(tmp_df_x, window)

    with sub_tabs1[2]:
        if not len(st.session_state.a2df_sel_crypto_ts) == 0:
            tmp_df_x = st.session_state.a2df_sel_crypto_ts.query("asset_code!= '{0}'".format(st.session_state.a2primary_asset))
            if not len(tmp_df_x) == 0:
                a2process_all_assets_ma(tmp_df_x, window)

    with sub_tabs1[3]:
        if not len(st.session_state.a2df_sel_crypto_etf_ts) == 0:
            tmp_df_x = st.session_state.a2df_sel_crypto_etf_ts.query("asset_code!= '{0}'".format(st.session_state.a2primary_asset))
            if not len(tmp_df_x) == 0:
                a2process_all_assets_ma(tmp_df_x, window)

    with sub_tabs1[4]:
        if not len(st.session_state.a2df_sel_index_ts) == 0:
            tmp_df_x = st.session_state.a2df_sel_index_ts.query("asset_code!= '{0}'".format(st.session_state.a2primary_asset))
            if not len(tmp_df_x) == 0:
                a2process_all_assets_ma(tmp_df_x, window)

with tabs[2]:
    with expander("What is Exponential Smoothing and its significance", expanded=True):
        st.html(st.session_state.a2html_str_exp_smoothing)

    a2tmp_col_alpha, a2tmp_col_beta, a2tmp_col_gamma = st.columns(3, border=True)
    with a2tmp_col_alpha:
        with expander("α - alpha - for Exponential Smoothing", expanded = True):
            a2alpha = a2set_alpha_slider()
    with a2tmp_col_beta:
        with expander("β - beta - for Double Exponential Smoothing", expanded = True):
            a2beta = a2set_beta_slider()
    with a2tmp_col_gamma:
        with expander("γ - beta - for Triple Exponential Smoothing", expanded = True):
            a2gamma = a2set_gamma_slider()

    # Define tab names and corresponding smoothing types
    a2titles_es = ['🛞 Simple Exp Smoothing', '🛞🛞 Double Exp Smoothing', '🛞🛞🛞 Triple Exp Smoothing']
    a2sub_tab_titles_es = st.tabs(a2titles_es)

    # Create tabs in Streamlit for three types of exponential smoothing
    with a2sub_tab_titles_es[0]: # single exponential smoothing
        # Run SES model
        for asset_name, df_ses, df_ses_metric, df_ses_3alpha in st.session_state.a2datasets_ses:
            if len(df_ses) ==0:
                continue
            a2process_all_assets(df_ses, df_ses_metric, df_ses_3alpha)

    with a2sub_tab_titles_es[1]: # Double exponential smoothing
        # Run DES model
        for asset_name, df_des, df_des_metric, df_des_3beta in st.session_state.a2datasets_des:
            if len(df_des) ==0:
                continue
            a2process_all_assets_des(df_des, df_des_metric, df_des_3beta)

    with a2sub_tab_titles_es[2]: # Double exponential smoothing
        # Run TES model
        for asset_name, df_tes, df_tes_metric, df_tes_3gamma in st.session_state.a2datasets_tes:
            if len(df_tes) ==0:
                continue
            a2process_all_assets_tes(df_tes, df_tes_metric, df_tes_3gamma)

with tabs[3]:
    st.subheader('ARIMA Model')
    st.write('ARIMA Model')

with tabs[4]:
    st.subheader('SARIMA Model')
    st.write('SARIMA Model')

with tabs[5]:
    st.subheader('GARIMA Model')
    st.write('ARIMA Model')

tmp_merged_df = pd.DataFrame()

if not len(st.session_state.a2merged_df) == 0:
    # Process the merged dataset for ARIMA/SARIMA/GARMIA/GARCH
    tmp_merged_df = st.session_state.a2merged_df.copy()
    # Convert 'date' column to datetime

    tmp_merged_df['date'] = pd.to_datetime(tmp_merged_df['date'], errors='coerce')
    # Drop NaN from essential columns
    tmp_merged_df.dropna(subset=['close', 'date'], inplace=True)
    # Sort by asset and date
    tmp_merged_df = tmp_merged_df.sort_values(['asset_code', 'date'])

    # Process each asset separately
    for asset_code in tmp_merged_df['asset_code'].unique():
        tmp_df = tmp_merged_df.query(f"asset_code == '{asset_code}'")

        with tabs[3]:# Display ARIMA forecasts
            with expander(asset_code + " - ARIMA Model Forecast, Metrics, Residual Results, Accuracy", expanded=True):
                a2results_df, a2forecast_df = arima_forecast(tmp_df, asset_code)
                st.info(asset_code + " - ARIMA Metrics")
                st.dataframe(a2results_df, hide_index = True, use_container_width=True)
                st.info(asset_code + " - ARIMA Forecast Details")
                #st.dataframe(a2forecast_df, hide_index = True, use_container_width=True)
                st.altair_chart(plot_forecast_arima_sarima(a2forecast_df), use_container_width=True)

        with tabs[4]:# Display ARIMA forecasts
            with expander(asset_code + " - SARIMA Model Forecast, Metrics, Residual Results, Accuracy", expanded=True):
                a2results_df, a2forecast_df, a2model_fit = sarima_forecast(tmp_df, asset_code)
                st.info(asset_code + " - SARIMA Metrics")
                st.dataframe(a2results_df, hide_index = True, use_container_width=True)
                st.info(asset_code + " - SARIMA Forecast Details")
                #st.dataframe(a2forecast_df, hide_index = True, use_container_width=True)
                st.altair_chart(plot_forecast_arima_sarima(a2forecast_df), use_container_width=True)
                st.info(asset_code + " - SARIMA Residual Analysis")
                a2residual_stats, a2hist_chart, a2line_chart, a2boxplot, tmp_model_fit = sarima_residuals(a2model_fit)
                tmp_col1, tmp_col2, tmp_col3 = st.columns(3, border=True)
                with tmp_col1:
                    if a2hist_chart:
                        st.altair_chart(a2hist_chart,  use_container_width=True)
                with tmp_col2:
                    if a2line_chart:
                        st.altair_chart(a2line_chart, use_container_width=True)
                with tmp_col3:
                    if a2boxplot:
                        st.altair_chart(a2boxplot, use_container_width=True)

        with tabs[5]:# Display ARIMA forecasts
            with expander(asset_code + " - GARIMA Model Forecast, Metrics, Residual Results, Accuracy", expanded=True):
                results_df, forecast_df, model_fit = garima_forecast(tmp_df, asset_code)
                st.info(asset_code + " - GARIMA Metrics")
                st.dataframe(results_df, hide_index = True, use_container_width=True)
                #st.dataframe(a2forecast_df, hide_index = True, use_container_width=True)
                st.info(asset_code + " - GARIMA Forecast Details")
                st.altair_chart(plot_forecast_arima_sarima(forecast_df), use_container_width=True)

                tmp_col1, tmp_col2, tmp_col3 = st.columns(3, border=True)
                with tmp_col1:
                    if a2hist_chart:
                        st.altair_chart(a2hist_chart,  use_container_width=True)
                with tmp_col2:
                    if a2line_chart:
                        st.altair_chart(a2line_chart, use_container_width=True)
                with tmp_col3:
                    if a2boxplot:
                        st.altair_chart(a2boxplot, use_container_width=True)
