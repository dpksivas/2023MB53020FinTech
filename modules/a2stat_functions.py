import duckdb as dd
import random

from statsmodels.tsa.stattools import adfuller
import plotly.express as px
import plotly.colors as pc
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sklearn.metrics as metrics
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from streamlit import expander
from statsmodels.stats.diagnostic import het_arch

from modules.a2init_session_variables import *
import warnings

warnings.filterwarnings("ignore")

a2db = dd.connect("a2db.db", read_only=True)

a2colors1_lst = ['darkorange', 'blueviolet', 'purple', 'lightcoral', 'brown','gold' ]
a2colors2_lst = ['tan','firebrick','deepskyblue','tomato','salmon','red']
a2colors3_lst = ['coral','lightsalmon','olive','blue','darkblue','orange']
a2colors4_lst = ['slateblue','lime','darkgreen', 'cyan','rosybrown','silver']
a2colors5_lst = ['royalblue','navy','magenta', 'deeppink','green','gray']

@st.cache_data (ttl=3*60*60)
def a2get_colour(in_num: int):
    if in_num == 1:
        return a2colors1_lst[random.randrange(0, 5)]
    elif in_num == 2:
        return a2colors2_lst[random.randrange(0, 5)]
    elif in_num == 3:
        return a2colors3_lst[random.randrange(0, 5)]
    elif in_num == 4:
        return a2colors4_lst[random.randrange(0, 5)]
    elif in_num == 5:
        return a2colors5_lst[random.randrange(0, 5)]
    else:
        return 'black'

def a2plot_forecast(df: pd.DataFrame):

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date', ascending=True)

    # Determine trend by comparing the first two close values
    trend_color = 'green' if df.iloc[0]['forecast'] <= df.iloc[4]['forecast'] else 'red'

    # Compute the domain for the y-axis with a 10% padding
    min_forecast = df['forecast'].min()
    max_forecast = df['forecast'].max()
    padding = (max_forecast - min_forecast) * 0.1
    y_domain = [min_forecast - padding, max_forecast + padding]

    # Create the line chart with orange color and group by asset_code
    line_chart = alt.Chart(df).mark_line(color=trend_color, strokeWidth=1).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('forecast:Q', title='Forecast', scale=alt.Scale(domain=y_domain, zero=False)),
        detail='asset_code:N'
    )

    # Create text labels for each forecast value
    text_labels = alt.Chart(df).mark_text(
        align='left',
        dx=5,  # horizontal offset for clarity
        dy=-5,  # vertical offset for clarity
        color='black'
    ).encode(
        x='date:T',
        y=alt.Y('forecast:Q', scale=alt.Scale(domain=y_domain, zero=False)),
        text=alt.Text('forecast:Q', format='.1f')
    )

    # Combine the line chart and text labels
    chart = (line_chart + text_labels).properties(title="Forecasting Trend")

    return chart

@st.cache_data(ttl=3*60*60)
def a2plot_ses_des_tes_chart(asset_df: pd.DataFrame):
    """
    Generate and display interactive Altair charts for Double Exponential Smoothing (DES).
    :param asset_df: DataFrame with 'date', 'close', 'smoothed', 'forecast', 'asset_code'.
    """

    a2asset_code = asset_df['asset_code'].unique()[0]

    # Melt DataFrame for Actual & Smoothed plotting
    melted_df = asset_df.melt(id_vars=['date', 'asset_code'],
                              value_vars=['close', 'smoothed'],
                              var_name='Type', value_name='Value')

    # Ensure forecast_df contains only valid values
    forecast_df = asset_df[['date', 'forecast']].dropna()

    # Create interactive selection (Zoom & Pan)
    zoom = alt.selection_interval(bind='scales')

    # Base chart
    base = alt.Chart(melted_df).encode(x='date:T')

    # Line chart for Actual & Smoothed values
    line_chart = base.mark_line(strokeWidth=.8).encode(
        y=alt.Y('Value:Q', title='Price'),
        color=alt.Color('Type:N', title="Legend",
                        scale=alt.Scale(domain=['close', 'smoothed'],
                                        range=['orange', 'green'])),
        tooltip=['date:T', 'Value:Q']
    ).add_selection(zoom)  # Enable zoom & pan

    # Forecast line (Purple, Dashed)
    forecast_chart = alt.Chart(forecast_df).mark_line(
        color='purple', strokeDash=[4, 4], strokeWidth=0.8
    ).encode(
        x='date:T',
        y='forecast:Q',
        tooltip=['date:T', 'forecast:Q']
    )

    # Forecast points with labels
    forecast_labels = alt.Chart(forecast_df).mark_text(
        align='left', dx=5, dy=-5, color='purple'
    ).encode(
        x='date:T',
        y='forecast:Q',
        text=alt.Text('forecast:Q', format='.2f')
    )

    # Combine all elements
    chart = alt.layer(line_chart, forecast_chart, forecast_labels).properties(
        title=f"{a2asset_code} - DES Forecast",
        width=900,
        height=450
    ).configure_legend(orient='top')  # ✅ Legend at the top

    # Display chart in Streamlit
    #st.info(f"{a2asset_code} - Double Exponential Smoothing (DES) Forecast")
    return chart

@st.cache_data(ttl=3*60*60)
def a2adf_test_matrix(df_pivot: pd.DataFrame):
    """
    Computes the ADF test for stationarity along with mean, variance, and autocorrelation.

    Parameters:
        df (pd.DataFrame): DataFrame with columns Date, asset_code, and closed_price.

    Returns:
        pd.DataFrame: ADF test results including mean, variance, and autocorrelation.
        :param df_pivot:
    """
    results = []

    # Pivot DataFrame
    #df_pivot = df.pivot(index='date', columns='asset_code', values='close')

    for asset_code in df_pivot.columns:
        series = df_pivot[asset_code].dropna()

        if len(series) > 1:  # Ensure sufficient data points
            adf_result = adfuller(series)
            mean_value = series.mean()
            variance_value = series.var()
            autocorr_value = series.autocorr(lag=1)  # Lag-1 autocorrelation
            reject_hypothesis = 'Yes' if adf_result[1] < 0.05 else 'No'

            results.append([
                asset_code,
                adf_result[0],  # ADF Statistic
                adf_result[1],  # p-value
                reject_hypothesis,
                round(mean_value,2),
                round(variance_value,2),
                round(autocorr_value,2)
            ])

    # Create DataFrame
    result_df = pd.DataFrame(
        results,
        columns=['Asset Code', 'ADF Statistic', 'p-value', 'Stationarity', 'Mean', 'Variance', 'Autocorrelation']
    )

    return result_df

@st.cache_data(ttl=3*60*60)
def a2process_correlations(df_pivot: pd.DataFrame, a2primary_asset, a2method='pearson'):
    """
    Computes the correlation matrix for a given dataframe based on a primary asset.

    Parameters:
        df (pd.DataFrame): Raw DataFrame with Date, asset_code, and closed_price.
        primary_asset (str): The primary asset to compute correlation against.
        method (str): Correlation method - 'pearson' or 'spearman'. Default is 'pearson'.

    Returns:
        pd.DataFrame: Correlation matrix filtered for the primary asset.
    """
    if a2method not in ['pearson', 'spearman']:
        raise ValueError("Method must be either 'pearson' or 'spearman'")

    # Pivot DataFrame
    #df_pivot = df.pivot(index='date', columns='asset_code', values='close')

    correlation_matrix = df_pivot.corr(method=a2method)

    if a2primary_asset not in correlation_matrix.columns:
        raise ValueError("Primary asset not found in DataFrame")

    return correlation_matrix[[a2primary_asset]]

@st.cache_data(ttl=3*60*60)
def plot_correlation_heatmap_px(correlation_matrix, a2primary_asset, method, a2colorscale):
    """
    Plots a correlation heatmap using Plotly Express.

    Parameters:
        correlation_matrix (pd.DataFrame): The correlation matrix DataFrame.
        a2primary_asset (str): The primary asset name.
        method (str): Correlation method used.
        a2colorscale (str): The color scale to use (e.g., 'RdBu', 'Viridis', 'Cividis').
    """

    fig = px.imshow(correlation_matrix,labels={"x": "Asset Code", "y": "Asset Code", "color": f"{method.capitalize()} Correlation"},
        x=correlation_matrix.columns, y=correlation_matrix.index, text_auto=".2f",
        color_continuous_scale=a2colorscale,  # Custom color scale
        zmin=-1, zmax=1)  # Keep the scale between -1 and 1

    fig.update_layout(title=f"{method.capitalize()} Correlation Heatmap for {a2primary_asset}",autosize=True)

    fig['layout']['yaxis'].update(autorange=True)

    return fig

@st.cache_data(ttl=3*60*60)
def plot_covariance_heatmap(df_pivot: pd.DataFrame, a2primary_asset, a2colorscale):
    """
    Computes the covariance matrix for asset prices and plots it as a heatmap against the primary asset.

    Parameters:
        tmp_df (pd.DataFrame): Input DataFrame with 'Date', 'asset_code', 'closed_price'.
        a2primary_asset (str): The primary asset against which covariance is computed.
        a2colorscale (str): The color scale to use for the heatmap.

    Returns:
        None (Displays the plot in Streamlit)
    """
    # Pivot the DataFrame to have dates as index and assets as columns
    #df_pivot = tmp_df.pivot(index="date", columns="asset_code", values="close")

    # Compute the covariance matrix
    covariance_matrix = df_pivot.cov()

    # Ensure primary asset exists
    if a2primary_asset not in covariance_matrix.columns:
        raise ValueError(f"Primary asset '{a2primary_asset}' not found in DataFrame.")

    # Extract covariance row (preserving 2D shape)
    covariance_with_primary = covariance_matrix[[a2primary_asset]]

    # Ensure correct shape for heatmap
    covariance_with_primary = covariance_with_primary.T  # Transpose to keep 2D shape

    # Plot covariance heatmap
    fig = px.imshow(
        covariance_with_primary,  # Properly structured 2D matrix
        labels={"x": "Primary Asset", "y": "Asset Code", "color": "Covariance"},
        x=covariance_with_primary.columns.tolist(),  # Primary asset(s)
        y=covariance_with_primary.index.tolist(),  # Other assets
        text_auto=".2f",  # Display covariance values inside
        color_continuous_scale=a2colorscale  # Allow custom color scale
    )

    fig.update_layout(
        title=f"Covariance Heatmap Against {a2primary_asset}",
        autosize=True
    )

    fig['layout']['yaxis'].update(autorange=True)

    return fig

@st.cache_data(ttl=3*60*60)
def eigen_decomposition_from_covariance(df_pivot: pd.DataFrame, color_scale):
    """
    Computes the covariance matrix and plots eigenvectors with asset codes as labels.

    Parameters:
        df (pd.DataFrame): DataFrame with columns Date, asset_code, and closed_price.

    Returns:
        Plotly figure of eigenvectors.
        :param color_scale:
        :param df_pivot:
    """
    #df_pivot = df.pivot(index='date', columns='asset_code', values='close').dropna()

    df_pivot = df_pivot.dropna()  # Remove rows with NaN

    # Compute covariance matrix
    covariance_matrix = np.cov(df_pivot.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    asset_codes = df_pivot.columns
    eigenvector_df = pd.DataFrame(eigenvectors, index=asset_codes,
                                  columns=[f"Eigenvector {i + 1}" for i in range(len(asset_codes))])

    # Convert to long format for better visualization
    eigenvector_df = eigenvector_df.reset_index().melt(id_vars='asset_code', var_name='Component', value_name='Value')
    # Get color list from the selected scale
    num_colors = eigenvector_df["Component"].nunique()

    colors = pc.sample_colorscale(color_scale, [i / (num_colors - 1) for i in range(num_colors)])
    fig = px.bar(
        eigenvector_df,
        x='asset_code',
        y='Value',
        color='Component',
        barmode='group',
        title='Eigenvectors of Covariance Matrix',
        labels={'asset_code': 'Asset Code', 'Value': 'Eigenvector Component'},
        color_discrete_sequence=colors  # Apply the generated color scale
    )

    fig['layout']['yaxis'].update(autorange=True)

    return fig

@st.cache_data(ttl=3*60*60)
def a2plot_with_param(df_pivot: pd.DataFrame, primary_asset, a2type: str):
    """
    Computes the percentage change of 'close' prices and plots the trend
    for the primary asset along with other assets.

    Parameters:
        df (pd.DataFrame): DataFrame with columns (asset_code, date, close, volume)
        primary_asset (str): The asset code to compare other assets against.

    Returns:
        Plotly figure of percentage change trends.
        :param a2type:
        :param primary_asset:
        :param df_pivot:
    """
    # Ensure date is in datetime format
    # df_pivot['date'] = df_pivot.to_datetime(df_pivot['date'])

    # Pivot DataFrame to have assets as columns
    # df_pivot = df.pivot(index='date', columns='asset_code', values='close')

    # Manually compute percentage change (avoiding deprecated pct_change())
    df_pct_change = ((df_pivot / df_pivot.shift(1)) - 1) * 100  # Convert to percentage

    # Reshape for Plotly
    df_pct_change = df_pct_change.reset_index().melt(id_vars='date', var_name='asset_code', value_name='pct_change')

    # Drop NaN values
    df_pct_change = df_pct_change.dropna()

    # Apply log transformation (log base e)
    if a2type == 'log':  # log transformation
        df_log = np.log(df_pivot)
        # Convert back to long format for Plotly
        df_log = df_log.reset_index().melt(id_vars='date', var_name='asset_code', value_name='log_close')

    # Plot using Plotly
    if a2type == 'p':  # percentage change
        fig = px.line(
            df_pct_change,
            x='date',
            y='pct_change',
            color='asset_code',
            title=f'Percentage Change in Close Prices (Primary: {primary_asset})',
            labels={'pct_change': 'Percentage Change (%)', 'date': 'Date', 'asset_code': 'Asset Code'},
            line_shape='linear'
        )
        # Highlight primary asset
        fig.update_traces(
            selector=lambda trace: trace.name == primary_asset,
            line=dict(width=3, dash='solid')
        )
    if a2type == 'd':  # daily distributed
        fig = px.histogram(
            df_pct_change,
            x='pct_change',
            color='asset_code',
            marginal='box',  # Adds a box plot on the side
            barmode='overlay',  # Overlays histograms for all assets
            opacity=0.6,  # Adjust transparency for overlapping bars
            title='Distribution of Daily Percentage Change',
            labels={'pct_change': 'Daily Percentage Change (%)', 'asset_code': 'Asset Code'}
        )

    if a2type == 'log':  # daily distributed
        # Plot with Plotly Express
        fig = px.line(
            df_log,
            x='date',
            y='log_close',
            color='asset_code',
            title='Log-Transformed Close Prices Over Time',
            labels={'log_close': 'Log(Close Price)', 'date': 'Date', 'asset_code': 'Asset Code'},
            line_dash=df_log['asset_code'].apply(lambda x: 'solid' if x == primary_asset else 'dot'),
            # Highlight primary asset
        )

    return fig

@st.cache_data(ttl=3*60*60)
def a2plot_asset_prices(df: pd.DataFrame, a2titleStr: str):
    """
    Plots equity asset close prices over time using Altair for Streamlit.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'date', 'close', and 'asset_code' columns.
        a2titleStr (str): Title for the chart.

    Returns:
        Altair Chart
    """
    selection = alt.selection_multi(fields=['asset_code'], bind='legend')

    # Create Altair Line Chart with clickable legend
    chart = alt.Chart(df).mark_line(strokeWidth=1).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('close:Q', title='Close Price'),
        color=alt.Color(
            'asset_code:N',
            legend=alt.Legend(title="Asset Code", orient='top')  # Horizontal legend
        ),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),  # Hide lines when deselected
        tooltip=['date:T', 'close:Q', 'asset_code:N']
    ).add_selection(
        selection
    ).properties(
        title=f'{a2titleStr} Asset Close Prices - INR - Over Time',
        width=800,
        height=500
    ).interactive()

    return chart

@st.cache_data(ttl=3*60*60)
def a2calculate_moving_avg(df: pd.DataFrame, ma_type='sma', window=15, weight=None):
    """
    Calculates and plots different types of moving averages for asset prices.

    Parameters:
        df (pd.DataFrame): DataFrame with 'date', 'asset_code', and 'close'.
        ma_type (str): Type of moving average ('sma', 'cma', 'ema', 'ewma').
        window (int): Number of days for the moving average window.
        weight (float, optional): Weight for EWMA.

    Returns:
        Altair chart object.
    """
    if len(df) == 0:
        return None

    df = df.copy()  # Avoid modifying the original DataFrame
    df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' is a datetime type

    if ma_type == 'sma':  # Simple Moving Average
        df['moving_avg'] = df.groupby('asset_code')['close'].transform(lambda x: x.rolling(window).mean())
    elif ma_type == 'cma':  # Cumulative Moving Average
        df['moving_avg'] = df.groupby('asset_code')['close'].transform(lambda x: x.expanding().mean())
    elif ma_type == 'ema':  # Exponential Moving Average
        df['moving_avg'] = df.groupby('asset_code')['close'].transform(lambda x: x.ewm(span=window, adjust=False).mean())
    elif ma_type == 'ewma':  # Exponentially Weighted Moving Average
        if weight is None:
            weight = 0.5  # Default weight
        df['moving_avg'] = df.groupby('asset_code')['close'].transform(lambda x: x.ewm(alpha=weight).mean())
    else:
        raise ValueError("Invalid moving average type. Choose from 'sma', 'cma', 'ema', 'ewma'.")

    # Altair Line Chart
    line_chart = alt.Chart(df).mark_line(strokeWidth=1).encode(
        x=alt.X('date:T', title='Date'),  # Ensure date is treated as temporal
        y=alt.Y('moving_avg:Q', title='Moving Average'),
        color=alt.Color('asset_code:N', title='Asset Code',
                        legend=alt.Legend(orient='top', title="Moving Averages")  # Horizontal legend
                        ),
        tooltip=['date:T', 'asset_code:N', 'moving_avg:Q']
    ).properties(
        autosize=alt.AutoSizeParams(type='fit', contains='content'),
        title=alt.TitleParams(
            text=f"{window}-Day {ma_type.upper()} Moving Average",
            anchor='middle'  # Centers the title
        )
    ).interactive()

    return line_chart


def calculate_moving_avg_forecast_summary(df: pd.DataFrame, window=15, weight=None, forecast_days=5):
    """
    For each moving average type (SMA, CMA, EMA, EWMA):
      1. Compute the moving average series and add a column "moving_avg_type".
      2. Calculate error metrics (MAE, MA, RMSE) per asset and aggregate them.
      3. Generate a dynamic forecast for the next `forecast_days` using linear regression on recent moving_avg values.

    Returns:
        combined_chart (Altair Chart): Combined interactive chart of historical moving averages (solid)
                                       and forecast (dashed) for all MA types.
        forecast_all_chart (Altair Chart): Interactive chart showing forecast lines (dashed) with forecast values overlaid.
        forecast_all (pd.DataFrame): Forecast DataFrame with columns [asset_code, date, forecast, moving_avg_type].
        summary_df (pd.DataFrame): Summary DataFrame with error metrics per moving average type.
    """
    # Ensure a copy and that dates are datetime
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    ma_types = ['sma', 'cma', 'ema', 'ewma']

    # Lists to hold historical data and forecast data for each MA type
    hist_list = []
    forecast_list = []

    # Dictionary to hold aggregated error metrics for each type
    summary_metrics = {}

    for ma_type in ma_types:
        temp_df = df.copy()

        # Compute the moving average series based on the type
        if ma_type == 'sma':
            temp_df['moving_avg'] = temp_df.groupby('asset_code')['close'].transform(
                lambda x: x.rolling(window).mean()
            )
        elif ma_type == 'cma':
            temp_df['moving_avg'] = temp_df.groupby('asset_code')['close'].transform(
                lambda x: x.expanding().mean()
            )
        elif ma_type == 'ema':
            temp_df['moving_avg'] = temp_df.groupby('asset_code')['close'].transform(
                lambda x: x.ewm(span=window, adjust=False).mean()
            )
        elif ma_type == 'ewma':
            # For EWMA, if weight is not provided, perform a brute-force search for the best weight (minimizing RMSE globally)
            if weight is None:
                candidate_weights = np.linspace(0.01, 0.99, 99)
                best_rmse = np.inf
                best_weight = None
                for cand in candidate_weights:
                    candidate_series = temp_df.groupby('asset_code')['close'].transform(
                        lambda x: x.ewm(alpha=cand).mean()
                    )
                    valid = temp_df['close'].notna() & candidate_series.notna()
                    rmse_candidate = np.sqrt(np.mean((temp_df.loc[valid, 'close'] - candidate_series.loc[valid]) ** 2))
                    if rmse_candidate < best_rmse:
                        best_rmse = rmse_candidate
                        best_weight = cand
                used_weight = best_weight
            else:
                used_weight = weight
            temp_df['moving_avg'] = temp_df.groupby('asset_code')['close'].transform(
                lambda x: x.ewm(alpha=used_weight).mean()
            )
        else:
            raise ValueError("Invalid moving average type. Choose from 'sma', 'cma', 'ema', 'ewma'.")

        # Add a column to indicate the moving average type
        temp_df['moving_avg_type'] = ma_type

        # Append the historical data (keeping only necessary columns)
        hist_list.append(temp_df[['date', 'asset_code', 'moving_avg', 'moving_avg_type']])

        # Calculate error metrics per asset (ignoring rows where moving_avg is NaN)
        mae_list, ma_list, rmse_list = [], [], []
        for asset in temp_df['asset_code'].unique():
            asset_data = temp_df[temp_df['asset_code'] == asset].dropna(subset=['moving_avg'])
            if asset_data.empty:
                continue
            errors = asset_data['close'] - asset_data['moving_avg']
            mae_list.append(np.mean(np.abs(errors)))
            ma_list.append(np.mean(errors))
            rmse_list.append(np.sqrt(np.mean(errors ** 2)))
        agg_mae = np.mean(mae_list) if mae_list else np.nan
        agg_ma = np.mean(ma_list) if ma_list else np.nan
        agg_rmse = np.mean(rmse_list) if rmse_list else np.nan
        summary_metrics[ma_type] = {'moving_avg_type': ma_type, 'MAE': agg_mae, 'MA': agg_ma, 'RMSE': agg_rmse}

        # Generate dynamic forecast for each asset: forecast next `forecast_days` using linear regression on recent moving_avg values
        last_date = temp_df['date'].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        for asset in temp_df['asset_code'].unique():
            asset_data = temp_df[temp_df['asset_code'] == asset].dropna(subset=['moving_avg']).sort_values('date')
            if asset_data.empty:
                continue
            if len(asset_data) >= 2:
                # Use the last min(5, len(asset_data)) points for linear regression
                n_points = min(5, len(asset_data))
                recent_data = asset_data.iloc[-n_points:]
                # Convert dates to numeric values (days relative to the first date in recent_data)
                x = (recent_data['date'] - recent_data['date'].min()).dt.days.values
                y = recent_data['moving_avg'].values
                # Fit a linear model: y = m*x + b
                m, b = np.polyfit(x, y, 1)
                forecast_vals = []
                for fd in forecast_dates:
                    x_fd = (fd - recent_data['date'].min()).days
                    forecast_vals.append(m * x_fd + b)
            else:
                # If only one data point is available, use a constant forecast
                forecast_vals = [asset_data['moving_avg'].iloc[-1]] * forecast_days

            forecast_temp = pd.DataFrame({
                'asset_code': asset,
                'date': forecast_dates,
                'forecast': forecast_vals,
                'moving_avg_type': ma_type
            })
            forecast_list.append(forecast_temp)

    # Combine all historical and forecast data into single DataFrames
    hist_all = pd.concat(hist_list, ignore_index=True)
    forecast_all = pd.concat(forecast_list, ignore_index=True)  # This is a pandas DataFrame.

    # Convert date columns to ISO-formatted strings for JSON serialization (required for Streamlit)
    if pd.api.types.is_datetime64_any_dtype(hist_all['date']):
        hist_all['date'] = hist_all['date'].dt.strftime('%Y-%m-%d')
    if pd.api.types.is_datetime64_any_dtype(forecast_all['date']):
        forecast_all['date'] = forecast_all['date'].dt.strftime('%Y-%m-%d')

    # Compute y-axis domain for forecast chart (min and max forecast values with padding)
    y_min = forecast_all['forecast'].min()
    y_max = forecast_all['forecast'].max()
    padding = (y_max - y_min) * 0.1 if y_max != y_min else 1
    forecast_domain = [y_min - padding, y_max + padding]

    # Create historical chart
    chart_hist = alt.Chart(hist_all).mark_line(strokeWidth=1).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('moving_avg:Q', title='Moving Average'),
        color=alt.Color('moving_avg_type:N', title='MA Type'),
        detail='asset_code:N',
        tooltip=['date:T', 'asset_code:N', 'moving_avg_type:N', 'moving_avg:Q']
    ).properties(
        width=600,
        height=400,
        title=f"{window}-Day Moving Averages (Historical and 5-Day Dynamic Forecast)"
    )

    # Create forecast chart (dashed lines)
    chart_forecast = alt.Chart(forecast_all).mark_line(strokeDash=[5, 5]).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('forecast:Q', title='Forecast', scale=alt.Scale(domain=forecast_domain)),
        color=alt.Color('moving_avg_type:N', title='MA Type'),
        detail='asset_code:N',
        tooltip=['date:T', 'asset_code:N', 'moving_avg_type:N', 'forecast:Q']
    )

    # Create forecast text labels to display forecast values on the chart
    forecast_text = alt.Chart(forecast_all).mark_text(
        dy=-5, color='black'
    ).encode(
        x=alt.X('date:T'),
        y=alt.Y('forecast:Q', scale=alt.Scale(domain=forecast_domain)),
        text=alt.Text('forecast:Q', format='.2f'),
        detail='asset_code:N'
    )

    # Combined chart: historical plus forecast (make interactive)
    combined_chart = (chart_hist + chart_forecast).interactive()

    # Create forecast-only chart (layer forecast lines and text) and make interactive
    forecast_all_chart = (chart_forecast + forecast_text).properties(
        width=600,
        height=400,
        title='5-Day Dynamic Forecast (Forecast Only)'
    ).interactive()

    # Create summary DataFrame from the error metrics dictionary
    summary_df = pd.DataFrame(list(summary_metrics.values()))

    return combined_chart, forecast_all_chart, forecast_all, summary_df

def a2set_alpha_slider():
    # Streamlit slider for alpha selection
    a2alpha = st.slider(
        "α → Controls the smoothing level trend-adj moving avg",
        min_value=0.1,
        max_value=1.0,
        step=0.1,
        value=0.5,  # Default alpha
        format="%.1f",
        help="Higher α (0.6 - 1.0) → Short-term trading, fast adaptation \n"
             "Mid β (0.3 - 0.6) → Balanced & Swing Trading \n"
             "Lower β (0.1 - 0.3) → Long-term investing, stable but slow",
        key="a2alpha_ses"
    )

    # Display alpha selection
    if a2alpha >= 0.6:
        st.info("📉 Higher α (0.6 - 1.0) → Short-term Trading (Fast but Noisy)")
    elif 0.3 <= a2alpha < 0.6:
        st.success("📊 Mid α (0.3 - 0.6) → Swing Trading (Balanced)")
    else:
        st.warning("📈 Lower α (0.1 - 0.3) → Long-term Investing (Stable but Slow)")

def a2set_beta_slider():
    # Streamlit slider for selecting Beta (β)
    a2beta = st.slider(
        "β → Controls the smoothing of the trend component: ",
        min_value=0.1, max_value=1.0, step=0.1, value=0.3,
        help="Higher β (0.6 - 1.0) → Short-term trend, fast adaptation \n"
             "Mid β (0.3 - 0.6) → Balanced trend following \n"
             "Lower β (0.1 - 0.3) → Long-term investing, stable trends",
        key="a2beta_des"
    )

    # Display selected category based on Beta value
    if a2beta >= 0.6:
        st.info("📉 **Higher β (0.6 - 1.0)** → Short-term trend, fast adaptation")
    elif 0.3 <= a2beta < 0.6:
        st.success("📊 **Mid β (0.3 - 0.6)** → Balanced trend following")
    else:
        st.warning("📈 **Lower β (0.1 - 0.3)** → Long-term investing, stable trends")

def a2set_gamma_slider():
    # Streamlit slider for selecting Gamma (γ)
    a2gamma = st.slider(
        "γ → Controls the smoothing of the seasonal component",
        min_value=0.1, max_value=1.0, step=0.1, value=0.5,
        help="Higher γ (0.6 - 1.0) → Short-term volatile, cyclical markets \n"
             "Mid γ (0.3 - 0.6) → Balanced seasonal adjustment \n"
             "Lower γ (0.1 - 0.3) → Long-term trend stability", key="a2gamma_tes")

    # Display selected category based on Gamma value
    if a2gamma >= 0.6:
        st.info("🌊 **Higher γ (0.6 - 1.0)** → Short-term volatile, cyclical markets")
    elif 0.3 <= a2gamma < 0.6:
        st.success("⚖ **Mid γ (0.3 - 0.6)** → Balanced seasonal adjustment")
    else:
        st.warning("🏦 **Lower γ (0.1 - 0.3)** → Long-term trend stability")

@st.cache_data(ttl=3*60*60)
def a2forecast_df(df, forecast_periods=4):
    """
    Performs time series forecasting using Simple Exponential Smoothing (SES).

    :param df: DataFrame with 'date', 'forecast'.
    :param forecast_periods: Number of days to forecast.
    :return: DataFrame with actual, smoothed, forecast values.
    """

    if df.empty:
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'])
    tmp_df = df.copy()

    a2max_dt = tmp_df['date'].max()
    date_threshold = a2max_dt - pd.Timedelta(days=forecast_periods)  # Use last year's data for predictions'

    tmp_df_predicted_1 = tmp_df[tmp_df['date'] >= date_threshold]
    tmp_df_predicted = tmp_df_predicted_1[['asset_code', 'date', 'forecast']]

    return date_threshold,tmp_df_predicted
    #pd.DataFrame([tmp_df_predicted['forecast'].values],
                              #columns=[f"Day{i + 1}_Forecast" for i in range(len(tmp_df_predicted))]))

def timeseries_evaluation_metrics_func(y_true, y_pred):
    """Calculates time series evaluation metrics."""
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true) else 0  # Avoid div by zero
    r2 = metrics.r2_score(y_true, y_pred)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R²": r2}


@st.cache_data(ttl=3*60*60)
def a2train_and_forecast_ses(df, forecast_periods=7):
    """
    Train & optimize a Simple Exponential Smoothing model, evaluate on test data,
    and forecast future values using Grid Search & Automated Smoothing.

    :param df: DataFrame with 'asset_code', 'date', 'close'.
    :param forecast_periods: Number of days to forecast.
    :return: DataFrame with actual, smoothed, forecast values & error metrics.
    """

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['asset_code', 'date'])

    results_list = []
    metrics_list = []
    grid_search_results = []  # To store top 5 smoothing factors for all assets

    for asset_code in df['asset_code'].unique():
        asset_df = df[df['asset_code'] == asset_code].copy()
        train_size = int(len(asset_df) * .999)  # Use 90% for training, 10% for testing
        train, test = asset_df.iloc[:train_size], asset_df.iloc[train_size:]

        best_rmse, best_alpha, best_model = float("inf"), None, None
        temp_list = []  # Store results for grid search

        # 🔹 Grid search over alpha values from 0 to 1 with step 0.1
        for i in np.arange(0, 1.1, 0.1):
            model = SimpleExpSmoothing(np.asarray(train['close'])).fit(smoothing_level=i, optimized=False)
            pred = model.forecast(len(test))
            rmse = np.sqrt(metrics.mean_squared_error(test['close'], pred))

            temp_list.append({'asset_code': asset_code, 'smoothing_level': i, 'RMSE': rmse})  # Store results

            if rmse < best_rmse:
                best_rmse, best_alpha, best_model = rmse, i, model

        # Convert list to DataFrame & select the best 5 smoothing parameters instead of 3
        temp_df = pd.DataFrame(temp_list).sort_values(by='RMSE', ascending=True).head(5)
        grid_search_results.append(temp_df)  # Store the top 5 results for this asset

        # 🔹 Train final model using best alpha from grid search
        best_model_gs = SimpleExpSmoothing(np.asarray(train['close'])).fit(smoothing_level=best_alpha, optimized=False)
        gs_pred = best_model_gs.forecast(len(test))
        gs_rmse = np.sqrt(metrics.mean_squared_error(test['close'], gs_pred))

        # 🔹 Train an automated model with brute-force optimization
        best_model_auto = SimpleExpSmoothing(np.asarray(train['close'])).fit(optimized=True, use_brute=True)
        auto_pred = best_model_auto.forecast(len(test))
        auto_rmse = np.sqrt(metrics.mean_squared_error(test['close'], auto_pred))

        # **Compare & Select the Best Model**
        if gs_rmse <= auto_rmse:
            final_model = best_model_gs
            final_alpha = best_alpha
            final_rmse = gs_rmse
        else:
            final_model = best_model_auto
            final_alpha = "Auto (Brute Force)"
            final_rmse = auto_rmse

        # 🔹 Assign fitted values only for training period
        asset_df.loc[train.index, 'smoothed'] = final_model.fittedvalues

        # 🔹 Final forecast
        forecast_index = pd.date_range(start=asset_df['date'].iloc[-1], periods=forecast_periods + 1, freq='D')[1:]
        forecast_values = final_model.forecast(forecast_periods)

        if len(forecast_index) != len(forecast_values):
            raise ValueError(f"Mismatch: forecast_index={len(forecast_index)}, forecast_values={len(forecast_values)}")

        forecast_df = pd.DataFrame({'date': forecast_index, 'forecast': forecast_values, 'asset_code': asset_code})

        # 🔹 Compute metrics
        test_pred = final_model.forecast(len(test))
        eval_metrics = timeseries_evaluation_metrics_func(test['close'], test_pred)
        eval_metrics["Alpha"] = final_alpha
        eval_metrics["asset_code"] = asset_code  # Fix: Add asset_code in metrics
        eval_metrics["RMSE"] = final_rmse

        # 🔹 Store results
        metrics_list.append(eval_metrics)
        results_list.append(
            pd.concat([asset_df[['date', 'close', 'smoothed', 'asset_code']], forecast_df], ignore_index=True))

    df_results = pd.concat(results_list, ignore_index=True)
    df_metrics = pd.DataFrame(metrics_list)  # Fix: Metrics now have asset_code
    best_grid_search_params = pd.concat(grid_search_results, ignore_index=True)  # Concatenate top 5 smoothing factors

    return df_results, df_metrics, best_grid_search_params

@st.cache_data(ttl=3*60*60)
def a2plot_ses_best_grid_search_plotly(best_grid_search_params):
    """
    Generate and display a horizontal bar chart for the best smoothing levels (alphas)
    from the Simple Exponential Smoothing (SES) grid search using Plotly.

    :param best_grid_search_params: DataFrame with 'asset_code', 'smoothing_level', 'RMSE'.
    """

    if best_grid_search_params.empty:
        st.warning("No grid search results available.")
        return

    # Sort values for better visualization
    best_grid_search_params = best_grid_search_params.sort_values(by=['asset_code', 'RMSE'])

    # Create unique colors for each bar
    unique_colors = px.colors.qualitative.Set3  # Distinct colors

    # Create Plotly Horizontal Bar Chart
    fig = px.bar(
        best_grid_search_params,
        y="smoothing_level",  # Horizontal bars (X-axis flipped)
        x="RMSE",
        color="smoothing_level",
        text="RMSE",  # Display RMSE values inside bars
        title="Top 5 Best Smoothing Levels (SES Grid Search)",
        labels={"smoothing_level": "Smoothing Level (Alpha)", "RMSE": "RMSE (Lower is Better)"},
        orientation='h',  # Horizontal bars
        width=900,
        height=500,
        color_discrete_sequence=unique_colors  # Assign distinct colors
    )

    # Customize bar appearance
    fig.update_traces(
        texttemplate='%{text:.2f}',  # Format RMSE inside bars
        textposition='inside',  # Place values inside bars
        insidetextanchor='start',  # Align text within bars
        marker=dict(line=dict(width=1, color='black')),  # Thin black outlines for bars
    )

    # Customize layout
    fig.update_layout(
        xaxis=dict(title="RMSE (Lower is Better)", gridcolor='lightgray'),
        yaxis=dict(title="Smoothing Level (Alpha)", categoryorder="total ascending"),
        legend_title="Smoothing Level",
        template="plotly_white"
    )

    # Display in Streamlit
    st.info("Best Alpha Values from Grid Search (Horizontal Bar Chart)")
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=3*60*60)
def a2train_and_forecast_des(df, forecast_periods=7):
    """
    Train and optimize a Double Exponential Smoothing (Holt's Linear Trend) model using
    brute-force grid search over (alpha, beta) and optimize based on RMSE.

    :param df: DataFrame with 'asset_code', 'date', 'close', 'volume'.
    :param forecast_periods: Number of days to forecast.
    :return: DataFrame with actual, smoothed, forecast values, and best grid search parameters.
    """

    if df.empty:
        print("[ERROR] ❌ No data in input DataFrame. Returning empty results.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ✅ Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # ✅ Remove previous forecast values
    df = df.drop(columns=['forecast'], errors='ignore')

    # ✅ Drop NaN from essential columns
    df.dropna(subset=['close', 'date'], inplace=True)

    # ✅ Sort by asset and date
    df = df.sort_values(['asset_code', 'date'])

    results_list = []
    metrics_list = []
    grid_search_results = []

    # ✅ Process each asset separately
    for asset_code in df['asset_code'].unique():
        print(f"\n[PROCESSING ASSET] 🔹 {asset_code}")

        asset_df = df[df['asset_code'] == asset_code].copy()

        train_size = int(len(asset_df) * 0.95)
        train, test = asset_df.iloc[:train_size], asset_df.iloc[train_size:]

        if len(train) < 5 or len(test) < 5:
            print(f"[WARNING] ⚠️ Skipping {asset_code} - Insufficient train/test split.")
            continue

        best_rmse, best_alpha, best_beta, best_model = float("inf"), None, None, None
        temp_list = []

        # ✅ Brute-force grid search over alpha & beta (0.1 to 1.0)
        for alpha in np.arange(0.1, 1.1, 0.1):
            for beta in np.arange(0.1, 1.1, 0.1):  # ✅ Ensure beta > 0
                try:
                    # ✅ Corrected: Removed trend from Holt() initialization
                    model = Holt(np.asarray(train['close'])).fit(
                        smoothing_level=alpha,
                        smoothing_trend=beta,
                        optimized=False
                    )

                    pred = model.forecast(len(test))

                    if np.isnan(pred).any():
                        continue

                    rmse = np.sqrt(mean_squared_error(test['close'], pred))

                    temp_list.append({
                        'asset_code': asset_code,
                        'smoothing_level': alpha,
                        'beta': beta,
                        'RMSE': rmse
                    })

                    if rmse < best_rmse:
                        best_rmse, best_alpha, best_beta, best_model = rmse, alpha, beta, model

                except Exception as e:
                    print(f"[ERROR] ❌ Issue with alpha={alpha}, beta={beta} for {asset_code}: {e}")
                    continue

        # ✅ Store the **top 5** best (alpha, beta) combinations
        if temp_list:
            temp_df = pd.DataFrame(temp_list).sort_values(by='RMSE', ascending=True).head(5)
            grid_search_results.append(temp_df)
        else:
            print(f"[WARNING] ⚠️ No valid RMSE values found for {asset_code}. Skipping.")
            continue

        print(f"[SELECTED MODEL] Alpha: {best_alpha}, Beta: {best_beta}")

        # ✅ Final forecast
        forecast_index = pd.date_range(start=asset_df['date'].iloc[-1], periods=forecast_periods + 1, freq='D')[1:]
        forecast_values = best_model.forecast(forecast_periods)

        if np.isnan(forecast_values).any():
            print(f"[WARNING] ⚠️ Skipping forecast for {asset_code} due to NaN values.")
            continue

        forecast_df = pd.DataFrame({'date': forecast_index, 'forecast': forecast_values, 'asset_code': asset_code})

        # ✅ Compute evaluation metrics (Only RMSE)
        test_pred = best_model.forecast(len(test))
        eval_metrics = {
            "Alpha": best_alpha,
            "Beta": best_beta,
            "RMSE": best_rmse,
            "asset_code": asset_code
        }

        # ✅ Ensure final dataset contains all previous data + new DES forecasts
        asset_df = pd.concat([asset_df, forecast_df], ignore_index=True)
        results_list.append(asset_df)
        metrics_list.append(eval_metrics)

    print("[FINAL OUTPUT] Returning final DES results.")
    return (
        pd.concat(results_list, ignore_index=True) if results_list else pd.DataFrame(),
        pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame(),
        pd.concat(grid_search_results, ignore_index=True) if grid_search_results else pd.DataFrame()
    )


# this is alternative to previous version of the function that include decomposing seasonality.
@st.cache_data(ttl=3*60*60)
def a2train_and_forecast_tes(df, forecast_periods=5): # focus only short term
    """
    Train and optimize a Triple Exponential Smoothing (Holt-Winters) model using brute-force
    grid search over (alpha, beta, gamma) for short-term forecasting.

    :param df: DataFrame with 'asset_code', 'date', 'close', 'volume'.
    :param forecast_periods: Number of days to forecast (default: 5 days).
    :return: DataFrame with actual, smoothed, forecast values, error metrics for TES, and best grid search parameters.
    """

    if df.empty:
        print("[ERROR] ❌ No data in input DataFrame. Returning empty results.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ✅ Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # ✅ Remove previous forecast values
    df = df.drop(columns=['forecast'], errors='ignore')

    # ✅ Drop NaN from essential columns
    df.dropna(subset=['close', 'date'], inplace=True)

    # ✅ Sort by asset and date
    df = df.sort_values(['asset_code', 'date'])

    results_list = []
    metrics_list = []
    grid_search_results = []

    # ✅ Process each asset separately
    for asset_code in df['asset_code'].unique():
        print(f"\n[PROCESSING ASSET for TES] 🔹 {asset_code}")

        asset_df = df[df['asset_code'] == asset_code].copy()

        # ✅ Train on the last 4 months of data
        min_train_date = asset_df['date'].max() - pd.DateOffset(months=4)
        train = asset_df[asset_df['date'] >= min_train_date]
        test = asset_df[asset_df['date'] > train['date'].max()]

        if len(train) < 60 or len(test) < 5:
            print(f"[WARNING] ⚠️ Skipping {asset_code} - Insufficient train/test split. Train={len(train)}, Test={len(test)}")
            continue

        # ✅ Set seasonality to 7 days (weekly pattern)
        best_seasonal_period = 7

        best_rmse, best_alpha, best_beta, best_gamma, best_seasonality, best_model = float("inf"), None, None, None, None, None
        temp_list = []

        # ✅ Brute-force grid search over alpha, beta, and gamma (Short-term tuning)
        for alpha in np.arange(0.7, 1.1, 0.1):  # ✅ Prioritize recent data
            for beta in np.arange(0.5, 1.1, 0.1):  # ✅ Adapt trend faster
                for gamma in np.arange(0.6, 1.1, 0.1):  # ✅ Adjust seasonality quickly
                    try:
                        # ✅ Use both additive & multiplicative seasonality
                        for seasonality_type in ["add", "mul"]:
                            model = ExponentialSmoothing(
                                np.asarray(train['close']),
                                trend="add",  # ✅ Additive trend
                                seasonal=seasonality_type,
                                seasonal_periods=best_seasonal_period
                            ).fit(
                                smoothing_level=alpha,
                                smoothing_trend=beta,
                                smoothing_seasonal=gamma,
                                optimized=False
                            )

                            pred = model.forecast(len(test))

                            if np.isnan(pred).any():
                                continue

                            rmse = np.sqrt(mean_squared_error(test['close'], pred))

                            temp_list.append({
                                'asset_code': asset_code,
                                'smoothing_level': alpha,
                                'beta': beta,
                                'gamma': gamma,
                                'seasonality': seasonality_type,
                                'seasonal_periods': best_seasonal_period,
                                'RMSE': rmse
                            })

                            if rmse < best_rmse:
                                best_rmse, best_alpha, best_beta, best_gamma, best_seasonality, best_model = rmse, alpha, beta, gamma, seasonality_type, model

                    except Exception as e:
                        print(f"[ERROR] ❌ Issue with alpha={alpha}, beta={beta}, gamma={gamma} for {asset_code}: {e}")
                        continue

        # ✅ Store the best top 5 combinations
        if temp_list:
            temp_df = pd.DataFrame(temp_list).sort_values(by='RMSE', ascending=True).head(5)
            grid_search_results.append(temp_df)
        else:
            print(f"[WARNING] ⚠️ No valid RMSE values found for {asset_code}. Skipping.")
            continue

        print(f"[SELECTED MODEL] Alpha: {best_alpha}, Beta: {best_beta}, Gamma: {best_gamma}, Seasonality: {best_seasonality}, Period: {best_seasonal_period}")

        # ✅ Final forecast (5 Days)
        forecast_index = pd.date_range(start=asset_df['date'].iloc[-1], periods=forecast_periods + 1, freq='D')[1:]
        forecast_values = best_model.forecast(forecast_periods)

        if np.isnan(forecast_values).any():
            print(f"[WARNING] ⚠️ Skipping forecast for {asset_code} due to NaN values.")
            continue

        forecast_df = pd.DataFrame({'date': forecast_index, 'forecast': forecast_values, 'asset_code': asset_code})

        # ✅ Compute evaluation metrics
        test_pred = best_model.forecast(len(test))
        eval_metrics = {
            "Alpha": best_alpha,
            "Beta": best_beta,
            "Gamma": best_gamma,
            "Seasonality": best_seasonality,
            "Seasonal Periods": best_seasonal_period,
            "RMSE": best_rmse,
            "asset_code": asset_code
        }

        # ✅ Ensure final dataset contains all previous forecasts + new TES forecasts
        asset_df = pd.concat([asset_df, forecast_df], ignore_index=True)
        results_list.append(asset_df)
        metrics_list.append(eval_metrics)

    print("[FINAL OUTPUT] Returning final short-term TES results.")
    return (
        pd.concat(results_list, ignore_index=True) if results_list else pd.DataFrame(),
        pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame(),
        pd.concat(grid_search_results, ignore_index=True) if grid_search_results else pd.DataFrame()
    )

def a2train_and_forecast_des_short_term(df, forecast_periods=5):
    """
    Train and optimize a Double Exponential Smoothing (Holt's Linear Trend) model using
    brute-force grid search over (alpha, beta) with at least 7 days of test data.

    :param df: DataFrame with 'asset_code', 'date', 'close', 'volume'.
    :param forecast_periods: Number of days to forecast (default: 5 days).
    :return: DataFrame with actual, smoothed, forecast values, and best grid search parameters.
    """

    if df.empty:
        print("[ERROR] ❌ No data in input DataFrame. Returning empty results.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ✅ Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # ✅ Remove previous forecast values
    df = df.drop(columns=['forecast'], errors='ignore')

    # ✅ Drop NaN from essential columns
    df.dropna(subset=['close', 'date'], inplace=True)

    # ✅ Sort by asset and date
    df = df.sort_values(['asset_code', 'date'])

    results_list = []
    metrics_list = []
    grid_search_results = []

    # ✅ Process each asset separately
    for asset_code in df['asset_code'].unique():
        print(f"\n[PROCESSING ASSET] 🔹 {asset_code}")

        asset_df = df[df['asset_code'] == asset_code].copy()

        # ✅ Get min and max date
        max_date = asset_df['date'].max()
        min_date = asset_df['date'].min()

        # ✅ Ensure test dataset has at least 7 days
        test_start_date = max_date - pd.Timedelta(days=7)  # atleast Last 7 days for testing considering market holidays

        # Ensure test_start_date is within dataset range
        if test_start_date < min_date:
            print(f"[ERROR] ❌ Not enough data for a 7-day test set. Min Date: {min_date}, Max Date: {max_date}")
            continue

        # Train-Test Split
        train = asset_df[asset_df['date'] < test_start_date]
        test = asset_df[asset_df['date'] >= test_start_date]

        print(f"✅ Train Data: {len(train)} rows (From {train['date'].min()} to {train['date'].max()})")
        print(f"✅ Test Data: {len(test)} rows (From {test['date'].min()} to {test['date'].max()})")

        if len(test) < 3:
            print(f"[WARNING] ⚠️ Test set has only {len(test)} days. Consider adjusting the dataset.")
            continue

        best_rmse, best_alpha, best_beta, best_model = float("inf"), None, None, None
        temp_list = []

        # ✅ Brute-force grid search optimized for short-term forecasting
        for alpha in np.arange(0.7, 1.1, 0.1):  # ✅ Focus on higher alpha (recent data)
            for beta in np.arange(0.6, 1.1, 0.1):  # ✅ Focus on adaptive trend updates
                try:
                    # ✅ Using additive trend
                    model = Holt(
                        np.asarray(train['close']),
                        exponential=False  # ✅ Using additive trend for better short-term prediction
                    ).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)

                    pred = model.forecast(len(test))

                    if np.isnan(pred).any():
                        continue

                    rmse = np.sqrt(mean_squared_error(test['close'], pred))

                    temp_list.append({
                        'asset_code': asset_code,
                        'smoothing_level': alpha,
                        'beta': beta,
                        'RMSE': rmse
                    })

                    if rmse < best_rmse:
                        best_rmse, best_alpha, best_beta, best_model = rmse, alpha, beta, model

                except Exception as e:
                    print(f"[ERROR] ❌ Issue with alpha={alpha}, beta={beta} for {asset_code}: {e}")
                    continue

        # ✅ Store the **top 5** best (alpha, beta) combinations
        if temp_list:
            temp_df = pd.DataFrame(temp_list).sort_values(by='RMSE', ascending=True).head(5)
            grid_search_results.append(temp_df)
        else:
            print(f"[WARNING] ⚠️ No valid RMSE values found for {asset_code}. Skipping.")
            continue

        print(f"[SELECTED MODEL] Alpha: {best_alpha}, Beta: {best_beta}")

        # ✅ Final forecast (Max 5 days)
        forecast_index = pd.date_range(start=asset_df['date'].iloc[-1], periods=forecast_periods + 1, freq='D')[1:]
        forecast_values = best_model.forecast(forecast_periods)

        if np.isnan(forecast_values).any():
            print(f"[WARNING] ⚠️ Skipping forecast for {asset_code} due to NaN values.")
            continue

        forecast_df = pd.DataFrame({'date': forecast_index, 'forecast': forecast_values, 'asset_code': asset_code})

        # ✅ Compute evaluation metrics (Only RMSE)
        test_pred = best_model.forecast(len(test))
        eval_metrics = {
            "Alpha": best_alpha,
            "Beta": best_beta,
            "RMSE": best_rmse,
            "asset_code": asset_code
        }

        # ✅ Ensure final dataset contains all previous data + new DES forecasts
        asset_df = pd.concat([asset_df, forecast_df], ignore_index=True)
        results_list.append(asset_df)
        metrics_list.append(eval_metrics)

    print("[FINAL OUTPUT] Returning final DES results.")
    return (
        pd.concat(results_list, ignore_index=True) if results_list else pd.DataFrame(),
        pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame(),
        pd.concat(grid_search_results, ignore_index=True) if grid_search_results else pd.DataFrame()
    )

def a2train_and_forecast_tes_stable(df, forecast_periods=5, seasonality_period=14):
    """
    Train and optimize a Triple Exponential Smoothing (Holt-Winters) model using brute-force
    grid search over (alpha, beta, gamma) optimized for **stable short-term** forecasting.

    :param df: DataFrame with 'asset_code', 'date', 'close', 'volume'.
    :param forecast_periods: Number of days to forecast (default: 5 days).
    :param seasonality_period: Seasonal period for TES model (default: 14 days).
    :return: DataFrame with actual, smoothed, forecast values, error metrics for TES, and best grid search parameters.
    """

    if df.empty:
        print("[ERROR] ❌ No data in input DataFrame. Returning empty results.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ✅ Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # ✅ Remove previous forecast values
    df = df.drop(columns=['forecast'], errors='ignore')

    # ✅ Drop NaN from essential columns
    df.dropna(subset=['close', 'date'], inplace=True)

    # ✅ Sort by asset and date
    df = df.sort_values(['asset_code', 'date'])

    results_list = []
    metrics_list = []
    grid_search_results = []

    # ✅ Process each asset separately
    for asset_code in df['asset_code'].unique():
        print(f"\n[PROCESSING ASSET] 🔹 {asset_code}")

        asset_df = df[df['asset_code'] == asset_code].copy()

        # ✅ Get min and max date
        max_date = asset_df['date'].max()
        min_date = asset_df['date'].min()

        # ✅ Use last 9 months for training
        train_start_date = max_date - pd.Timedelta(days=270)

        # Ensure train_start_date is within dataset range
        if train_start_date < min_date:
            train_start_date = min_date  # Adjust to available data range

        # ✅ Ensure test dataset has at least 7 days
        test_start_date = max_date - pd.Timedelta(days=7)

        # Ensure test_start_date is within dataset range
        if test_start_date < train_start_date:
            print(f"[ERROR] ❌ Not enough data for a 7-day test set. Min Date: {min_date}, Max Date: {max_date}")
            continue

        # Train-Test Split
        train = asset_df[asset_df['date'] >= train_start_date]
        train = train[train['date'] < test_start_date]
        test = asset_df[asset_df['date'] >= test_start_date]

        print(f"✅ Train Data: {len(train)} rows (From {train['date'].min()} to {train['date'].max()})")
        print(f"✅ Test Data: {len(test)} rows (From {test['date'].min()} to {test['date'].max()})")

        if len(test) < 3:
            print(f"[WARNING] ⚠️ Test set has only {len(test)} days. Consider adjusting the dataset.")
            continue

        best_rmse, best_alpha, best_beta, best_gamma, best_seasonality, best_model = float("inf"), None, None, None, None, None
        temp_list = []

        # ✅ Brute-force grid search optimized for **stable short-term** forecasting
        for alpha in np.arange(0.5, 0.9, 0.1):  # ✅ Lower alpha to reduce overreaction
            for beta in np.arange(0.3, 0.8, 0.1):  # ✅ Moderate trend adjustments
                for gamma in np.arange(0.2, 0.6, 0.1):  # ✅ Smoother seasonal effect
                    try:
                        # ✅ Use both Additive & Multiplicative Seasonality
                        for seasonality_type in ["add", "mul"]:
                            model = ExponentialSmoothing(
                                np.asarray(train['close']),
                                trend="add",  # ✅ Additive trend for stability
                                seasonal=seasonality_type,
                                seasonal_periods=seasonality_period
                            ).fit(
                                smoothing_level=alpha,
                                smoothing_trend=beta,
                                smoothing_seasonal=gamma,
                                optimized=False
                            )

                            pred = model.forecast(len(test))

                            if np.isnan(pred).any():
                                continue

                            rmse = np.sqrt(mean_squared_error(test['close'], pred))

                            temp_list.append({
                                'asset_code': asset_code,
                                'smoothing_level': alpha,
                                'beta': beta,
                                'gamma': gamma,
                                'seasonality': seasonality_type,
                                'seasonal_periods': seasonality_period,
                                'RMSE': rmse
                            })

                            if rmse < best_rmse:
                                best_rmse, best_alpha, best_beta, best_gamma, best_seasonality, best_model = (
                                    rmse, alpha, beta, gamma, seasonality_type, model
                                )

                    except Exception as e:
                        print(f"[ERROR] ❌ Issue with alpha={alpha}, beta={beta}, gamma={gamma} for {asset_code}: {e}")
                        continue

        # ✅ Store the **top 5** best (alpha, beta, gamma) combinations
        if temp_list:
            temp_df = pd.DataFrame(temp_list).sort_values(by='RMSE', ascending=True).head(5)
            grid_search_results.append(temp_df)
        else:
            print(f"[WARNING] ⚠️ No valid RMSE values found for {asset_code}. Skipping.")
            continue

        print(f"[SELECTED MODEL] Alpha: {best_alpha}, Beta: {best_beta}, Gamma: {best_gamma}, Seasonality: {best_seasonality}, Period: {seasonality_period}")

        # ✅ Final forecast (Max 5 days)
        forecast_index = pd.date_range(start=asset_df['date'].iloc[-1], periods=forecast_periods + 1, freq='D')[1:]
        forecast_values = best_model.forecast(forecast_periods)

        forecast_df = pd.DataFrame({'date': forecast_index, 'forecast': forecast_values, 'asset_code': asset_code})

        eval_metrics = {
            "Alpha": best_alpha,
            "Beta": best_beta,
            "Gamma": best_gamma,
            "Seasonality": best_seasonality,
            "Seasonal Periods": seasonality_period,
            "RMSE": best_rmse,
            "asset_code": asset_code
        }

        asset_df = pd.concat([asset_df, forecast_df], ignore_index=True)
        results_list.append(asset_df)
        metrics_list.append(eval_metrics)

    print("[FINAL OUTPUT] Returning final TES results.")
    return (
        pd.concat(results_list, ignore_index=True),
        pd.DataFrame(metrics_list),
        pd.concat(grid_search_results, ignore_index=True)
    )

@st.cache_data(ttl=3*60*60)
def a2train_and_forecast_des(df, forecast_periods=5):
    """
    Train and optimize a Double Exponential Smoothing (Holt's Linear Trend) model
    using brute-force grid search for (alpha, beta), optimized based on RMSE, MAE, MAPE, and R².

    :param df: DataFrame with 'asset_code', 'date', 'close', 'volume'.
    :param forecast_periods: Number of days to forecast.
    :return: DataFrame with actual, smoothed, forecast values, error metrics for DES, and best grid search parameters.
    """

    if df.empty:
        print("[ERROR] ❌ No data in input DataFrame. Returning empty results.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ✅ Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # ✅ Remove previous forecast values
    df = df.drop(columns=['forecast'], errors='ignore')

    # ✅ Drop NaN from essential columns
    df.dropna(subset=['close', 'date'], inplace=True)

    # ✅ Sort by asset and date
    df = df.sort_values(['asset_code', 'date'])

    results_list = []
    metrics_list = []
    grid_search_results = []

    # ✅ Process each asset separately
    for asset_code in df['asset_code'].unique():
        print(f"\n[PROCESSING ASSET] 🔹 {asset_code}")

        asset_df = df[df['asset_code'] == asset_code].copy()

        train_size = int(len(asset_df) * 0.9)
        train, test = asset_df.iloc[:train_size], asset_df.iloc[train_size:]

        if len(train) < 5 or len(test) < 5:
            print(f"[WARNING] ⚠️ Skipping {asset_code} - Insufficient train/test split.")
            continue

        best_rmse, best_mae, best_mape, best_r2 = float("inf"), float("inf"), float("inf"), -float("inf")
        best_alpha, best_beta, best_model = None, None, None
        temp_list = []

        # ✅ Brute-force grid search over alpha and beta (0.1 to 1.0)
        for alpha in np.arange(0.7, 1.1, 0.1):
            for beta in np.arange(0.7, 1.1, 0.1):  # ✅ Ensure beta > 0
                try:
                    model = Holt(np.asarray(train['close']), exponential=False).fit(
                        smoothing_level=alpha,
                        smoothing_trend=beta,
                        optimized=False
                    )

                    pred = model.forecast(len(test))

                    if np.isnan(pred).any():
                        continue

                    rmse = np.sqrt(mean_squared_error(test['close'], pred))
                    mae = mean_absolute_error(test['close'], pred)
                    mape = np.mean(np.abs((test['close'] - pred) / test['close'])) * 100
                    r2 = r2_score(test['close'], pred)

                    temp_list.append({
                        'asset_code': asset_code,
                        'smoothing_level': alpha,
                        'beta': beta,
                        'RMSE': rmse,
                        'MAE': mae,
                        'MAPE (%)': mape,
                        'R²': r2
                    })

                    # ✅ Select best model based on RMSE, MAE, MAPE, and R²
                    if rmse < best_rmse and mae < best_mae and mape < best_mape and r2 > best_r2:
                        best_rmse, best_mae, best_mape, best_r2 = rmse, mae, mape, r2
                        best_alpha, best_beta, best_model = alpha, beta, model

                except Exception as e:
                    print(f"[ERROR] ❌ Issue with alpha={alpha}, beta={beta} for {asset_code}: {e}")
                    continue

        # ✅ Store the best top 5 combinations
        if temp_list:
            temp_df = pd.DataFrame(temp_list).sort_values(by=['RMSE', 'MAE', 'MAPE (%)', 'R²'], ascending=[True, True, True, False]).head(5)
            grid_search_results.append(temp_df)
        else:
            print(f"[WARNING] ⚠️ No valid RMSE values found for {asset_code}. Skipping.")
            continue

        print(f"[SELECTED MODEL] Alpha: {best_alpha}, Beta: {best_beta}")

        # ✅ Final forecast
        forecast_index = pd.date_range(start=asset_df['date'].iloc[-1], periods=forecast_periods + 1, freq='D')[1:]
        forecast_values = best_model.forecast(forecast_periods)

        if np.isnan(forecast_values).any():
            print(f"[WARNING] ⚠️ Skipping forecast for {asset_code} due to NaN values.")
            continue

        forecast_df = pd.DataFrame({'date': forecast_index, 'forecast': forecast_values, 'asset_code': asset_code})

        # ✅ Compute evaluation metrics
        test_pred = best_model.forecast(len(test))
        eval_metrics = {
            "Alpha": best_alpha,
            "Beta": best_beta,
            "RMSE": best_rmse,
            "MAE": best_mae,
            "MAPE (%)": best_mape,
            "R²": best_r2,
            "asset_code": asset_code
        }

        # ✅ Ensure final dataset contains all previous forecasts + new DES forecasts
        asset_df = pd.concat([asset_df, forecast_df], ignore_index=True)
        results_list.append(asset_df)
        metrics_list.append(eval_metrics)

    print("[FINAL OUTPUT] Returning final DES results.")
    return (
        pd.concat(results_list, ignore_index=True) if results_list else pd.DataFrame(),
        pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame(),
        pd.concat(grid_search_results, ignore_index=True) if grid_search_results else pd.DataFrame()
    )

@st.cache_data(ttl=3*60*60)
def a2process_all_assets(df_ses, df_ses_metric, df_ses_3alpha):
    """
    Process SES forecasting for all assets in the dataset.
    :param df_ses_3alpha:
    :param df_ses_metric:
    :param df_ses:
    """
    if len(df_ses) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if df_ses.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if not df_ses.empty:
        df_ses['date'] = pd.to_datetime(df_ses['date'])
        df_ses = df_ses.sort_values(['asset_code', 'date'])

    asset_codes = df_ses["asset_code"].unique()
    num_assets = len(asset_codes)

    # Create dynamic columns with max 2 per row
 #   cols = st.columns(2, border=True)

    for index, asset_code in enumerate(asset_codes):
        #with cols[index % 2]:  # Alternate between columns
        # Display best model metrics
        st.info(asset_code + " - Best Model Metrics & forcasting ")
        metrics_df = df_ses_metric.query(f"asset_code == '{asset_code}'")
        st.dataframe(metrics_df[['Alpha','RMSE','MAE', 'MAPE (%)', 'R²']], hide_index=True, use_container_width=True)
        if not len(df_ses_3alpha) == 0:
            tmp_df = df_ses_3alpha.query(f"asset_code == '{asset_code}'")
            #a2plot_ses_best_grid_search_plotly(tmp_df)
            #st.dataframe(tmp_df[['smoothing_level', 'RMSE']])

        # Display forecast chart
        tmp_df = df_ses.query(f"asset_code == '{asset_code}'")
        tmp_df['date'] = pd.to_datetime(tmp_df['date'])

        st.info(asset_code + " - Forecast Values for Next 5 Days")
        a2forecast_dt, a2forecast_df_tmp = a2forecast_df(tmp_df, 4)

        st.altair_chart(a2plot_forecast(a2forecast_df_tmp), use_container_width=True)
        with expander("Simple Exponential Smoothing Chart", expanded=False):
            st.altair_chart(a2plot_ses_des_tes_chart(tmp_df), use_container_width=True)

        # If it's an odd index, create new row of columns
#        if index % 2 == 1 and index != num_assets - 1:
#            cols = st.columns(2)

@st.cache_data(ttl=3*60*60)
def a2process_all_assets_des(df_des, df_des_metric, df_des_3beta):
    """
    Process SES forecasting for all assets in the dataset.
    :param df_des_3beta:
    :param df_des_metric:
    :param df_des:
    """
    if len(df_des) == 0:
        return

    asset_codes = df_des["asset_code"].unique()
    num_assets = len(asset_codes)

    # Create dynamic columns with max 2 per row
#    cols = st.columns(2, border=True)

    for index, asset_code in enumerate(asset_codes):
#        with cols[index % 2]:  # Alternate between columns
            # Display best model metrics
            st.info(asset_code + " - Best Model Metrics & forcasting ")
            metrics_df = df_des_metric.query(f"asset_code == '{asset_code}'")

            st.dataframe(metrics_df[['Alpha','Beta','RMSE']], hide_index=True, use_container_width=True)
            if not len(df_des_3beta) == 0:
                tmp_df = df_des_3beta.query(f"asset_code == '{asset_code}'")
                #st.info("Top 3 alpha parameters with lowest RMSE")
                #st.dataframe(tmp_df[['smoothing_level', 'beta','RMSE']])

            # Display forecast chart
            tmp_df = df_des.query(f"asset_code == '{asset_code}'")
            tmp_df['date'] = pd.to_datetime(tmp_df['date'])

            st.info(asset_code + " - Forecast Values for Next 5 Days")

            a2forecast_dt, a2forecast_df_tmp = a2forecast_df(tmp_df, 4)
            st.write("Forecast Date - Day 1: " , a2forecast_dt)
            st.altair_chart(a2plot_forecast(a2forecast_df_tmp), use_container_width=True)
            with expander("Double Exponential Smoothing Chart", expanded=False):
                st.altair_chart(a2plot_ses_des_tes_chart(tmp_df), use_container_width=True)


        # If it's an odd index, create new row of columns
#        if index % 2 == 1 and index != num_assets - 1:
#            cols = st.columns(2)

@st.cache_data(ttl=3*60*60)
def a2process_all_assets_tes(df_tes, df_tes_metric, df_tes_3gamma):
    """
    Process SES forecasting for all assets in the dataset.
    :param df_tes_3gamma:
    :param df_tes_metric:
    :param df_tes:
    """
    if len(df_tes) == 0:
        return

    asset_codes = df_tes["asset_code"].unique()
    num_assets = len(asset_codes)

    # Create dynamic columns with max 2 per row
#    cols = st.columns(2, border=True)

    for index, asset_code in enumerate(asset_codes):
#        with cols[index % 2]:  # Alternate between columns
            # Display best model metrics
            st.info(asset_code + " - Best Model Metrics & forcasting ")
            metrics_df = df_tes_metric.query(f"asset_code == '{asset_code}'")
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            if not len(df_tes_3gamma) == 0:
                tmp_df = df_tes_3gamma.query(f"asset_code == '{asset_code}'")
                #st.info("Top 5 alpha parameters with lowest RMSE")
                #st.dataframe(tmp_df[['smoothing_level', 'beta','gamma', 'RMSE']])

            # Display forecast chart
            tmp_df = df_tes.query(f"asset_code == '{asset_code}'")
            tmp_df['date'] = pd.to_datetime(tmp_df['date'])

            st.info(asset_code + " - Forecast Values for Next 5 Days")
            a2forecast_dt, a2forecast_df_tmp = a2forecast_df(tmp_df, 4)
            st.write("Forecast Date - Day 1: " , a2forecast_dt)
            #st.dataframe(a2forecast_df_tmp, use_container_width=True, hide_index=True)
            st.altair_chart(a2plot_forecast(a2forecast_df_tmp), use_container_width=True)
            with expander("Triple Exponential Smoothing Chart", expanded=False):
                st.altair_chart(a2plot_ses_des_tes_chart(tmp_df), use_container_width=True)

        # If it's an odd index, create new row of columns
#        if index % 2 == 1 and index != num_assets - 1:
#            cols = st.columns(2)

def a2process_all_assets_ma(df: pd.DataFrame, window=50):
    """
    Process SES forecasting for all assets in the dataset.
    :param df input:
    :param window:
    """
    if len(df) == 0:
        return

    asset_codes = df["asset_code"].unique()
    num_assets = len(asset_codes)

    for index, asset_code in enumerate(asset_codes):
        # Display best model metrics
        tmp_df = df.query(f"asset_code == '{asset_code}'")
        tmp_df['date'] = pd.to_datetime(tmp_df['date'])

        tmp_chart,tmp_forecast_chart, tmp_df_forecast, tmp_df_ma = calculate_moving_avg_forecast_summary(tmp_df, window, None, 5)
        st.info(asset_code + " - Moving Average Metrics")
        #st.dataframe(tmp_df_ma, hide_index=True, use_container_width=True)
        st.altair_chart(a2plot_box_chart_ma_metrics(tmp_df_ma),use_container_width=True)

        # Display forecast chart
        #st.dataframe(tmp_df_forecast, hide_index=True, use_container_width=True)
        st.info(asset_code + " - Forecast for Next 5 days")
        st.altair_chart(tmp_forecast_chart,use_container_width=True)

        with expander(asset_code + " - Moving Average Trends", expanded=False):
            st.altair_chart(tmp_chart, use_container_width=True)


def a2plot_box_chart_ma_metrics(df: pd.DataFrame):
    """
    Create a grouped bar chart where the x-axis shows metrics (mae, ma, rmse)
    and, within each metric, bars are grouped by moving_avg_type. Each bar is colored
    using a custom color scale and the bar value is displayed on the chart.

    The size (thickness) of the bars is reduced by specifying the `size` property.

    Parameters:
      df (pd.DataFrame): DataFrame containing columns:
                         - moving_avg_type
                         - mae
                         - ma
                         - rmse

    Returns:
      An Altair Chart object.
    """
    # Reshape the data from wide to long format
    melted = df.melt(
        id_vars='moving_avg_type',
        value_vars=['MAE', 'MA', 'RMSE'],
        var_name='metric',
        value_name='value'
    )

    # Define a custom color scale for moving_avg_type.
    custom_colors = ["purple", "orange", "green", "darkblue"]

    # Create the grouped bar chart
    # Here, size=20 reduces the bar thickness (you can adjust this value)
    bar = alt.Chart(melted).mark_bar(size=20).encode(
        x=alt.X('metric:N', title='Metric'),
        xOffset=alt.X('moving_avg_type:N', title='Moving Average Type'),
        y=alt.Y('value:Q', title='Value'),
        color=alt.Color('moving_avg_type:N',
                        scale=alt.Scale(range=custom_colors),
                        legend=alt.Legend(title='Moving Average Type'))
    )

    # Add text labels on top of each bar showing the value
    text = bar.mark_text(
        dy=-5,  # vertical offset
        color='black'
    ).encode(
        text=alt.Text('value:Q', format='.2f')
    )

    chart = (bar + text).properties()

    return chart