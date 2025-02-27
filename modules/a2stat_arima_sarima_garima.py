import random
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import altair as alt

from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from modules.a2stat_functions import a2get_colour
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import het_arch
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error


warnings.filterwarnings("ignore")
import warnings

def arima_forecast(df, asset_code):
    """
    Fits an ARIMA model on the closing price of a given asset and forecasts the next 5 days.
    Optimizes ARIMA for short-term forecasting by limiting historical data usage, reducing parameter grid complexity,
    and focusing on near-term trends.
    Uses dynamic forecasting to capture trend changes over the forecast period.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['asset_code', 'date', 'close', 'volume']
        asset_code (str): Asset code to filter the data

    Returns:
        pd.DataFrame: DataFrame containing best model parameters, metrics, and forecasted values.
    """
    # Filter data for the specified asset
    asset_data = df[df['asset_code'] == asset_code].copy()
    asset_data['date'] = pd.to_datetime(asset_data['date'])
    asset_data = asset_data.sort_values('date')
    asset_data.set_index('date', inplace=True)

    # Use only the last 6 months of data to focus on short-term trends
    six_months_ago = asset_data.index.max() - pd.DateOffset(months=6)
    asset_data = asset_data[asset_data.index >= six_months_ago]

    # Handle missing values
    asset_data['close'].fillna(method='ffill', inplace=True)  # Forward fill
    asset_data.dropna(subset=['close'], inplace=True)  # Drop remaining NaNs

    # Ensure sufficient data points
    if len(asset_data) < 15:
        return pd.DataFrame([{"error": "Not enough data for asset_code: " + asset_code}])

    # Reduce ARIMA grid search complexity for faster short-term predictions with additional optimization
    p_values = range(0, 4)
    d_values = [0, 1, 2]
    q_values = range(0, 4)
    best_rmse, best_cfg, best_model, best_aic, best_bic, best_mae, best_mape = float("inf"), None, None, float(
        "inf"), float("inf"), float("inf"), float("inf")

    # Optimized search: Prioritize models with lower RMSE using GridSearch approach
    param_grid = list(itertools.product(p_values, d_values, q_values))
    for params in param_grid:
        if params == (0, 1, 0):
            continue  # Exclude (0,1,0) model
        try:
            model = ARIMA(asset_data['close'], order=params)
            model_fit = model.fit()
            predictions = model_fit.fittedvalues
            rmse = np.sqrt(mean_squared_error(asset_data['close'], predictions))
            mae = mean_absolute_error(asset_data['close'], predictions)
            mape = np.mean(np.abs((asset_data['close'] - predictions) / asset_data['close'])) * 100
            aic = model_fit.aic
            bic = model_fit.bic

            # Select the best model based strictly on RMSE, keeping AIC & BIC for comparison
            if rmse < best_rmse or (rmse == best_rmse and aic < best_aic):
                best_rmse, best_cfg, best_model, best_aic, best_bic, best_mae, best_mape = rmse, params, model_fit, aic, bic, mae, mape
        except:
            continue

    # Train the best ARIMA model
    if not best_model:
        return pd.DataFrame([{"error": "No valid ARIMA model found."}])

    # Forecast next 5 days dynamically
    forecast_values = []
    lower_bounds = []
    upper_bounds = []
    history = list(asset_data['close'])

    for i in range(5):
        if i == 0:
            model_fit = best_model
        else:
            model = ARIMA(history, order=best_cfg)
            model_fit = model.fit()

        forecast_result = model_fit.get_forecast(steps=1)
        forecast_mean = forecast_result.predicted_mean.item()
        conf_int = forecast_result.conf_int().values[0] if isinstance(forecast_result.conf_int(), pd.DataFrame) else \
        forecast_result.conf_int()[0]

        forecast_values.append(forecast_mean)
        lower_bounds.append(conf_int[0])
        upper_bounds.append(conf_int[1])

        # Append forecast to history for next iteration
        history.append(forecast_mean)
        history = history[-len(asset_data):]  # Keep rolling window to ensure stability

    forecast_dates = pd.date_range(start=asset_data.index[-1] + pd.Timedelta(days=1), periods=5)

    forecast_df = pd.DataFrame({
        "date": forecast_dates,
        "forecast": forecast_values,  # Dynamically generated forecast values
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds
    })

    # Create results DataFrame
    results_df = pd.DataFrame({
        "best_model_order": [best_cfg],
        "rmse": [best_rmse],
        "mae": [best_mae],
        "mape": [best_mape],
        "aic": [best_aic],
        "bic": [best_bic]
    })

    return results_df, forecast_df


def sarima_residuals(model_fit):
    """
    Computes and visualizes residual diagnostics for a SARIMA model.

    Parameters:
        model_fit: The fitted SARIMA model.

    Returns:
        residual_stats (dict): Statistical summary of residuals.
        hist_chart (Altair Chart): Histogram of residuals.
        line_chart (Altair Chart): Residuals over time.
        boxplot (Altair Chart): Box plot of residuals.
        refined_model (SARIMAXResults): Updated model (if needed).
    """

    # Check if the model is valid
    if model_fit is None or not hasattr(model_fit, "resid"):
        print("Warning: Model fit is invalid or missing residuals.")
        return None, None, None, None, None

    # Extract residuals
    residuals = model_fit.resid.dropna()

    # Ensure residuals exist
    if residuals.empty:
        print("Warning: No residuals available.")
        return None, None, None, None, None

    # Filter last 1 year of residuals
    one_year_ago = residuals.index.max() - pd.DateOffset(years=1)
    residuals = residuals[residuals.index >= one_year_ago]

    # Ensure residuals are still available
    if residuals.empty:
        print("Warning: Residuals are empty after filtering 1-year data.")
        return None, None, None, None, None

    # Compute statistics
    residual_stats = {
        "Mean": residuals.mean(),
        "Std Dev": residuals.std(),
        "Min": residuals.min(),
        "Max": residuals.max()
    }

    # Convert residuals to DataFrame for Altair
    residuals_df = pd.DataFrame({"date": residuals.index, "residuals": residuals.values})

    preferred_colour = a2get_colour(random.randrange(1, 5))

    # Histogram of residuals (Colored with preferred_colour)
    hist_chart = alt.Chart(residuals_df).mark_bar(color=preferred_colour).encode(
        x=alt.X("residuals:Q", bin=alt.Bin(maxbins=30), title="Residuals"),
        y=alt.Y("count()", title="Frequency")
    ).properties(title="Histogram of Residuals")

    preferred_colour = a2get_colour(random.randrange(1, 5))

    # Residuals over time (Line chart with preferred_colour)
    line_chart = alt.Chart(residuals_df).mark_line(color=preferred_colour).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("residuals:Q", title="Residuals"),
        tooltip=["date:T", "residuals:Q"]
    ).properties(title="Residuals Over Time").interactive()

    preferred_colour = a2get_colour(random.randrange(1, 5))

    # Box plot (Colored with preferred_colour)
    boxplot = alt.Chart(residuals_df).mark_boxplot(color=preferred_colour).encode(
        y=alt.Y("residuals:Q", title="Residuals")
    ).properties(title="Box Plot of Residuals")

    # Return outputs
    return residual_stats, hist_chart, line_chart, boxplot, model_fit

def garima_residuals(model_fit):
    """
    Computes and visualizes residual diagnostics for a GARIMA model.

    Parameters:
        model_fit: The fitted GARIMA model.

    Returns:
        residual_stats (dict): Statistical summary of residuals.
        hist_chart (Altair Chart): Histogram of residuals.
        line_chart (Altair Chart): Residuals over time.
        boxplot (Altair Chart): Box plot of residuals.
        refined_model (GARIMAResults): Updated model (if needed).
    """

    # Get a random preferred color
    preferred_colour = a2get_colour(random.randrange(1, 5))

    # Check if the model is valid
    if model_fit is None or not hasattr(model_fit, "resid"):
        print("Warning: Model fit is invalid or missing residuals.")
        return None, None, None, None, None

    # Extract residuals
    residuals = model_fit.resid.dropna()

    # Ensure residuals exist
    if residuals.empty:
        print("Warning: No residuals available.")
        return None, None, None, None, None

    # Filter last 1 year of residuals
    one_year_ago = residuals.index.max() - pd.DateOffset(years=1)
    residuals = residuals[residuals.index >= one_year_ago]

    # Ensure residuals are still available
    if residuals.empty:
        print("Warning: Residuals are empty after filtering 1-year data.")
        return None, None, None, None, None

    # Compute statistics
    residual_stats = {
        "Mean": residuals.mean(),
        "Std Dev": residuals.std(),
        "Min": residuals.min(),
        "Max": residuals.max(),
        "Heteroskedasticity P-Value": het_arch(residuals)[1]  # ARCH Test for heteroskedasticity
    }

    # Convert residuals to DataFrame for Altair
    residuals_df = pd.DataFrame({"date": residuals.index, "residuals": residuals.values})

    # Histogram of residuals (Colored with preferred_colour)
    hist_chart = alt.Chart(residuals_df).mark_bar(color=preferred_colour).encode(
        x=alt.X("residuals:Q", bin=alt.Bin(maxbins=30), title="Residuals"),
        y=alt.Y("count()", title="Frequency")
    ).properties(title="Histogram of Residuals")

    # Residuals over time (Line chart with preferred_colour)
    line_chart = alt.Chart(residuals_df).mark_line(color=preferred_colour).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("residuals:Q", title="Residuals"),
        tooltip=["date:T", "residuals:Q"]
    ).properties(title="Residuals Over Time").interactive()

    # Box plot (Colored with preferred_colour)
    boxplot = alt.Chart(residuals_df).mark_boxplot(color=preferred_colour).encode(
        y=alt.Y("residuals:Q", title="Residuals")
    ).properties(title="Box Plot of Residuals")

    # Return outputs
    return residual_stats, hist_chart, line_chart, boxplot, model_fit


def sarima_forecast(df, asset_code):
    """
    Fits a SARIMA model on the closing price of a given asset and forecasts the next 5 days.
    Optimizes SARIMA for short-term forecasting by limiting historical data usage, reducing parameter grid complexity,
    and focusing on near-term trends. Uses dynamic forecasting to capture trend changes over the forecast period.
    Additionally, refines the model based on residual diagnostics.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['asset_code', 'date', 'close', 'volume']
        asset_code (str): Asset code to filter the data

    Returns:
        pd.DataFrame: DataFrame containing best model parameters, metrics, and forecasted values.
        model_fit: Fitted SARIMA model for further analysis.
    """
    # Filter data for the specified asset
    asset_data = df[df['asset_code'] == asset_code].copy()
    asset_data['date'] = pd.to_datetime(asset_data['date'])
    asset_data = asset_data.sort_values('date')
    asset_data.set_index('date', inplace=True)

    # Use only the last 6 months of data to focus on short-term trends
    six_months_ago = asset_data.index.max() - pd.DateOffset(months=6)
    asset_data = asset_data[asset_data.index >= six_months_ago]

    # Handle missing values
    asset_data['close'].fillna(method='ffill', inplace=True)
    asset_data.dropna(subset=['close'], inplace=True)

    # Ensure sufficient data points
    if len(asset_data) < 15:
        return pd.DataFrame([{"error": "Not enough data for asset_code: " + asset_code}])

    # SARIMA parameter grid
    p_values = range(0, 3)
    d_values = [0, 1]
    q_values = range(0, 3)
    P_values = range(0, 2)
    D_values = [0, 1]
    Q_values = range(0, 2)
    s_values = [7]  # Weekly seasonality

    best_rmse, best_cfg, best_model, best_aic, best_bic = float("inf"), None, None, float("inf"), float("inf")

    param_grid = list(itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, s_values))
    for params in param_grid:
        try:
            model = SARIMAX(asset_data['close'], order=params[:3], seasonal_order=params[3:],
                            enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            predictions = model_fit.fittedvalues
            rmse = np.sqrt(mean_squared_error(asset_data['close'], predictions))
            aic = model_fit.aic
            bic = model_fit.bic

            if rmse < best_rmse or (rmse == best_rmse and aic < best_aic):
                best_rmse, best_cfg, best_model, best_aic, best_bic = rmse, params, model_fit, aic, bic
        except:
            continue

    if not best_model:
        return pd.DataFrame([{"error": "No valid SARIMA model found."}])

    # Perform residual diagnostics
    refined_model = best_model

    # Forecast next 5 days dynamically
    forecast_values, lower_bounds, upper_bounds = [], [], []
    history = pd.Series(asset_data['close'].values, index=asset_data.index)

    for i in range(5):
        if i == 0:
            model_fit = refined_model
        else:
            history_series = pd.Series(history.values,
                                       index=pd.date_range(start=history.index[0], periods=len(history)))
            model_fit = SARIMAX(history_series, order=best_cfg[:3], seasonal_order=best_cfg[3:],
                                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

        forecast_result = model_fit.get_forecast(steps=1)
        conf_int = forecast_result.conf_int()
        forecast_mean = forecast_result.predicted_mean.iloc[0] if isinstance(forecast_result.predicted_mean,
                                                                             pd.Series) else forecast_result.predicted_mean
        lower_bound, upper_bound = (conf_int.iloc[0, 0], conf_int.iloc[0, 1]) if isinstance(conf_int, pd.DataFrame) and \
                                                                                 conf_int.shape[1] > 1 else (
        conf_int[0], conf_int[0])

        forecast_values.append(forecast_mean)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

        # Append forecast to history for next iteration using pd.concat
        new_entry = pd.Series([forecast_mean], index=[history.index[-1] + pd.Timedelta(days=1)])
        history = pd.concat([history, new_entry])
        history = history[-len(asset_data):]  # Maintain rolling window

    forecast_dates = pd.date_range(start=asset_data.index[-1] + pd.Timedelta(days=1), periods=5)

    forecast_df = pd.DataFrame({
        "date": forecast_dates,
        "forecast": forecast_values,
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds
    })

    results_df = pd.DataFrame({
        "best_model_order": [best_cfg],
        "rmse": [best_rmse],
        "aic": [best_aic],
        "bic": [best_bic]
    })

    return results_df, forecast_df, refined_model

def plot_forecast_arima_sarima(df):
    """
    Creates an Altair line chart showing forecast, lower bound, and upper bound
    with dynamically adjusted Y-axis and **values displayed on the lines**.

    Custom colors:
    - Forecast: Orange
    - Lower Bound: Red
    - Upper Bound: Green

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['date', 'forecast', 'lower_bound', 'upper_bound']

    Returns:
        Altair Chart: Line chart visualization with value labels
    """
    # Ensure 'date' is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Melt dataframe for Altair multi-line plotting
    df_melted = df.melt(id_vars=["date"], value_vars=["forecast", "lower_bound", "upper_bound"],
                        var_name="Type", value_name="Value")

    # Custom color mapping
    color_mapping = {
        "forecast": "orange",
        "lower_bound": "red",
        "upper_bound": "green"
    }

    # Determine min and max values for Y-axis
    y_min = df_melted["Value"].min() * 0.95  # 5% margin below
    y_max = df_melted["Value"].max() * 1.05  # 5% margin above

    # Create the base line chart with custom colors
    line_chart = alt.Chart(df_melted).mark_line(point=True, strokeWidth=1).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("Value:Q", title="Price", scale=alt.Scale(domain=[y_min, y_max])),
        color=alt.Color("Type:N", scale=alt.Scale(domain=list(color_mapping.keys()),
                                                  range=list(color_mapping.values())),
                        title="Legend"),
        tooltip=["date:T", "Type:N", "Value:Q"]
    ).properties(
        title="Forecast vs Confidence Interval"
    )

    # Create text labels for the points on each line
    labels = alt.Chart(df_melted).mark_text(
        align='left', dx=5, dy=-5  # Offset labels slightly above points
    ).encode(
        x=alt.X("date:T"),
        y=alt.Y("Value:Q"),
        text=alt.Text("Value:Q", format=".2f"),  # Format numbers to 2 decimal places
        color=alt.Color("Type:N", scale=alt.Scale(domain=list(color_mapping.keys()),
                                                  range=list(color_mapping.values())))  # Match label color
    )

    # Combine the line chart with labels
    final_chart = (line_chart + labels).interactive()

    return final_chart

def compute_forecast_errors(actual, predicted):
    """Computes RMSE, MAE, and MAPE."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return rmse, mae, mape


def compute_model_selection_criteria(model_fit):
    """Computes AIC and BIC for model selection."""
    return model_fit.aic, model_fit.bic


def compute_residual_diagnostics(model_fit):
    """Computes residual statistics (Mean, Std Dev, Min, Max)."""
    residuals = model_fit.resid.dropna()
    return {
        "Mean": residuals.mean(),
        "Std Dev": residuals.std(),
        "Min": residuals.min(),
        "Max": residuals.max()
    }


def compute_prediction_intervals(model_fit, asset_data):
    """Generates forecast with confidence intervals."""
    forecast_values, lower_bounds, upper_bounds = [], [], []
    forecast_dates = pd.date_range(start=asset_data.index[-1] + pd.Timedelta(days=1), periods=5)

    for i in range(5):
        forecast_result = model_fit.get_forecast(steps=1)
        conf_int = forecast_result.conf_int()
        forecast_mean = forecast_result.predicted_mean.iloc[0] if isinstance(forecast_result.predicted_mean, pd.Series) else forecast_result.predicted_mean
        lower_bound, upper_bound = (conf_int.iloc[0, 0], conf_int.iloc[0, 1]) if isinstance(conf_int, pd.DataFrame) and conf_int.shape[1] > 1 else (conf_int[0], conf_int[0])

        forecast_values.append(forecast_mean)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    return pd.DataFrame({
        "date": forecast_dates,
        "forecast": forecast_values,
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds
    })


def check_heteroskedasticity(model_fit):
    """Checks for heteroskedasticity using the ARCH test."""
    residuals = model_fit.resid.dropna()
    test_result = het_arch(residuals)
    return test_result[1]  # p-value


def compute_box_cox_lambda(model_fit):
    """Computes Box-Cox transformation lambda if applicable."""
    return model_fit.model.endog.mean() if hasattr(model_fit.model, "endog") else None


def compute_gls_log_likelihood(model_fit):
    """Computes GLS log-likelihood if applicable."""
    return model_fit.llf if hasattr(model_fit, "llf") else None

def garima_forecast_past(df, asset_code):
    """
    Fits a GARIMA model on the closing price of a given asset and forecasts the next 5 days.
    Uses a modular approach to compute key metrics dynamically.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['asset_code', 'date', 'close', 'volume']
        asset_code (str): Asset code to filter the data

    Returns:
        pd.DataFrame: Best model parameters, metrics, and forecasted values.
        model_fit: Fitted GARIMA model.
    """
    # Filter data for the specified asset
    asset_data = df[df['asset_code'] == asset_code].copy()
    asset_data['date'] = pd.to_datetime(asset_data['date'])
    asset_data = asset_data.sort_values('date')
    asset_data.set_index('date', inplace=True)

    # Use last 6 months of data for short-term forecasting
    six_months_ago = asset_data.index.max() - pd.DateOffset(months=6)
    asset_data = asset_data[asset_data.index >= six_months_ago]

    # Handle missing values
    asset_data['close'].fillna(method='ffill', inplace=True)
    asset_data.dropna(subset=['close'], inplace=True)

    if len(asset_data) < 15:
        return pd.DataFrame([{"error": "Not enough data for asset_code: " + asset_code}])

    # GARIMA Parameter Grid
    p_values = range(0, 3)
    d_values = [0, 1]
    q_values = range(0, 3)
    best_rmse, best_cfg, best_model, best_aic, best_bic = float("inf"), None, None, float("inf"), float("inf")

    param_grid = list(itertools.product(p_values, d_values, q_values))
    for params in param_grid:
        try:
            model = SARIMAX(asset_data['close'], order=params, enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            predictions = model_fit.fittedvalues

            # Compute Forecast Errors
            rmse, mae, mape = compute_forecast_errors(asset_data['close'], predictions)

            # Compute Model Selection Criteria
            aic, bic = compute_model_selection_criteria(model_fit)

            if rmse < best_rmse or (rmse == best_rmse and aic < best_aic):
                best_rmse, best_cfg, best_model, best_aic, best_bic = rmse, params, model_fit, aic, bic
        except:
            continue

    if not best_model:
        return pd.DataFrame([{"error": "No valid GARIMA model found."}])

    # Compute Residual Diagnostics
    residual_stats = compute_residual_diagnostics(best_model)

    # Compute Heteroskedasticity Test
    heteroskedasticity_pvalue = check_heteroskedasticity(best_model)

    # Compute Box-Cox Transformation (if applicable)
    box_cox_lambda = compute_box_cox_lambda(best_model)

    # Compute GLS Log-Likelihood (if applicable)
    gls_log_likelihood = compute_gls_log_likelihood(best_model)

    # Forecast Next 5 Days
    forecast_df = compute_prediction_intervals(best_model, asset_data)

    # Create Results DataFrame
    results_df = pd.DataFrame({
        "best_model_order": [best_cfg],
        "rmse": [best_rmse],
        "mae": [mae],
        "mape": [mape],
        "aic": [best_aic],
        "bic": [best_bic],
        "heteroskedasticity_pvalue": [heteroskedasticity_pvalue],
        "box_cox_lambda": [box_cox_lambda],
        "gls_log_likelihood": [gls_log_likelihood]
    })

    return results_df, forecast_df, best_model

def garima_forecast(df, asset_code):
    """
    Fits a dynamically optimized GARIMA model on the closing price of a given asset and forecasts the next 5 days.
    The function adjusts for stationarity issues, heteroskedasticity, and other residual characteristics
    to enhance forecasting accuracy.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['asset_code', 'date', 'close', 'volume']
        asset_code (str): Asset code to filter the data

    Returns:
        results_df (pd.DataFrame): DataFrame containing all key model evaluation metrics.
        forecast_df (pd.DataFrame): Forecasted values for the next 5 days.
        refined_model: The best-fitted GARIMA model.
    """

    # Filter data for the specified asset
    asset_data = df[df['asset_code'] == asset_code].copy()
    asset_data['date'] = pd.to_datetime(asset_data['date'])
    asset_data = asset_data.sort_values('date')
    asset_data.set_index('date', inplace=True)

    # Use last 6 months of data
    six_months_ago = asset_data.index.max() - pd.DateOffset(months=6)
    asset_data = asset_data[asset_data.index >= six_months_ago]

    # Handle missing values
    asset_data['close'].fillna(method='ffill', inplace=True)
    asset_data.dropna(subset=['close'], inplace=True)

    # Ensure sufficient data points
    if len(asset_data) < 15:
        return pd.DataFrame([{"error": "Not enough data for asset_code: " + asset_code}])

    # Step 1: Check for Stationarity and Adjust Differencing (d)
    def check_stationarity(timeseries):
        """ Returns stationarity status and optimal differencing (d) """
        result = adfuller(timeseries)
        p_value = result[1]
        return p_value, 0 if p_value < 0.05 else 1  # If non-stationary, set d=1

    stationarity_pvalue, optimal_d = check_stationarity(asset_data['close'])

    # Step 2: GARIMA parameter grid with dynamic differencing
    p_values = range(0, 3)
    d_values = [optimal_d]  # Only try optimal differencing level
    q_values = range(0, 3)
    best_rmse, best_mae, best_mape, best_cfg, best_model = float("inf"), float("inf"), float("inf"), None, None
    best_aic, best_bic, best_heteroskedasticity = float("inf"), float("inf"), None

    param_grid = list(itertools.product(p_values, d_values, q_values))
    for params in param_grid:
        try:
            model = ARIMA(asset_data['close'], order=params)
            model_fit = model.fit()
            predictions = model_fit.fittedvalues
            rmse = np.sqrt(mean_squared_error(asset_data['close'], predictions))
            mae = mean_absolute_error(asset_data['close'], predictions)
            mape = np.mean(np.abs((asset_data['close'] - predictions) / asset_data['close'])) * 100
            aic = model_fit.aic
            bic = model_fit.bic

            if rmse < best_rmse or (rmse == best_rmse and aic < best_aic):
                best_rmse, best_mae, best_mape, best_cfg = rmse, mae, mape, params
                best_model, best_aic, best_bic = model_fit, aic, bic
        except:
            continue

    if not best_model:
        return pd.DataFrame([{"error": "No valid GARIMA model found."}])

    # Step 3: Perform Residual Diagnostics and Refine Model
    residual_stats, hist_chart, line_chart, boxplot, refined_model = garima_residuals(best_model)

    # Step 4: Check for Heteroskedasticity (ARCH effect)
    def check_heteroskedasticity(model_fit):
        """Returns heteroskedasticity p-value (ARCH test)"""
        residuals = model_fit.resid.dropna()
        if len(residuals) < 15:
            return None  # Not enough residuals to test

        test_result = het_arch(residuals)
        return test_result[1]  # P-value for heteroskedasticity

    best_heteroskedasticity = check_heteroskedasticity(refined_model)

    # Step 5: Generate Forecast Based on Residual Analysis
    forecast_result = refined_model.get_forecast(steps=5)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    forecast_df = pd.DataFrame({
        "date": pd.date_range(start=asset_data.index[-1] + pd.Timedelta(days=1), periods=5),
        "forecast": forecast_mean.values,
        "lower_bound": conf_int.iloc[:, 0].values,
        "upper_bound": conf_int.iloc[:, 1].values
    })

    # Store all key metrics in results_df
    results_df = pd.DataFrame({
        "best_model_order": [best_cfg],
        "rmse": [best_rmse],
        "mae": [best_mae],
        "mape": [best_mape],
        "aic": [best_aic],
        "bic": [best_bic],
        "stationarity_pvalue": [stationarity_pvalue],  # ADF Test result
        "heteroskedasticity_pvalue": [best_heteroskedasticity]  # ARCH Test result
    })

    return results_df, forecast_df, refined_model