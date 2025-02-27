import streamlit as st
from modules.a2init_session_variables import *
a2init_session_variables()

st.session_state.a2html_str_exp_smoothing_legends = ' '
st.session_state.a2html_str_exp_smoothing_legends += """
<div style="display: flex; gap: 20px; align-items: center; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9; width: fit-content;">
    <div style="display: flex; align-items: center;">
        <div style="width: 15px; height: 15px; background-color: blue; margin-right: 5px;"></div>
        <span>Actual Close</span>
    </div>
    <div style="display: flex; align-items: center;">
        <div style="width: 15px; height: 15px; background-color: green; margin-right: 5px;"></div>
        <span>Smoothed Close</span>
    </div>
    <div style="display: flex; align-items: center;">
        <div style="width: 15px; height: 15px; border: 2px dashed purple; margin-right: 5px;"></div>
        <span>Forecasted Close</span>
    </div>
</div>
"""

st.session_state.a2html_str_exp_smoothing = ' '
st.session_state.a2html_str_exp_smoothing += """
<div style="
    font-family: Calibri, sans-serif; 
    font-size: 18px; 
    line-height: 1.2; 
    border: 1px solid #4CAF50; /* Green border */
    padding: 25px; 
    border-radius: 12px; 
    background-color: #f9f9f9;
    box-shadow: inset 4px 4px 8px rgba(0, 0, 0, 0.1); /* Subtle inner shadow */
">
    <ul>
        <li>
            <strong>Exponential Smoothing (ES)</strong> is a time series forecasting technique that 
            <strong>assigns exponentially decreasing weights to past observations</strong> to predict 
            future stock prices. It is useful for capturing trends and patterns while reducing noise 
            in stock price movements. 
        </li>
    </ul>
    <ul>
        <li>
            It works well with short term forecasting 🚀, more weightage to recent asset prices 📊 
            and computationally efficient ⚡.
        </li>
    </ul>
    
    <ul>
        <li>
            It comes in three main types:
            <ul>
                <li><em>Simple Exponential Smoothing (SES)</em> → Best for data <strong>without trends or seasonality</strong>.</li>
                <li><em>Double Exponential Smoothing (Holt’s Method)</em> → Suitable for data with <strong>trends</strong>.</li>
                <li><em>Triple Exponential Smoothing (Holt-Winters Method)</em> → Best for data with <strong>trends & seasonality</strong>.</li>
            </ul>
        </li>
    </ul>
</div>


"""



st.session_state.a2html_str_outer_box = ' '
st.session_state.a2html_str_outer_box += """
    <style>
        .outer-box {
            border: 1.2 px solid black;
            padding: 20px;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
    </style>
"""

st.session_state.a2html_str_de_smoothing = ' '
st.session_state.a2html_str_de_smoothing +="""
<div style="
    font-family: Calibri, sans-serif; 
    font-size: 18px; 
    line-height: 1.4; 
    border: 1px solid #4CAF50; /* Green border */
    padding: 25px; 
    border-radius: 12px; 
    background-color: #f9f9f9;
    box-shadow: inset 4px 4px 8px rgba(0, 0, 0, 0.1); /* Subtle inner shadow */
">
    <ul>
        <li>
            <strong>📌 Double Exponential Smoothing (Holt’s Exponential Smoothing)</strong> is an 
            extension of <strong>Simple Exponential Smoothing (SES)</strong> that accounts for both 
            <strong>trend</strong> and <strong>level</strong> in time series forecasting.
        </li>
    </ul>
    <ul>
        <li>
            <strong>How It Works:</strong>
            <ul>
                <li>Introduces an additional smoothing factor to model <strong>trends</strong>.</li>
                <li>Uses two key parameters:
                    <ul>
                        <li><strong>α (alpha)</strong> → Controls how much weight is given to recent observations.</li>
                        <li><strong>β (beta)</strong> → Controls how much trend is adjusted over time.</li>
                    </ul>
                </li>
            </ul>
        </li>
    </ul>
    <ul>
        <li>
            <strong>✅ Advantages of Double Exponential Smoothing:</strong>
            <ul>
                <li>📈 <strong>Captures Trends Efficiently:</strong> Unlike SES, Holt’s method accounts for 
                    <strong>uptrends and downtrends</strong> in stock prices.</li>
                <li>📊 <strong>Better for Short- to Medium-Term Forecasting:</strong> Works well for 
                    <strong>trending stocks</strong> and is more responsive than SMA or CMA.</li>
                <li>🔄 <strong>Smoother Than Moving Averages:</strong> Reduces noise while still adapting to 
                    <strong>trending stock movements</strong>.</li>
            </ul>
        </li>
    </ul>
    <ul>
        <li>
            <strong>❌ Limitations of Double Exponential Smoothing:</strong>
            <ul>
                <li>⚠ <strong>Fails to Capture Seasonality:</strong> If stock prices fluctuate cyclically 
                    (e.g., earnings season, business cycles), Holt’s method alone won’t be enough.</li>
                <li>🔄 <strong>Trend Overfitting Risk:</strong> In highly volatile stocks, it may 
                    <strong>incorrectly extrapolate short-term trends</strong>.</li>
                <li>⚡ <strong>Less Effective in Noisy Markets:</strong> May overreact to short-term fluctuations 
                    if a stock lacks a strong trend.</li>
            </ul>
        </li>
    </ul>
    <ul>
        <li>
            <strong>📌 When to Use Double Exponential Smoothing?</strong>
            <ul>
                <li>✔ <strong>Use it when</strong> stock prices show a <strong>clear trend</strong> 
                    (bullish/bearish momentum).</li>
                <li>❌ <strong>Avoid using it when</strong> prices fluctuate heavily without a clear direction.</li>
            </ul>
        </li>
    </ul>
</div>

"""

st.session_state.a2html_str_se_smoothing = ' '
st.session_state.a2html_str_se_smoothing +="""
<div style="
    font-family: Calibri, sans-serif; 
    font-size: 18px; 
    line-height: 1.4; 
    border: 1px solid #4CAF50; /* Green border */
    padding: 25px; 
    border-radius: 12px; 
    background-color: #f9f9f9;
    box-shadow: inset 4px 4px 8px rgba(0, 0, 0, 0.1); /* Subtle inner shadow */
">
    <ul>
        <li>
            <strong>Simple Exponential Smoothing (SES)</strong> is a time series forecasting method that 
            <strong>assigns exponentially decreasing weights</strong> to past observations, meaning 
            <strong>recent data points have more influence</strong> on the forecast than older data. It is useful for 
            <strong>predicting stock prices</strong> when there is <strong>no clear trend or seasonality</strong> in the data.
        </li>
    </ul>
    <ul>
        <li>
            <strong>Alpha (α)</strong> is the smoothing parameter that controls how much weight recent data has:
            <ul>
                <li><strong>Higher α</strong> → Reacts quickly to new data, making forecasts sensitive to recent price changes.</li>
                <li><strong>Lower α</strong> → More weight on past values, resulting in a smoother but slower response.</li>
            </ul>
        </li>
    </ul>
    <ul>
        <li>
            <strong>Advantages of SES</strong>:
            <ul>
                <li>✅ Captures short-term price movements by emphasizing recent data.</li>
                <li>✅ Simple to implement with only one parameter (α).</li>
                <li>✅ Works well for stocks with no clear trend or seasonality.</li>
            </ul>
        </li>
    </ul>
    <ul>
        <li>
            <strong>Limitations of SES</strong>:
            <ul>
                <li>❌ Does not handle trends or seasonality—ineffective if prices follow an upward or downward trend.</li>
                <li>❌ Can be slow to adapt to sudden market shifts, leading to lagging predictions.</li>
                <li>❌ High α values may cause excessive sensitivity to noise, making forecasts erratic.</li>
            </ul>
        </li>
    </ul>
    <ul>
        <li>
            <strong>When to Use SES:</strong>
            <ul>
                <li>✔ Best for <strong>stable stocks</strong> that lack a clear trend.</li>
                <li>✔ Suitable for <strong>short-term forecasting</strong> of stocks with <strong>low volatility</strong>.</li>
                <li>✔ <strong>Not recommended for trending stocks</strong>; use Holt’s method instead.</li>
            </ul>
        </li>
    </ul>
</div>
"""

st.session_state.a2html_str_daily_dist_percent = ' '
st.session_state.a2html_str_daily_dist_percent +="""
<div style="border: 1.2px solid green; border-radius: 12px; 
    background: linear-gradient(to bottom, #e6f9e6, #ffffff); padding: 10px; 
    box-shadow: 4px 4px 10px rgba(0, 128, 0, 0.2); font-family: Calibri, sans-serif; font-size: 15px;">
    <p data-start="70" data-end="276">Charts showing <strong data-start="85" data-end="112">Daily Percentage Price Change</strong> of stocks help track relative performance rather than absolute price movements. This allows for a fair comparison of stocks, regardless of their price differences.</p>    
    <h4 style="color: green; font-weight: bold;">📊 Distribution of Daily Percentage Changes</h4>
    <p>A <strong>distribution plot</strong> of <strong>daily percentage changes</strong> (log returns or simple percentage changes) helps identify:</p>
    <ul style="padding-left: 20px;">
        <li>✅ <strong>Volatility</strong> → Wide spread = Higher volatility, narrow spread = Lower volatility</li>
        <li>✅ <strong>Skewness</strong> → Asymmetry in distribution (right skew = more gains, left skew = more losses)</li>
        <li>✅ <strong>Kurtosis (Tails)</strong> → Fat tails = Higher risk of extreme events (e.g., crashes, surges)</li>
        <li>✅ <strong>Mean & Median</strong> → The central tendency of daily returns</li>
    </ul>
</div>
"""

st.session_state.a2html_str_pearson = ' '
st.session_state.a2html_str_pearson +="""
<div style="
    font-family: Calibri, sans-serif; 
    font-size: 18px; 
    line-height: 1.2; 
    border: 1px solid #4CAF50; /* Green border */
    padding: 25px; 
    border-radius: 12px; 
    background-color: #f9f9f9;
    box-shadow: inset 4px 4px 8px rgba(0, 0, 0, 0.1); /* Subtle inner shadow */
">
    <ul>
        <li>
            <strong>Pearson correlation</strong> helps measure the <strong>linear relationship</strong> between stock prices, 
            showing how closely two stocks move together, with values ranging from <strong>-1 (inverse movement)</strong> to 
            <strong>+1 (similar movement)</strong>, and <strong>0 meaning no correlation</strong>. It works best when stock 
            returns follow a <strong>consistent trend</strong>, but it can be influenced by <strong>outliers or sudden market 
            shocks</strong>, making it more suitable for stable market conditions.
        </li>
    </ul>
    <ul>
        <li>
            <strong>Spearman correlation</strong> helps identify whether stocks move in the same relative direction, 
            even if their relationship is nonlinear, making it useful for detecting broader trends and ranking-based 
            dependencies over a long period of time. Unlike Pearson correlation, Spearman is less sensitive to outliers 
            and works well when stock relationships are not strictly linear, but long-term shifts in market conditions, 
            industry trends, and external factors can still impact its stability over time.
        </li>
    </ul>
</div>
"""

st.session_state.a2html_str_eda = """
<div style="
    font-family: Calibri, sans-serif; 
    font-size: 18px; 
    line-height: 1.2; 
    border: 1px solid #800080; /* Purple border with reduced thickness */
    padding: 25px; 
    border-radius: 12px; 
    background-color: #f9f9f9;
    box-shadow: inset 4px 4px 8px rgba(128, 0, 128, 0.1); /* Subtle purple inner shadow */
">
    Exploratory Data Analysis focus on Time-series plots of stock prices and charts provide insights into market trends.  
    Also helpful with Heatmaps of correlation matrices and volatility distributions help in identifying patterns 
    that influence trading strategies and risk management.
</div>
"""

st.session_state.a2html_str_adf = ' '
st.session_state.a2html_str_adf += """
<div style="
    font-family: Calibri; 
    font-size: 18px; 
    line-height: 1.2; 
    border: 1px solid #4CAF50; 
    padding: 25px; 
    border-radius: 12px; 
    background-color: #f9f9f9;
    box-shadow: inset 5px 5px 10px rgba(0, 0, 0, 0.1);
">
    <ul>
        <li><strong>Stationarity in time series</strong> means that <strong>mean, variance, and autocorrelation</strong> remain 
        <strong>constant over time</strong>, making the data suitable for forecasting models.</li>
        
        <li><strong>The Augmented Dickey-Fuller (ADF) test</strong> checks stationarity by testing whether a time series has a 
        <strong>unit root</strong> (non-stationary) or not.</li>
        
        <li><strong>If p-value &lt; 0.05</strong>, the series is <strong>stationary</strong> (reject H₀); otherwise, it is 
        <strong>non-stationary</strong>, requiring transformations like <strong>differencing or log transformation</strong>.</li>
        
        <li>Most time series models (ARIMA, SARIMA, etc.) <strong>assume stationarity</strong> to make accurate predictions. 
        If a time series is <strong>non-stationary</strong>, we need to <strong>transform it</strong> 
        (e.g., differencing, log transformation) before modeling.</li>
    </ul>
</div>
"""

st.session_state.a2html_str_covariance = ' '
st.session_state.a2html_str_covariance += """
    <style>
        .green-box {
            font-family: Calibri; 
            font-size: 18px; 
            line-height: 1.2; 
            border: 1px solid #4CAF50; 
            padding: 25px; 
            border-radius: 12px; 
            background-color: #f9f9f9;
            box-shadow: inset 5px 5px 10px rgba(0, 0, 0, 0.1);
        }
        .bold-text {
            font-weight: bold;
            color: #2e7d32;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            padding: 10px;
            border: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
    </style>

    <div class="green-box">
        <h3>📊 Understanding Covariance in Stocks</h3>
        <p>Covariance measures the relationship between two stocks (or variables). It shows how two stocks move together—whether their prices tend to increase and decrease together (<span class="bold-text">positive covariance</span>) or move in opposite directions (<span class="bold-text">negative covariance</span>).</p>

        <h3>🏆 Key Points About Covariance</h3>

        <h4>📈 Positive Covariance</h4>
        <ul>
            <li>If <b>Stock A goes up</b> and <b>Stock B also goes up</b>, covariance is <span class="bold-text">positive</span>.</li>
            <li><b>Example:</b> AAPL & MSFT might have positive covariance because both are tech stocks and tend to move similarly.</li>
        </ul>

        <h4>📈📉 Negative Covariance</h4>
        <ul>
            <li>If <b>Stock A goes up</b> but <b>Stock B goes down</b>, covariance is <span class="bold-text">negative</span>.</li>
            <li><b>Example:</b> A defensive stock (<b>Gold ETF</b>) vs. a growth stock (<b>Tesla</b>) may have negative covariance.</li>
        </ul>

        <h4>⚖️ Zero Covariance</h4>
        <ul>
            <li>If two stocks move <b>independently</b>, their covariance is <span class="bold-text">near zero</span>.</li>
        </ul>

        <h3>🚀 Quick Summary</h3>
        <table>
            <tr>
                <th>Covariance Value</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>📈📈 <b>Positive</b></td>
                <td>Stocks move together (same direction)</td>
            </tr>
            <tr>
                <td>📈📉 <b>Negative</b></td>
                <td>Stocks move in opposite directions</td>
            </tr>
            <tr>
                <td>⚖️ <b>Near Zero</b></td>
                <td>Stocks move independently</td>
            </tr>
        </table>
    </div>
"""
st.session_state.a2html_eigenvector_str = ' '
st.session_state.a2html_eigenvector_str += """
    <style>
        .green-box {
            font-family: Calibri; 
            font-size: 18px; 
            line-height: 1.2; 
            border: 1px solid #4CAF50; 
            padding: 25px; 
            border-radius: 12px; 
            background-color: #f9f9f9;
            box-shadow: inset 5px 5px 10px rgba(0, 0, 0, 0.1);
        }
        .bold-text {
            font-weight: bold;
            color: #2e7d32;
        }
    </style>
    
    <div class="green-box" >
        <div class="bold-text"> Eigenvectors & Asset Movements</div>
        <div class="bold-text"> ************************************************************************************* </div>
        <ul>
            <li><span class="bold-text">Positive or negative values</span> in eigenvectors show how assets move relative to each other.</li>
            <li>If <span class="bold-text">two assets have the same sign</span> in an eigenvector, they tend to move together.</li>
            <li>If <span class="bold-text">one is positive and the other is negative</span>, they move in opposite directions.</li>
        </ul>
    </div>
"""

st.session_state.a2html_log_trans_str = ' '
st.session_state.a2html_log_trans_str += """
    <style>
        .green-box {
            font-family: Calibri; 
            font-size: 18px; 
            line-height: 1.2; 
            border: 1px solid #4CAF50; 
            padding: 25px; 
            border-radius: 12px; 
            background-color: #f9f9f9;
            box-shadow: inset 5px 5px 10px rgba(0, 0, 0, 0.1);
        }
        .bold-text {
            font-weight: bold;
            color: #2E7D32;  /* Dark green for emphasis */
        }
        h3, h4 {
            color: #1B5E20;  /* Darker green for headings */
        }
        ul {
            padding-left: 20px;
        }
    </style>

    <div class="green-box">
        <h4>📌 Log Transformation & Its Importance in Asset Prices</h4>

        <h4>🔹What is Log Transformation?</h4>
        <p>Log transformation is the process of applying the <span class="bold-text">natural logarithm (log base e)</span> to a dataset.</p>

        <h4>1️⃣ Converts Multiplicative Changes into Additive Changes</h4>
        <ul>
            <li>Financial data (like stock prices) usually grows <span class="bold-text">multiplicatively</span> rather than <span class="bold-text">additively</span>.</li>
            <li><span class="bold-text">Example:</span> A stock price moving from <span class="bold-text">100 to 110</span> (+10%) is not the same as from <span class="bold-text">200 to 210</span> (+5%).</li>
            <li>Taking logs converts <span class="bold-text">percentage changes</span> into <span class="bold-text">differences</span>, making analysis easier.</li>
            <li>This makes log returns approximately <span class="bold-text">equal to percentage changes</span> for small values.</li>
        </ul>

        <h4>2️⃣ Normalizes Skewed Data (Reduces Volatility Effects)</h4>
        <ul>
            <li>Asset prices tend to be <span class="bold-text">positively skewed</span> (long tails on the right).</li>
            <li>Log transformation <span class="bold-text">compresses large values</span>, making the data <span class="bold-text">more symmetric</span>.</li>
            <li>This is useful when applying <span class="bold-text">statistical models</span> that assume <span class="bold-text">normally distributed data</span>.</li>
        </ul>

        <h4>3️⃣ Comparability Across Different Assets</h4>
        <ul>
            <li>Different assets have different price ranges (<span class="bold-text">$10 stock vs. $3000 stock</span>).</li>
            <li>Applying log transformation allows <span class="bold-text">fair comparison</span> across assets by analyzing <span class="bold-text">relative changes</span> instead of absolute prices.</li>
        </ul>

        <h4>4️⃣ When to Use Log Transformation in Asset Prices?</h4>
        <p style="padding-left: 20px;">
            ✅ When dealing with <span class="bold-text">long-term trends</span> in financial time series.<br>
            ✅ When analyzing <span class="bold-text">returns instead of raw prices</span>.<br>
            ✅ When data is <span class="bold-text">positively skewed</span> and needs normalization.<br>
            ✅ When performing <span class="bold-text">econometric modeling</span> like regression.
        </p>
    </div>
"""

st.session_state.a2html_sma_str = ' '
st.session_state.a2html_sma_str += """
    <style>
        .green-box {
            font-family: Calibri; 
            font-size: 18px; 
            line-height: 1.2; 
            border: 1px solid #4CAF50; 
            padding: 25px; 
            border-radius: 12px; 
            background-color: #f9f9f9;
            box-shadow: inset 5px 5px 10px rgba(0, 0, 0, 0.1);
        }
        .bold-text {
            font-weight: bold;
            color: #2E7D32;  /* Dark green for emphasis */
        }
        .underline {
            text-decoration: underline;
            font-weight: bold;
            color: #1B5E20;
        }
        ul {
            padding-left: 20px;
        }
    </style>

    <div class="green-box">
        <p><span class="bold-text">Simple Moving Average (SMA) - </span>
        is the average closing price of a stock over a specific period (e.g., 50-day or 200-day). It helps smooth out short-term fluctuations to identify trends.</p>

        <ul>
            <li><span class="underline">Advantage in comparing multiple stocks:</span>
                SMA allows you to compare trends across different stocks, helping identify which stocks are in an uptrend, downtrend, or consolidation phase. 
                It also helps detect crossovers, which can indicate potential buy/sell signals.
            </li>
            <li><span class="underline">Drawback:</span> SMA is a lagging indicator, meaning it reacts to price changes with a delay. 
                This can make it less useful for short-term trading and slow to respond to sudden market shifts.
            </li>
        </ul>
    </div>
"""

st.session_state.a2html_cma_str = ' '
st.session_state.a2html_cma_str += """
    <style>
        .green-box {
            font-family: Calibri; 
            font-size: 18px; 
            line-height: 1.2; 
            border: 1px solid #4CAF50; 
            padding: 25px; 
            border-radius: 12px; 
            background-color: #f9f9f9;
            box-shadow: inset 5px 5px 10px rgba(0, 0, 0, 0.1);
        }
        .bold-text {
            font-weight: bold;
            color: #2E7D32;  /* Dark green for emphasis */
        }
        .underline {
            text-decoration: underline;
            font-weight: bold;
            color: #1B5E20;
        }
        ul {
            padding-left: 20px;
        }
    </style>

    <div class="green-box">
        <p><span class="bold-text">Cumulative Moving Average (CMA) - </span>
        calculates the average price of a stock from the beginning of a dataset to each point in time, 
        smoothing fluctuations by incorporating all past data.</p>

        <p class="underline">Advantages in Comparing Multiple Stocks:</p>
        <ul>
            <li>Helps in identifying long-term trends by reducing short-term noise.</li>
            <li>Provides a clear overall performance comparison across different stocks.</li>
            <li>Useful for detecting steady growth or decline over extended periods.</li>
        </ul>

        <p class="underline">Drawbacks:</p>
        <ul>
            <li>Slow to react to recent price changes since older data heavily influences the average.</li>
            <li>Less effective for short-term trend analysis or volatile stocks.</li>
            <li>Can mislead in comparing stocks with different volatility levels, as sudden changes have less immediate impact.</li>
        </ul>
    </div>
"""

st.session_state.a2html_ema_str = ' '
st.session_state.a2html_ema_str += """
    <style>
        .green-box {
            font-family: Calibri; 
            font-size: 18px; 
            line-height: 1.2; 
            border: 1px solid #4CAF50; 
            padding: 25px; 
            border-radius: 12px; 
            background-color: #f9f9f9;
            box-shadow: inset 5px 5px 10px rgba(0, 0, 0, 0.1);
        }
        .bold-text {
            font-weight: bold;
            color: #2E7D32;  /* Dark green for emphasis */
        }
        .underline {
            text-decoration: underline;
            font-weight: bold;
            color: #1B5E20;
        }
        ul {
            padding-left: 20px;
        }
    </style>

    <div class="green-box">
        <p><strong style="color: #006400;" data-start="47" data-end="83">Exponential Moving Average (EMA)</strong>&nbsp;gives <strong style="color: #006400;" data-start="123" data-end="160">more weight to recent data points</strong>, making it more responsive to price changes compared to a Simple Moving Average (SMA). It is widely used in <strong style="color: #006400;" data-start="269" data-end="291">technical analysis</strong> for trend identification and trading strategies.</p>
        <p><span style="text-decoration: underline;">Advantages</span></p>
        <ul data-start="370" data-end="758">
            <li data-start="370" data-end="505"><em>More Responsive to Price Changes</em> &ndash; Since it gives higher weight to recent prices, it reacts faster to new trends and reversals.</li>
            <li data-start="506" data-end="635"><em>Reduces Lag Compared to SMA</em> &ndash; Unlike SMA, which equally weights all data points, EMA adjusts quickly to market movements.</li>
            <li data-start="636" data-end="758"><em>Useful for Short-Term Traders</em> &ndash; Ideal for day traders and swing traders who need quick signals for entry and exit.</li>
        </ul>
        <p><span style="text-decoration: underline;">Drawbacks</span></p>
        <ul data-start="787" data-end="1219">
            <li data-start="787" data-end="935"><em>Can Generate False Signals</em> &ndash; Its sensitivity to recent price changes may cause it to react to short-term volatility rather than true trends.</li>
            <li data-start="936" data-end="1055"><em>Less Effective in Choppy Markets</em> &ndash; In sideways or highly volatile markets, EMA may result in frequent whipsaws.</li>
            <li data-start="1056" data-end="1219"><em>Requires Optimization</em> &ndash; Choosing the right time period (e.g., 10-day, 50-day, or 200-day EMA) is crucial for effectiveness and varies by asset or strategy.</li>
        </ul>
    </div>
"""

st.session_state.a2html_ewma_str = ' '
st.session_state.a2html_ewma_str += """
    <style>
        .green-box {
            font-family: Calibri; 
            font-size: 18px; 
            line-height: 1.2; 
            border: 1px solid #4CAF50; 
            padding: 25px; 
            border-radius: 12px; 
            background-color: #f9f9f9;
            box-shadow: inset 5px 5px 10px rgba(0, 0, 0, 0.1);
        }
        .bold-text {
            font-weight: bold;
            color: #2E7D32;  /* Dark green for emphasis */
        }
        .underline {
            text-decoration: underline;
            font-weight: bold;
            color: #1B5E20;
        }
        ul {
            padding-left: 20px;
        }
    </style>

    <div class="green-box">
        <p><span class="bold-text">Exponential Weighted Moving Average (EWMA) - </span>
        gives more weight to recent stock prices while gradually decreasing the weight of older prices, 
        making it more responsive to recent changes.</p>

        <p class="underline">Advantages in Comparing Multiple Stocks:</p>
        <ul>
            <li>Reacts quickly to price fluctuations, making it useful for detecting short-term trends.</li>
            <li>Helps compare momentum shifts between stocks more effectively than simple moving averages.</li>
            <li>Useful for risk management, as it highlights sudden changes in volatility.</li>
        </ul>

        <p class="underline">Drawbacks:</p>
        <ul>
            <li>Can be too sensitive to short-term noise, leading to false trend signals.</li>
            <li>Requires choosing an appropriate smoothing factor (alpha), which may differ for each stock.</li>
            <li>May not be as effective for long-term trend analysis compared to cumulative or simple moving averages.</li>
        </ul>
    </div>
"""

st.session_state.a2html_50days_ma_str = ' '
st.session_state.a2html_50days_ma_str += """
    <style>
        .green-box {
            font-family: Calibri; 
            font-size: 10px; 
            line-height: 1.2; 
            border: 1px solid #4CAF50; 
            padding: 15px; 
            border-radius: 12px; 
            background-color: #f9f9f9;
            box-shadow: inset 5px 5px 10px rgba(0, 0, 0, 0.1);
        }
        .bold-text {
            font-weight: bold;
            color: #2E7D32;  /* Dark green for emphasis */
        }
        .underline {
            text-decoration: underline;
            font-weight: bold;
            color: #1B5E20;
        }
        ul {
            padding-left: 20px;
        }
    </style>

    <div class="green-box">
        <h4 data-start="1476" data-end="1501"><strong data-start="164" data-end="190">Short-Term: (15 days or 21 days or 50 days Moving Average)</strong></h4>
        <ul>
        <li>Best for short-term traders (swing trading)</li>
        <li>Reacts quickly to price changes but is prone to noise and false signals.</li>
        <li>Useful for identifying short-term momentum shifts.</li>
        </ul>
        <p><strong data-start="164" data-end="190">Medium-Term: (50 days or 100 days Moving Average)</strong></p>
        <ul>
        <li>Balanced approach for trend identification.</li>
        <li>Used by traders and investors to confirm medium-term trends.</li>
        <li>The <strong data-start="597" data-end="610">50-day MA</strong> is commonly used for support/resistance levels.</li>
        </ul>
        <p><strong data-start="164" data-end="190">Long-Term: (100 days / 200 days Moving Average)</strong></p>
        <ul>
        <li>Best for long-term trend analysis and investment decisions.</li>
        <li><strong data-start="815" data-end="829">200-day MA</strong> is widely used to define bull/bear markets.</li>
        <li>Less responsive to short-term price movements but filters out noise effectively.</li>
        </ul>
        <h4 data-start="1476" data-end="1501"><strong data-start="1481" data-end="1499">Best Strategy?</strong></h4>
        <p data-start="1502" data-end="1556">Using a <strong data-start="1510" data-end="1525">combination</strong> is often ideal. For example:</p>
        <ul data-start="1557" data-end="1739">
        <li data-start="1557" data-end="1648"><strong data-start="1559" data-end="1575">Golden Cross</strong> (Bullish Signal): When the <strong data-start="1603" data-end="1645">50-day MA crosses above the 200-day MA</strong>.</li>
        <li data-start="1649" data-end="1739"><strong data-start="1651" data-end="1666">Death Cross</strong> (Bearish Signal): When the <strong data-start="1694" data-end="1736">50-day MA crosses below the 200-day MA</strong>.</li>
        </ul>
    </div>
"""


@st.cache_data
def a2return_html_str(a2parameter: str):
    if a2parameter == 'sma':
        return st.session_state.a2html_sma_str

    if a2parameter == 'cma':
        return st.session_state.a2html_cma_str

    if a2parameter == 'ema':
        return st.session_state.a2html_ema_str

    if a2parameter == 'ewma':
        return st.session_state.a2html_ewma_str

    if a2parameter == 'sem': #simple exponential smoothing
        return st.session_state.a2html_str_se_smoothing

    if a2parameter == 'dem': #double exponential smoothing
        return st.session_state.a2html_str_de_smoothing

    if a2parameter == 'tem': #tripple exponential smoothing
        return st.session_state.a2html_ewma_str
