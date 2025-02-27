from datetime import date, datetime, timedelta
import json
import pandas as pd
from datetime import datetime
import yfinance as yf

from dateutil.relativedelta import relativedelta
import duckdb as dd

a2format = '%Y-%b-%d'
a2db = dd.connect("a2db.db")

#Check the process is run first time:

#Check the process is run first time:
jsonFile = open("a2config.json", "r") # Open the JSON file for reading
a2data_json_df = json.load(jsonFile) # Read the JSON into the buffer
jsonFile.close()

print(a2data_json_df['a2last_run_date'])


def a2load_ts_data_dr():
    asset_lists = ['equity','crypto','index','crypto_etf', 'mutual_fund' ]
    asset_meta_tbl = ''
    asset_ts_tbl = ''

    for asset_type in asset_lists:
        if asset_type == 'equity':
            asset_meta_tbl = 'a2equity_meta'
            asset_ts_tbl = 'a2equity_ts_data'
        elif asset_type == 'crypto':
            asset_meta_tbl = 'a2crypto_meta'
            asset_ts_tbl = 'a2crypto_ts_data'
        elif asset_type == 'index':
            asset_meta_tbl = 'a2index_meta'
            asset_ts_tbl = 'a2index_ts_data'
        elif asset_type == 'crypto_etf':
            asset_meta_tbl = 'a2crypto_etf_meta'
            asset_ts_tbl = 'a2crypto_etf_ts_data'
        elif asset_type == 'mutual_fund':
            asset_meta_tbl = 'a2mfi_meta'
            asset_ts_tbl = 'a2mfi_ts_data'

        tmp_str = "select asset_code from {0}".format(asset_meta_tbl)

        a2results_df = pd.DataFrame(a2db.execute(tmp_str).df())
        a2results_df.reset_index(inplace=True)

        for index, row in a2results_df.iterrows():
            ticker = row['asset_code']
            if ticker is None:
                continue

            tmp_sql = f"SELECT count(1) FROM a2master_data_vw WHERE asset_code='{ticker}'"
            if a2db.execute(tmp_sql).fetchall() == 0:
                continue

            tmp_sql = f"SELECT max(date) max_dt FROM a2master_data_vw WHERE asset_code='{ticker}'"
            startDt_df = a2db.execute(tmp_sql).df()
            startDt_df['max_dt'] = pd.to_datetime(startDt_df['max_dt'])

            # Ensure startDt is properly assigned, handling NaT values
            if startDt_df['max_dt'].isna().any():
                startDt = date.today() - relativedelta(years=5)
            else:
                startDt = startDt_df.iloc[0]['max_dt'].date() + timedelta(days=1)

            endDt = date.today()
            if startDt >= endDt:
                continue

            print(f"{asset_type} - Ticker - {ticker} being processed")
            a2data = yf.download(ticker, start=startDt, end=endDt, auto_adjust=False)

            a2data.reset_index(inplace=True)
            a2data.columns = a2data.columns.get_level_values(0)

            a2data.insert(0, "asset_code", ticker)
            a2data.insert(1, "asset_type", asset_type)

            if "Adj Close" not in a2data:
                a2data.insert(2, "Adj Close", 0)
            else:
                a2data["Close"] = a2data["Adj Close"]

            a2data.rename(columns={"Adj Close": "adj_close"}, inplace=True)
            a2data.columns = a2data.columns.str.lower()

            try:
                a2db.execute(f"INSERT INTO {asset_ts_tbl} BY NAME SELECT * FROM a2data")
            except Exception as e:
                print(f"Error inserting data for {ticker}: {e}")