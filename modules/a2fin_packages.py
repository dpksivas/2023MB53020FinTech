import pandas as pd
import duckdb as dd
import streamlit as st

a2db = dd.connect("a2db.db", read_only=True)

@st.cache_data(ttl=3*60*60)
def a2asset_code_names(in_list: list):
    df = pd.DataFrame()

    sql_in_clause = ", ".join(["'" + str(item) + "'" for item in in_list])
    tmp_str = "SELECT asset_type, asset_code, asset_name FROM a2assets_code_master_vw WHERE asset_code IN ({0}) order by asset_type asc".format(sql_in_clause)

    df = pd.DataFrame(a2db.execute(tmp_str).df())
    return df

@st.cache_data(ttl=3*60*60)
def a2selected_assets_df_ts(in_list: list, primary_asset: str, asset_type: str):
    # This function provides the transposed assets data in the dataframe for the selected assets
    # example: [date,asset_code_eq,asset_close_eq,volume_eq], [date, asset_code_mfi,asset_close_mfi,volume_mfi] etc.,
    # creating the selected list of assets for IN Clause
    tmp_lst: list
    tmp_lst = in_list
    if primary_asset in tmp_lst:
        tmp_lst.remove(primary_asset)

    sql_in_clause = ", ".join(["'" + str(item) + "'" for item in tmp_lst])

    if sql_in_clause is None or sql_in_clause == '':
        sql_in_clause = "'NoData'"

    a2tmp_df: pd.DataFrame()

    tmp_str = "Select date, asset_code, close, volume "
    tmp_str1 = tmp_str
    tmp_str += "FROM a2master_data_vw WHERE asset_code in ({0}) and asset_type = '{1}' order by date, asset_code asc".format(sql_in_clause, asset_type)

    # creating the all assets into the dataframe
    a2tmp_df = pd.DataFrame(a2db.execute(tmp_str).df())
    a2tmp_df = a2tmp_df.reset_index()[["date", "asset_code", "close", "volume"]]
    tmp_str = tmp_str1
    if asset_type == "crypto" or asset_type == "crypto_etf":
        a2tmp_df["close"] = a2tmp_df["close"] * 84

    tmp_str = "Select date, asset_code, close, volume "
    tmp_str1 = tmp_str
    tmp_str += "FROM a2master_data_vw WHERE asset_code = '{0}' order by date, asset_code asc".format(primary_asset)

    a2tmp_df_primary = pd.DataFrame(a2db.execute(tmp_str).df())
    a2tmp_df_primary = a2tmp_df_primary.reset_index()[["date", "asset_code", "close", "volume"]]
    tmp_str = tmp_str1

    if st.session_state.a2primary_asset == "Crypto_Assets":
        a2tmp_df_primary["close"] = a2tmp_df_primary["close"] * 84

    if primary_asset == 'primary':
        return a2tmp_df_primary
    else:
        a2tmp_df = pd.concat([a2tmp_df_primary, a2tmp_df], axis=0)
        return a2tmp_df


