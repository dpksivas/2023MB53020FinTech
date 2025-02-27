# The DuckDB will be created based on the csv files / parquet files, tables are created in the Parquet files
import os
import duckdb as dd
import sys

def create_tables_from_parquet(a2db: dd):
    a2root = sys.path[1]
    a2data_path = "{0}\\data\\".format(a2root)
    #a2db = dd.connect()
    files = os.listdir(a2data_path)
    for file in files:
        if file.endswith(".parquet"):
            table_name = file.split(".")[0]
            tmp_str = "create or replace table " + table_name
            tmp_str += " as select * from read_parquet('"
            tmp_str += "{1}{0}.parquet".format(table_name,a2data_path) + "')"
            #print (tmp_str)
            a2db.execute(tmp_str)
            print(a2db.execute("select '{0}', count(1) from {0}".format(table_name)).df())

    #create view for Asset code, asset name, asset type
    tmp_str = "CREATE OR REPLACE VIEW a2assets_code_master_vw AS "
    tmp_str += "SELECT asset_code, asset_name, 'EQ' asset_type FROM a2equity_meta UNION "
    tmp_str += "SELECT asset_code, asset_name, 'MF' asset_type FROM a2mfi_meta UNION "
    tmp_str += "SELECT asset_code, asset_name, 'Index' asset_type FROM a2index_meta UNION "
    tmp_str += "SELECT asset_code, asset_name, 'Crypto' asset_type FROM a2crypto_meta UNION "
    tmp_str += "SELECT asset_code, asset_name, 'CryptoETF' asset_type FROM a2crypto_etf_meta"

    # a2db.execute(tmp_str)

    return 1

