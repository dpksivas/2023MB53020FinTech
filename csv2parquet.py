# Delete the parquet files existing data folder and converting the csv files into parquet files
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import sys, os

def a2process_files(csv_file_name: str, csv_file_path: str):
    new_filename = Path(csv_file_name).stem + '.parquet'
    df = pd.read_csv(csv_file_name)
    table = pa.Table.from_pandas(df)
    parquet_file_path = csv_file_path + '/' + new_filename
    pq.write_table(table, parquet_file_path)
    return 1

def delete_parquet():
    a2root = sys.path[1]
    a2parquet_path = a2root + '\\data\\'
    #print(a2parquet_path)
    for f in os.listdir(a2parquet_path):
        if f.endswith(".parquet"):
            os.remove(os.path.join(a2parquet_path, f))
    return 1

def get_file_names(folder_path: str):
    """Gets the names of all files in a given folder."""
    file_names = []
    for entry in os.listdir(folder_path):
        if entry.endswith('.csv'):
            if os.path.isfile(os.path.join(folder_path, entry)):
                file_names.append(entry)
    return file_names

def csv2parquet():
    a2root = sys.path[1]
    a2parquet_path = a2root + '\\data\\'
    folder_path = a2parquet_path
    file_names = []
    file_names = get_file_names(folder_path)

    for file in file_names:
        csv_file_path = a2parquet_path + file
        a2process_files(csv_file_path, a2parquet_path)
    return 1
