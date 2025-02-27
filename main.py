import time
from datetime import date
import pandas as pd
from duckdb.experimental.spark.sql.functions import to_date
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import json, sys, os
import duckdb as dd
import streamlit as st
#custom package download
import a2load_data as a2load_data
app = FastAPI()

templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("microservicesHome.html", {"request": request, "stuff": 123})

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

#Check the process is run first time:
jsonFile = open("a2config.json", "r") # Open the JSON file for reading
data = json.load(jsonFile) # Read the JSON into the buffer
jsonFile.close()

a2db = dd.connect("a2db.db")
print('Before Processing Counts')
a2sql_string = "select asset_code, count(1) from a2master_data_vw where asset_type = 'equity' group by asset_code"
a2tmp_df = a2db.execute(a2sql_string).df()
print(a2tmp_df.head(10))

a2load_data.a2load_ts_data_dr()

print('After Processing Counts')
a2sql_string = "select asset_code, count(1) from a2master_data_vw where asset_type = 'equity' group by asset_code"
a2tmp_df = a2db.execute(a2sql_string).df()
print(a2tmp_df.head(10))

# update a2config.json
data["a2first_time_run"] = 'false'
data["a2last_run_date"] = str(date.today())
data["a2last_run_time"] = str(time.time())

## Save the run update to JSON file
jsonFile = open("a2config.json", "w+")
jsonFile.write(json.dumps(data))
jsonFile.close()
