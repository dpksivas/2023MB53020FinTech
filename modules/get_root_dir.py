import os as os
import streamlit as st

@st.cache_data
def get_root_dir():
    root_dir = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
    path_ascend_count = 0
    for i in range(path_ascend_count):
        root_dir = os.path.dirname(root_dir)
    return root_dir