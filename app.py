# app.py

import streamlit as st
import scipy.io
import numpy as np
import pandas as pd
import plotly.express as px
import os

# -------------- Load Data --------------
@st.cache_data
def load_data():
    # Load .mat data
    metrics = scipy.io.loadmat("CountyLevelMetrics.mat")
    return {
        "AWAREUSCF": metrics["AWAREUSCF"].flatten(),
        "EFkgkWh": metrics["EFkgkWh"].flatten(),
        "EWIF": metrics["EWIF"].flatten(),
        "CountyFIPS": metrics["CountyFIPS"].flatten(),
    }

data = load_data()

# -------------- Sidebar Input UI --------------
st.title("Environmental Impact Explorer")

# (1) State selection dropdown
state = st.selectbox("Select a state:", options=[
    "USA", "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", 
    "Connecticut", "Delaware", "Florida", "Georgia", "Idaho", "Illinois", "Indiana",
    "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
    "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
    "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", 
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", 
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", 
    "West Virginia", "Wisconsin", "Wyoming"
])

# (2) Metric selection
metric_option = st.selectbox("Select a metric:", options=[
    "carbon footprint", "scope 1 & 2 water footprint", "water scarcity footprint"
])

# (3) On-site power input
power_value = st.text_input("On-site power consumption:")
power_unit = st.selectbox("Power unit:", ["kWh/yr", "kWh/mo", "kW", "MW"])

# (4) Water input
water_value = st.text_input("On-site water consumption:")
water_unit = st.selectbox("Water unit:", ["L/yr", "L/mo", "L/s", "gpm", "gal/mo"])

# (5) About the Tool button
if st.button("About the Tool"):
    st.info("""
        **About This Tool**
        
        This app helps estimate environmental impacts by visualizing county-level data
        for selected U.S. states. Metrics include:
        - Carbon footprint (kg CO₂/kWh)
        - Scope 1 & 2 water footprint (L/kWh)
        - Water scarcity footprint (L-eq/kWh)
        
        Input your facility's power and water usage, choose a state and metric,
        then click "Make Plot" to visualize relative county-level impact.
    """)

# (6) Make Plot button
if st.button("Make Plot"):
    metric_map = {
        "carbon footprint": data["EFkgkWh"],
        "scope 1 & 2 water footprint": data["EWIF"],
        "water scarcity footprint": data["AWAREUSCF"]
    }
    values = metric_map[metric_option]
    fips = data["CountyFIPS"]

    # Create a DataFrame for mapping
    df = pd.DataFrame({
        "fips": fips.astype(str).str.zfill(5),
        "value": values
    })

    # Compute percentiles
    low = np.percentile(df['value'], 33)
    mid = np.percentile(df['value'], 66)

    def color_category(val):
        if val <= low:
            return "Low (Green)"
        elif val <= mid:
            return "Medium (Yellow)"
        else:
            return "High (Red)"

    df["category"] = df["value"].apply(color_category)

    fig = px.choropleth(df, geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
                        locations="fips", color="category",
                        color_discrete_map={
                            "Low (Green)": "green",
                            "Medium (Yellow)": "yellow",
                            "High (Red)": "red"
                        },
                        scope="usa",
                        labels={"category": "Impact Level"},
                        title=f"{metric_option.title()} Map")
    st.plotly_chart(fig)

# (7) Exit button
if st.button("Exit"):
    st.warning("App closed. Please refresh to restart.")
    st.stop()
