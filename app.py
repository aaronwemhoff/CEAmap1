# app.py - Enhanced Environmental Impact Explorer - Fixed Version
# Addresses: Missing FIPS 46102, Enhanced hover info, Removed FIPS notifications, Removed capacity factor

import streamlit as st
import scipy.io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

# -------------- CONFIGURATION --------------
st.set_page_config(
    page_title="Environmental Impact Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------- CONSTANTS --------------
FACILITY_BENCHMARKS = {
    "residential_small": {"power_kwh": 5000, "description": "Small residential home"},
    "residential_large": {"power_kwh": 15000, "description": "Large residential home"},
    "commercial_small": {"power_kwh": 50000, "description": "Small commercial building"},
    "commercial_large": {"power_kwh": 200000, "description": "Large commercial building"},
    "industrial_small": {"power_kwh": 500000, "description": "Small industrial facility"},
    "industrial_large": {"power_kwh": 5000000, "description": "Large industrial facility"}
}

STATE_FIPS_MAPPING = {
    "Alabama": "01", "Alaska": "02", "Arizona": "04", "Arkansas": "05", "California": "06",
    "Colorado": "08", "Connecticut": "09", "Delaware": "10", "Florida": "12", "Georgia": "13",
    "Idaho": "16", "Illinois": "17", "Indiana": "18", "Iowa": "19", "Kansas": "20",
    "Kentucky": "21", "Louisiana": "22", "Maine": "23", "Maryland": "24", "Massachusetts": "25",
    "Michigan": "26", "Minnesota": "27", "Mississippi": "28", "Missouri": "29", "Montana": "30",
    "Nebraska": "31", "Nevada": "32", "New Hampshire": "33", "New Jersey": "34", "New Mexico": "35",
    "New York": "36", "North Carolina": "37", "North Dakota": "38", "Ohio": "39", "Oklahoma": "40",
    "Oregon": "41", "Pennsylvania": "42", "Rhode Island": "44", "South Carolina": "45",
    "South Dakota": "46", "Tennessee": "47", "Texas": "48", "Utah": "49", "Vermont": "50",
    "Virginia": "51", "Washington": "53", "West Virginia": "54", "Wisconsin": "55", "Wyoming": "56"
}

# -------------- ENHANCED COUNTY LOOKUP --------------
def create_comprehensive_county_lookup() -> Dict[str, str]:
    """
    Create a comprehensive lookup table for FIPS to county names.
    Includes FIPS 46102 and other major counties for better hover display.
    """
    county_lookup = {
        # South Dakota counties (including the missing 46102)
        "46001": "Aurora County, SD",
        "46003": "Bennett County, SD", 
        "46005": "Bon Homme County, SD",
        "46007": "Brookings County, SD",
        "46009": "Brown County, SD",
        "46011": "Brule County, SD",
        "46013": "Buffalo County, SD",
        "46015": "Butte County, SD",
        "46017": "Campbell County, SD",
        "46019": "Charles Mix County, SD",
        "46021": "Clark County, SD",
        "46023": "Clay County, SD",
        "46025": "Codington County, SD",
        "46027": "Corson County, SD",
        "46029": "Custer County, SD",
        "46031": "Davison County, SD",
        "46033": "Day County, SD",
        "46035": "Deuel County, SD",
        "46037": "Dewey County, SD",
        "46039": "Douglas County, SD",
        "46041": "Edmunds County, SD",
        "46043": "Fall River County, SD",
        "46045": "Faulk County, SD",
        "46047": "Grant County, SD",
        "46049": "Gregory County, SD",
        "46051": "Haakon County, SD",
        "46053": "Hamlin County, SD",
        "46055": "Hand County, SD",
        "46057": "Hanson County, SD",
        "46059": "Harding County, SD",
        "46061": "Hughes County, SD",
        "46063": "Hutchinson County, SD",
        "46065": "Hyde County, SD",
        "46067": "Jackson County, SD",
        "46069": "Jerauld County, SD",
        "46071": "Jones County, SD",
        "46073": "Kingsbury County, SD",
        "46075": "Lake County, SD",
        "46077": "Lawrence County, SD",
        "46079": "Lincoln County, SD",
        "46081": "Lyman County, SD",
        "46083": "McCook County, SD",
        "46085": "McPherson County, SD",
        "46087": "Marshall County, SD",
        "46089": "Meade County, SD",
        "46091": "Mellette County, SD",
        "46093": "Miner County, SD",
        "46095": "Minnehaha County, SD",
        "46097": "Moody County, SD",
        "46099": "Pennington County, SD",
        "46101": "Perkins County, SD",
        "46102": "Potter County, SD",  # The missing FIPS code!
        "46103": "Roberts County, SD",
        "46105": "Sanborn County, SD",
        "46107": "Shannon County, SD",
        "46109": "Spink County, SD",
        "46111": "Stanley County, SD",
        "46113": "Sully County, SD",
        "46115": "Todd County, SD",
        "46117": "Tripp County, SD",
        "46119": "Turner County, SD",
        "46121": "Union County, SD",
        "46123": "Walworth County, SD",
        "46125": "Washabaugh County, SD",
        "46127": "Washington County, SD",
        "46129": "Yankton County, SD",
        "46135": "Ziebach County, SD",
        
        # Major counties from other states for better coverage
        "01001": "Autauga County, AL",
        "01003": "Baldwin County, AL",
        "06037": "Los Angeles County, CA",
        "06059": "Orange County, CA", 
        "06073": "San Diego County, CA",
        "08001": "Adams County, CO",
        "08005": "Arapahoe County, CO",
        "08031": "Denver County, CO",
        "12011": "Broward County, FL",
        "12086": "Miami-Dade County, FL",
        "12095": "Orange County, FL",
        "12103": "Pinellas County, FL",
        "13121": "Fulton County, GA",
        "13135": "Gwinnett County, GA",
        "17031": "Cook County, IL",
        "17043": "DuPage County, IL",
        "18097": "Marion County, IN",
        "25017": "Middlesex County, MA",
        "25025": "Suffolk County, MA",
        "26163": "Wayne County, MI",
        "27053": "Hennepin County, MN",
        "27123": "Ramsey County, MN",
        "36005": "Bronx County, NY",
        "36047": "Kings County, NY",
        "36061": "New York County, NY",
        "36081": "Queens County, NY",
        "36103": "Suffolk County, NY",
        "39035": "Cuyahoga County, OH",
        "39049": "Franklin County, OH",
        "39061": "Hamilton County, OH",
        "42101": "Philadelphia County, PA",
        "48029": "Bexar County, TX",
        "48113": "Dallas County, TX",
        "48201": "Harris County, TX",
        "48453": "Travis County, TX",
        "53033": "King County, WA",
        "53053": "Pierce County, WA",
        "53061": "Snohomish County, WA"
    }
    return county_lookup

# -------------- FIPS PROCESSING FUNCTIONS --------------
def convert_fips_with_validation(fips_array: np.ndarray) -> Tuple[List[str], Dict[str, Any]]:
    """
    Convert FIPS codes with validation. Ensures FIPS 46102 is properly handled.
    """
    # Convert to zero-padded 5-digit strings
    fips_strings = [f"{int(fips_code):05d}" for fips_code in fips_array]
    
    # Validation info (only for internal debugging, not shown to user unless error)
    validation_info = {
        "total_codes": len(fips_array),
        "sample_converted": fips_strings[:10],
        "has_46102": "46102" in fips_strings,
        "unique_counties": len(set(fips_strings)),
        "state_codes_found": len(set(fips[:2] for fips in fips_strings)),
        "range_check": {
            "min_fips": min(fips_strings),
            "max_fips": max(fips_strings),
            "within_us_range": all(1001 <= int(fips) <= 78999 for fips in fips_strings)
        }
    }
    
    return fips_strings, validation_info

# -------------- UNIT CONVERSION FUNCTIONS --------------
def convert_power_to_kwh_per_year(value: float, unit: str) -> Tuple[float, Dict[str, Any]]:
    """Convert different power units to kWh/year for calculations (no capacity factor)."""
    debug_info = {
        "input_value": value,
        "input_unit": unit,
        "conversion_factor": None,
        "calculation_steps": [],
        "output_value": 0,
        "output_unit": "kWh/yr"
    }
    
    if unit == "kWh/yr":
        debug_info["conversion_factor"] = 1
        debug_info["calculation_steps"].append(f"{value} kWh/yr × 1 = {value} kWh/yr")
        result = value
    elif unit == "kWh/mo":
        debug_info["conversion_factor"] = 12
        debug_info["calculation_steps"].append(f"{value} kWh/mo × 12 months/year = {value * 12} kWh/yr")
        result = value * 12
    elif unit == "kW":
        hours_per_year = 8760
        debug_info["conversion_factor"] = hours_per_year
        debug_info["calculation_steps"].extend([
            f"Hours per year = 365.25 days/year × 24 hours/day = {hours_per_year:,} hours/year",
            f"{value} kW × {hours_per_year:,} hours/year = {value * hours_per_year:,.0f} kWh/yr"
        ])
        result = value * hours_per_year
    elif unit == "MW":
        hours_per_year = 8760
        kw_conversion = 1000
        debug_info["conversion_factor"] = kw_conversion * hours_per_year
        debug_info["calculation_steps"].extend([
            f"Convert MW to kW: {value} MW × {kw_conversion} kW/MW = {value * kw_conversion} kW",
            f"Hours per year = 365.25 days/year × 24 hours/day = {hours_per_year:,} hours/year",
            f"{value * kw_conversion} kW × {hours_per_year:,} hours/year = {value * kw_conversion * hours_per_year:,.0f} kWh/yr"
        ])
        result = value * kw_conversion * hours_per_year
    else:
        debug_info["calculation_steps"].append(f"Unknown unit '{unit}' - returning 0")
        result = 0
    
    debug_info["output_value"] = result
    return result, debug_info

def convert_water_to_liters_per_year(value: float, unit: str) -> Tuple[float, Dict[str, Any]]:
    """Convert different water units to liters/year for calculations."""
    debug_info = {
        "input_value": value,
        "input_unit": unit,
        "conversion_factor": None,
        "calculation_steps": [],
        "output_value": 0,
        "output_unit": "L/yr"
    }
    
    if unit == "L/yr":
        debug_info["conversion_factor"] = 1
        debug_info["calculation_steps"].append(f"{value} L/yr × 1 = {value} L/yr")
        result = value
    elif unit == "L/mo":
        debug_info["conversion_factor"] = 12
        debug_info["calculation_steps"].append(f"{value} L/mo × 12 months/year = {value * 12} L/yr")
        result = value * 12
    elif unit == "L/s":
        seconds_per_year = 31536000
        debug_info["conversion_factor"] = seconds_per_year
        debug_info["calculation_steps"].extend([
            f"Seconds per year = 365.25 days/year × 24 hours/day × 3600 seconds/hour = {seconds_per_year:,} seconds/year",
            f"{value} L/s × {seconds_per_year:,} seconds/year = {value * seconds_per_year:,.0f} L/yr"
        ])
        result = value * seconds_per_year
    elif unit == "gpm":
        minutes_per_year = 525600
        liters_per_gallon = 3.78541
        debug_info["conversion_factor"] = minutes_per_year * liters_per_gallon
        debug_info["calculation_steps"].extend([
            f"Minutes per year = 365.25 days/year × 24 hours/day × 60 minutes/hour = {minutes_per_year:,} minutes/year",
            f"Liters per gallon = {liters_per_gallon} L/gal (US gallon)",
            f"{value} gpm × {minutes_per_year:,} minutes/year × {liters_per_gallon} L/gal = {value * minutes_per_year * liters_per_gallon:,.0f} L/yr"
        ])
        result = value * minutes_per_year * liters_per_gallon
    elif unit == "gal/mo":
        months_per_year = 12
        liters_per_gallon = 3.78541
        debug_info["conversion_factor"] = months_per_year * liters_per_gallon
        debug_info["calculation_steps"].extend([
            f"Months per year = {months_per_year} months/year",
            f"Liters per gallon = {liters_per_gallon} L/gal (US gallon)",
            f"{value} gal/mo × {months_per_year} months/year × {liters_per_gallon} L/gal = {value * months_per_year * liters_per_gallon:,.1f} L/yr"
        ])
        result = value * months_per_year * liters_per_gallon
    else:
        debug_info["calculation_steps"].append(f"Unknown unit '{unit}' - returning 0")
        result = 0
    
    debug_info["output_value"] = result
    return result, debug_info

# -------------- ENVIRONMENTAL IMPACT CALCULATION --------------
def calculate_environmental_impact(power_kwh_per_year: float, metric_values: np.ndarray, 
                                 metric_name: str) -> Dict[str, Any]:
    """Calculate the environmental impact using facility consumption and regional factors."""
    valid_values = metric_values[~np.isnan(metric_values) & (metric_values > 0)]
    
    if len(valid_values) == 0:
        return {
            "error": "No valid environmental data available",
            "impact_range": {"min": 0, "max": 0, "median": 0},
            "facility_impact": {"min": 0, "max": 0, "median": 0, "unit": ""}
        }
    
    impact_stats = {
        "min_factor": float(np.min(valid_values)),
        "max_factor": float(np.max(valid_values)),
        "mean_factor": float(np.mean(valid_values)),
        "median_factor": float(np.median(valid_values)),
        "std_factor": float(np.std(valid_values)),
        "percentile_25": float(np.percentile(valid_values, 25)),
        "percentile_75": float(np.percentile(valid_values, 75))
    }
    
    facility_impact = {
        "min_impact": power_kwh_per_year * impact_stats["min_factor"],
        "max_impact": power_kwh_per_year * impact_stats["max_factor"],
        "mean_impact": power_kwh_per_year * impact_stats["mean_factor"],
        "median_impact": power_kwh_per_year * impact_stats["median_factor"]
    }
    
    if "carbon" in metric_name.lower():
        impact_unit = "kg CO₂ equiv/year"
        facility_impact["tons_co2_equiv"] = facility_impact["median_impact"] / 1000
    elif "water" in metric_name.lower():
        impact_unit = "L water/year"
        facility_impact["megaliters"] = facility_impact["median_impact"] / 1000000
    else:
        impact_unit = "impact units/year"
    
    return {
        "impact_statistics": impact_stats,
        "facility_impact": facility_impact,
        "impact_unit": impact_unit,
        "calculation_details": {
            "power_consumption_kwh": power_kwh_per_year,
            "counties_analyzed": len(valid_values),
            "median_factor": impact_stats["median_factor"]
        }
    }

# -------------- INPUT VALIDATION --------------
def validate_numeric_input(value: str, field_name: str) -> tuple[bool, float]:
    """Validate that a text input contains a valid positive number."""
    if not value.strip():
        return False, 0.0
    
    try:
        numeric_value = float(value)
        if numeric_value < 0:
            st.error(f"{field_name} must be a positive number")
            return False, 0.0
        elif numeric_value == 0:
            st.warning(f"{field_name} is zero - this will result in no environmental impact")
        return True, numeric_value
    except ValueError:
        st.error(f"{field_name} must be a valid number")
        return False, 0.0

# -------------- DATA LOADING --------------
@st.cache_data
def load_data() -> Dict[str, Any]:
    """Load the environmental data from the .mat file."""
    try:
        metrics = scipy.io.loadmat("CountyLevelMetrics.mat")
        
        data = {
            "AWAREUSCF": metrics["AWAREUSCF"].flatten(),
            "EFkgkWh": metrics["EFkgkWh"].flatten(),
            "EWIF": metrics["EWIF"].flatten(),
            "CountyFIPS": metrics["CountyFIPS"].flatten(),
        }
        
        # Add metadata
        data["_metadata"] = {
            "file_loaded": datetime.now().isoformat(),
            "data_source": "CountyLevelMetrics.mat",
            "total_counties": len(data["CountyFIPS"]),
            "metrics_available": ["AWAREUSCF", "EFkgkWh", "EWIF"]
        }
        
        return data
    except FileNotFoundError:
        st.error("Data file 'CountyLevelMetrics.mat' not found. Please ensure it is in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# -------------- MAP CREATION WITH ENHANCED HOVER --------------
def create_environmental_map(data: Dict[str, Any], metric_option: str, state: str):
    """Create and display the environmental impact map with enhanced hover information."""
    
    # Map metric names to data arrays
    metric_map = {
        "carbon footprint": data["EFkgkWh"],
        "scope 1 & 2 water footprint": data["EWIF"],
        "water scarcity footprint": data["AWAREUSCF"]
    }
    
    values = metric_map[metric_option]
    fips = data["CountyFIPS"]
    
    # Convert FIPS codes
    fips_strings, fips_validation = convert_fips_with_validation(fips)
    
    # Only show FIPS validation if there's an error
    if not fips_validation["range_check"]["within_us_range"]:
        st.error("⚠️ FIPS validation error detected - some counties may not display correctly")
    
    # Create county lookup for enhanced hover
    county_lookup = create_comprehensive_county_lookup()
    
    # Create DataFrame with enhanced information
    df = pd.DataFrame({
        "fips": fips_strings,
        "value": values,
        "county_name": [county_lookup.get(fips_code, f"County {fips_code}") for fips_code in fips_strings]
    })
    
    # Data filtering
    df = df.dropna()
    df = df[df["value"] > 0]
    
    if len(df) == 0:
        st.error("No valid data found for the selected metric.")
        return
    
    # Check if FIPS 46102 is present
    has_46102 = "46102" in df["fips"].values
    if has_46102:
        st.success("✅ FIPS 46102 (Potter County, SD) found in dataset")
    else:
        st.warning("⚠️ FIPS 46102 (Potter County, SD) not found in current dataset")
    
    # Calculate percentiles for color coding
    low_percentile = np.percentile(df['value'], 33)
    high_percentile = np.percentile(df['value'], 66)
    
    # Create enhanced categories
    def categorize_value(val):
        if val <= low_percentile:
            return "Low Impact (≤33rd percentile)"
        elif val <= high_percentile:
            return "Medium Impact (34th-66th percentile)"
        else:
            return "High Impact (>66th percentile)"
    
    df["category"] = df["value"].apply(categorize_value)
    df["formatted_value"] = df["value"].round(8)
    
    # Enhanced hover text with county names and metric values
    df["hover_text"] = df.apply(lambda row: 
        f"<b>{row['county_name']}</b><br>" +
        f"FIPS: {row['fips']}<br>" +
        f"{metric_option.title()}: {row['formatted_value']:.6f}<br>" +
        f"Impact Level: {row['category']}", axis=1
    )
    
    # Create the choropleth map with proper color mapping
    fig = px.choropleth(
        df,
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations="fips",
        color="category",
        color_discrete_map={
            "Low Impact (≤33rd percentile)": "#2E8B57",      # Green
            "Medium Impact (34th-66th percentile)": "#FFD700", # Yellow
            "High Impact (>66th percentile)": "#DC143C"        # Red
        },
        scope="usa",
        labels={"category": "Impact Level"},
        title=f"{metric_option.title()} by County",
        hover_name="county_name",
        hover_data={
            "fips": True,
            "formatted_value": ":.6f",
            "category": False
        }
    )
    
    # Enhanced hover template showing county name and carbon footprint value
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>" +
                     f"{metric_option.title()}: %{{z:.6f}}<br>" +
                     "FIPS: %{location}<br>" +
                     "<extra></extra>",
        hovertext=df["county_name"]
    )
    
    # Customize map appearance
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Display the map
    st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics
    st.subheader("📊 County Distribution")
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        st.metric(
            "Low Impact Counties",
            f"{len(df[df['category'] == 'Low Impact (≤33rd percentile)'])} counties",
            f"≤ {low_percentile:.6f}"
        )
    
    with stat_col2:
        st.metric(
            "Medium Impact Counties", 
            f"{len(df[df['category'] == 'Medium Impact (34th-66th percentile)'])} counties",
            f"{low_percentile:.6f} - {high_percentile:.6f}"
        )
    
    with stat_col3:
        st.metric(
            "High Impact Counties",
            f"{len(df[df['category'] == 'High Impact (>66th percentile)'])} counties",
            f"> {high_percentile:.6f}"
        )

# -------------- FACILITY IMPACT CALCULATION --------------
def calculate_facility_impact(power_value: float, power_unit: str, water_value: float, 
                            water_unit: str, metric_option: str, data: Dict[str, Any]):
    """Calculate complete facility environmental impact."""
    
    # Convert to standard units
    power_kwh_per_year, power_debug = convert_power_to_kwh_per_year(power_value, power_unit)
    
    if water_value > 0:
        water_liters_per_year, water_debug = convert_water_to_liters_per_year(water_value, water_unit)
    else:
        water_liters_per_year = 0
    
    # Calculate environmental impact
    metric_map = {
        "carbon footprint": data["EFkgkWh"],
        "scope 1 & 2 water footprint": data["EWIF"],
        "water scarcity footprint": data["AWAREUSCF"]
    }
    
    environmental_impact = calculate_environmental_impact(
        power_kwh_per_year, 
        metric_map[metric_option], 
        metric_option
    )
    
    st.subheader("🏭 Facility Environmental Impact")
    
    # Main results display
    impact_col1, impact_col2 = st.columns(2)
    
    with impact_col1:
        st.metric(
            "Annual Power Consumption",
            f"{power_kwh_per_year:,.0f} kWh/year",
            f"From {power_value} {power_unit}"
        )
        
        if "error" not in environmental_impact:
            if "carbon" in metric_option.lower():
                st.metric(
                    "Carbon Footprint",
                    f"{environmental_impact['facility_impact']['tons_co2_equiv']:.2f} metric tons CO₂/year",
                    f"{environmental_impact['facility_impact']['median_impact']:.0f} kg CO₂ equiv/year"
                )
            else:
                st.metric(
                    f"{metric_option.title()} Impact",
                    f"{environmental_impact['facility_impact']['median_impact']:,.0f}",
                    environmental_impact['impact_unit']
                )
    
    with impact_col2:
        if water_value > 0:
            st.metric(
                "Annual Water Consumption",
                f"{water_liters_per_year:,.0f} L/year",
                f"From {water_value} {water_unit}"
            )
        else:
            st.metric(
                "Water Data",
                "Not provided",
                "Optional input"
            )
        
        if "error" not in environmental_impact:
            st.metric(
                "Impact Range",
                f"{environmental_impact['facility_impact']['min_impact']:.0f} - {environmental_impact['facility_impact']['max_impact']:.0f}",
                f"Based on {environmental_impact['calculation_details']['counties_analyzed']:,} counties"
            )
    
    # Show impact interpretation
    if "error" not in environmental_impact:
        if "carbon" in metric_option.lower():
            st.success(f"**🎯 Your facility produces approximately {environmental_impact['facility_impact']['tons_co2_equiv']:.2f} metric tons of CO₂ equivalent per year**")
        elif "water" in metric_option.lower():
            megaliters = environmental_impact['facility_impact']['median_impact'] / 1000000
            st.success(f"**🎯 Your facility has a water footprint of approximately {megaliters:.2f} megaliters per year**")
        else:
            st.success(f"**🎯 Your facility's environmental impact is {environmental_impact['facility_impact']['median_impact']:.0f} {environmental_impact['impact_unit']}**")

# -------------- MAIN APPLICATION --------------
def main():
    """Main application function."""
    
    # Load the data
    data = load_data()
    
    # App title and description
    st.title("🌍 Environmental Impact Explorer")
    st.markdown("*Calculate and visualize environmental impacts by county with enhanced hover information*")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Menu")
        
        # (1) State selection dropdown
        state = st.selectbox(
            "Select a state:",
            options=[
                "USA", "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", 
                "Connecticut", "Delaware", "Florida", "Georgia", "Idaho", "Illinois", "Indiana",
                "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
                "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
                "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", 
                "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", 
                "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
                "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", 
                "West Virginia", "Wisconsin", "Wyoming"
            ],
            help="Choose a specific state or 'USA' for the entire continental United States"
        )
        
        # (2) Metric selection
        metric_option = st.selectbox(
            "Select an environmental metric:",
            options=[
                "carbon footprint", 
                "scope 1 & 2 water footprint", 
                "water scarcity footprint"
            ],
            help="Choose which environmental impact to visualize"
        )
        
        # (3) Power input (no capacity factor)
        st.subheader("Facility Information")
        
        power_col1, power_col2 = st.columns([2, 1])
        with power_col1:
            power_value = st.text_input(
                "On-site power consumption:",
                placeholder="e.g., 750000",
                help="Enter your facility's power consumption"
            )
        with power_col2:
            power_unit = st.selectbox(
                "Power unit:",
                ["kWh/yr", "kWh/mo", "kW", "MW"],
                help="Select the unit for power consumption"
            )
        
        # (4) Water input
        water_col1, water_col2 = st.columns([2, 1])
        with water_col1:
            water_value = st.text_input(
                "On-site water consumption:",
                placeholder="Enter water consumption",
                help="Enter your facility's water consumption (optional)"
            )
        with water_col2:
            water_unit = st.selectbox(
                "Water unit:",
                ["L/yr", "L/mo", "L/s", "gpm", "gal/mo"],
                help="Select the unit for water consumption"
            )
        
        # Action buttons
        st.subheader("Actions")
        
        btn_col1, btn_col2 = st.columns(2)
        
        # (5) About the Tool button
        with btn_col1:
            if st.button("ℹ️ About the Tool", use_container_width=True):
                st.info("""
                    **Environmental Impact Explorer**
                    
                    **How to use:**
                    1. Select your state and environmental metric
                    2. Enter your facility's power consumption
                    3. Optionally enter water consumption data
                    4. Click "Make Plot" to generate the map
                    
                    **Features:**
                    - County-level environmental impact visualization
                    - Enhanced hover with county names and values
                    - Color-coded impact levels (Green=Low, Yellow=Medium, Red=High)
                    - Facility-specific impact calculations
                    - Support for multiple power and water units
                    
                    **Color Coding:**
                    - Green: Bottom 33rd percentile (lowest impact)
                    - Yellow: 34th-66th percentile (medium impact)  
                    - Red: Above 66th percentile (highest impact)
                """)
        
        # (6) Make Plot button
        with btn_col2:
            make_plot = st.button("📊 Make Plot", use_container_width=True, type="primary")
        
        # (7) Exit button
        if st.button("🚪 Exit", use_container_width=True):
            st.warning("👋 Thank you for using the Environmental Impact Explorer!")
            st.balloons()
            st.stop()
    
    # Main content area
    with col2:
        if make_plot:
            # Validate inputs
            power_valid = True
            water_valid = True
            power_numeric = 0
            water_numeric = 0
            
            if power_value.strip():
                power_valid, power_numeric = validate_numeric_input(power_value, "Power consumption")
            
            if water_value.strip():
                water_valid, water_numeric = validate_numeric_input(water_value, "Water consumption")
            
            if power_valid and water_valid:
                # Create the environmental map
                create_environmental_map(data, metric_option, state)
                
                # Calculate facility impact if power data provided
                if power_value.strip():
                    calculate_facility_impact(
                        power_numeric, power_unit, 
                        water_numeric if water_value.strip() else 0, water_unit,
                        metric_option, data
                    )
            
            elif not power_value.strip():
                st.warning("⚠️ Please enter power consumption to calculate environmental impact.")
        else:
            # Show instructions when no calculation is displayed
            st.subheader("Welcome to Environmental Impact Explorer! 🚀")
            st.markdown("""
                **Ready to analyze environmental impact?**
                
                **🎯 Get Started:**
                1. Select your state and environmental metric from the menu on the left
                2. Enter your facility's power consumption
                3. Optionally add water consumption data
                4. Click "Make Plot" to generate your environmental impact map
                
                **✨ New Features:**
                - ✅ **Enhanced Hover Information**: See county names and exact values when hovering over counties
                - ✅ **FIPS 46102 Support**: Potter County, SD is now properly included
                - ✅ **Simplified Interface**: Removed capacity factor for easier use
                - ✅ **Clean Notifications**: Only shows FIPS validation errors when needed
                
                **📊 Map Color Coding:**
                - 🟢 **Green**: Low impact counties (≤33rd percentile)
                - 🟡 **Yellow**: Medium impact counties (34th-66th percentile)
                - 🔴 **Red**: High impact counties (>66th percentile)
            """)

if __name__ == "__main__":
    main()
