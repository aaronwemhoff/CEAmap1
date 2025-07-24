# app.py - Facility Environmental Impact Explorer
# A beginner-friendly Streamlit app for visualizing environmental impacts

import streamlit as st
import scipy.io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from typing import Dict, Any

# -------------- CONFIGURATION --------------
# Set up the page configuration (this should be the first Streamlit command)
st.set_page_config(
    page_title="Facility Environmental Impact Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------- HELPER FUNCTIONS --------------
def convert_power_to_kwh_per_year(value: float, unit: str) -> float:
    """
    Convert different power units to kWh/year for calculations.
    
    Args:
        value: The numerical value
        unit: The unit of measurement
    
    Returns:
        float: Value converted to kWh/year
    """
    if unit == "kWh/yr":
        return value
    elif unit == "kWh/mo":
        return value * 12  # 12 months per year
    elif unit == "kW":
        return value * 8760  # 8760 hours per year
    elif unit == "MW":
        return value * 1000 * 8760  # Convert MW to kW, then to kWh/year
    else:
        return 0

def convert_water_to_liters_per_year(value: float, unit: str) -> float:
    """
    Convert different water units to liters/year for calculations.
    
    Args:
        value: The numerical value
        unit: The unit of measurement
    
    Returns:
        float: Value converted to liters/year
    """
    if unit == "L/yr":
        return value
    elif unit == "L/mo":
        return value * 12  # 12 months per year
    elif unit == "L/s":
        return value * 31536000  # 365.25 * 24 * 3600 seconds per year
    elif unit == "gpm":  # gallons per minute
        return value * 525600 * 3.78541  # minutes per year * liters per gallon
    elif unit == "gal/mo":  # gallons per month
        return value * 12 * 3.78541  # months per year * liters per gallon
    else:
        return 0

def validate_numeric_input(value: str, field_name: str) -> tuple[bool, float]:
    """
    Validate that a text input contains a valid positive number.
    
    Args:
        value: The input string to validate
        field_name: Name of the field for error messages
    
    Returns:
        tuple: (is_valid, numeric_value)
    """
    if not value.strip():
        return False, 0.0
    
    try:
        numeric_value = float(value)
        if numeric_value < 0:
            st.error(f"{field_name} must be a positive number")
            return False, 0.0
        return True, numeric_value
    except ValueError:
        st.error(f"{field_name} must be a valid number")
        return False, 0.0

def get_state_fips_codes(state: str) -> list:
    """
    Get FIPS codes for counties in a specific state.
    
    Args:
        state: State name or "USA" for all continental US
    
    Returns:
        list: FIPS codes for the state (first 2 digits)
    """
    state_fips = {
        "USA": None,  # Special case for all continental US
        "Alabama": ["01"],
        "Alaska": ["02"],
        "Arizona": ["04"],
        "Arkansas": ["05"],
        "California": ["06"],
        "Colorado": ["08"],
        "Connecticut": ["09"],
        "Delaware": ["10"],
        "Florida": ["12"],
        "Georgia": ["13"],
        "Idaho": ["16"],
        "Illinois": ["17"],
        "Indiana": ["18"],
        "Iowa": ["19"],
        "Kansas": ["20"],
        "Kentucky": ["21"],
        "Louisiana": ["22"],
        "Maine": ["23"],
        "Maryland": ["24"],
        "Massachusetts": ["25"],
        "Michigan": ["26"],
        "Minnesota": ["27"],
        "Mississippi": ["28"],
        "Missouri": ["29"],
        "Montana": ["30"],
        "Nebraska": ["31"],
        "Nevada": ["32"],
        "New Hampshire": ["33"],
        "New Jersey": ["34"],
        "New Mexico": ["35"],
        "New York": ["36"],
        "North Carolina": ["37"],
        "North Dakota": ["38"],
        "Ohio": ["39"],
        "Oklahoma": ["40"],
        "Oregon": ["41"],
        "Pennsylvania": ["42"],
        "Rhode Island": ["44"],
        "South Carolina": ["45"],
        "South Dakota": ["46"],
        "Tennessee": ["47"],
        "Texas": ["48"],
        "Utah": ["49"],
        "Vermont": ["50"],
        "Virginia": ["51"],
        "Washington": ["53"],
        "West Virginia": ["54"],
        "Wisconsin": ["55"],
        "Wyoming": ["56"]
    }
    
    return state_fips.get(state, None)

def get_metric_units(metric_option: str) -> str:
    """
    Get the appropriate units for display based on the selected metric.
    
    Args:
        metric_option: The selected environmental metric
    
    Returns:
        str: Units for the metric
    """
    if metric_option == "carbon footprint":
        return "kg CO₂-eq/year"
    elif metric_option == "scope 1 & 2 water footprint":
        return "L water/year"
    elif metric_option == "water scarcity footprint":
        return "L water-eq/year"
    else:
        return "units/year"

# -------------- DATA LOADING --------------
@st.cache_data
def load_data() -> Dict[str, Any]:
    """
    Load the environmental data from the .mat file.
    The @st.cache_data decorator ensures this only runs once and caches the result.
    
    Returns:
        Dict containing the loaded data arrays
    """
    try:
        # Load .mat data file
        metrics = scipy.io.loadmat("CountyLevelMetrics.mat")
        
        # Extract and flatten the arrays (convert from 2D to 1D)
        return {
            "AWAREUSCF": metrics["AWAREUSCF"].flatten(),    # Water scarcity footprint
            "EFkgkWh": metrics["EFkgkWh"].flatten(),        # Carbon footprint
            "EWIF": metrics["EWIF"].flatten(),              # Water footprint
            "CountyFIPS": metrics["CountyFIPS"].flatten(),  # County identification codes
        }
    except FileNotFoundError:
        st.error("Data file 'CountyLevelMetrics.mat' not found. Please ensure it is in the same directory as this app.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# -------------- MAIN APP --------------
def main():
    """Main application function that contains all the UI and logic."""
    
    # Load the data
    data = load_data()
    
    # App title and description
    st.title("🌍 Facility Environmental Impact Explorer")
    st.markdown("*Visualize county-level environmental impacts across the United States*")
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
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
        
        # (3) On-site power input
        st.subheader("Facility Information")
        
        power_col1, power_col2 = st.columns([2, 1])
        with power_col1:
            power_value = st.text_input(
                "On-site power consumption:",
                placeholder="Enter power consumption",
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
                help="Enter your facility's water consumption"
            )
        with water_col2:
            water_unit = st.selectbox(
                "Water unit:",
                ["L/yr", "L/mo", "L/s", "gpm", "gal/mo"],
                help="Select the unit for water consumption"
            )
        
        # Action buttons
        st.subheader("Actions")
        
        # Create button columns for better layout
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            # (5) About the Tool button
            if st.button("ℹ️ About", use_container_width=True):
                st.info("""
                    **About This Tool**
                    
                    The Facility Environmental Impact Explorer helps estimate environmental 
                    impacts by visualizing county-level data for selected U.S. states.
                    
                    **Available Metrics:**
                    - **Carbon footprint**: kg CO₂ equivalent per kWh
                    - **Scope 1 & 2 water footprint**: Liters of water per kWh
                    - **Water scarcity footprint**: Liters water-equivalent per kWh
                    
                    **How to Use:**
                    1. Select a state and environmental metric
                    2. Enter your facility's power and water consumption
                    3. Click "Make Plot" to visualize county-level impacts
                    
                    **Color Coding:**
                    - 🟢 Green: Bottom 33% (lowest impact)
                    - 🟡 Yellow: Middle 33% (medium impact)  
                    - 🔴 Red: Top 33% (highest impact)
                    
                    **Hover Information:**
                    When you hover over counties, you'll see the calculated environmental
                    impact for your specific facility in that location.
                """)
        
        with btn_col2:
            # (6) Make Plot button
            make_plot = st.button("📊 Make Plot", use_container_width=True, type="primary")
        
        with btn_col3:
            # (7) Exit button
            if st.button("🚪 Exit", use_container_width=True):
                st.warning("👋 Thank you for using the Facility Environmental Impact Explorer!")
                st.balloons()
                st.stop()
    
    # Main content area
    with col2:
        if make_plot:
            # Validate inputs if provided
            power_valid = True
            water_valid = True
            power_numeric = 0
            water_numeric = 0
            
            if power_value.strip():
                power_valid, power_numeric = validate_numeric_input(power_value, "Power consumption")
            
            if water_value.strip():
                water_valid, water_numeric = validate_numeric_input(water_value, "Water consumption")
            
            if power_valid and water_valid:
                # Create the plot
                create_environmental_map(data, metric_option, state, power_numeric, power_unit, water_numeric, water_unit)
                
                # Display facility impact if inputs provided
                if power_value.strip() and water_value.strip():
                    calculate_facility_impact(power_numeric, power_unit, water_numeric, water_unit, metric_option)
        else:
            # Show instructions when no plot is displayed
            st.subheader("Welcome! 👋")
            st.markdown("""
                **Get Started:**
                1. Select your state and environmental metric on the left
                2. Optionally enter your facility's consumption data
                3. Click "Make Plot" to visualize environmental impacts
                
                **Features:**
                - Interactive county-level maps
                - Multiple environmental metrics
                - Facility impact calculations
                - Color-coded impact levels
            """)
            
            # Show a sample visualization placeholder
            st.image("https://via.placeholder.com/600x400/E8F4FD/1E88E5?text=Environmental+Impact+Map+Will+Appear+Here", 
                    caption="Your environmental impact map will appear here")

def create_environmental_map(data: Dict[str, Any], metric_option: str, state: str, power_value: float, power_unit: str, water_value: float, water_unit: str):
    """
    Create and display the environmental impact map for the selected state.
    
    Args:
        data: Dictionary containing the environmental data
        metric_option: Selected environmental metric
        state: Selected state (or "USA" for all)
        power_value: Power consumption value
        power_unit: Power consumption unit
        water_value: Water consumption value
        water_unit: Water consumption unit
    """
    # Map metric names to data arrays
    metric_map = {
        "carbon footprint": data["EFkgkWh"],
        "scope 1 & 2 water footprint": data["EWIF"],
        "water scarcity footprint": data["AWAREUSCF"]
    }
    
    # Get the values for the selected metric
    values = metric_map[metric_option]
    fips = data["CountyFIPS"]
    
    # Convert power to kWh/year for calculations
    power_kwh_per_year = convert_power_to_kwh_per_year(power_value, power_unit) if power_value > 0 else 0
    
    # Create a DataFrame for easier manipulation
    # Convert FIPS codes to strings with leading zeros (5 digits total)
    fips_strings = [str(int(fips_code)).zfill(5) for fips_code in fips]
    
    df = pd.DataFrame({
        "fips": fips_strings,
        "emission_factor": values  # This is the factor per kWh
    })
    
    # Remove any invalid values
    df = df.dropna()
    df = df[df["emission_factor"] > 0]  # Remove zero or negative values
    
    # Calculate the actual environmental impact for the facility
    if power_kwh_per_year > 0:
        df["calculated_impact"] = df["emission_factor"] * power_kwh_per_year
        impact_values = df["calculated_impact"]
        hover_label = f"Facility {metric_option.title()}"
        hover_units = get_metric_units(metric_option)
    else:
        # If no power consumption provided, show the emission factors
        df["calculated_impact"] = df["emission_factor"]
        impact_values = df["emission_factor"]
        hover_label = f"{metric_option.title()} Factor"
        if metric_option == "carbon footprint":
            hover_units = "kg CO₂-eq/kWh"
        elif metric_option == "scope 1 & 2 water footprint":
            hover_units = "L water/kWh"
        elif metric_option == "water scarcity footprint":
            hover_units = "L water-eq/kWh"
        else:
            hover_units = "units/kWh"
    
    # Filter data for selected state
    if state != "USA":
        state_fips_codes = get_state_fips_codes(state)
        if state_fips_codes:
            # Filter to only include counties from the selected state
            df = df[df["fips"].str[:2].isin(state_fips_codes)]
            impact_values = df["calculated_impact"]
            
            if df.empty:
                st.warning(f"No data available for {state}. Please select a different state.")
                return
    
    # Calculate percentiles for color categories based on filtered data
    low_percentile = np.percentile(impact_values, 33)
    high_percentile = np.percentile(impact_values, 66)
    
    # Create color categories
    def categorize_value(val):
        if val <= low_percentile:
            return "Low Impact"
        elif val <= high_percentile:
            return "Medium Impact"
        else:
            return "High Impact"
    
    df["category"] = df["calculated_impact"].apply(categorize_value)
    df["formatted_impact"] = df["calculated_impact"].round(4)  # Round for display
    
    # Create custom hover text with scientific notation for carbon footprint
    def format_hover_value(value, metric):
        if metric == "carbon footprint":
            return f"{value:.2e}"  # Scientific notation with 2 decimal places
        else:
            return f"{value:.4f}"  # Regular formatting for other metrics
    
    df["hover_text"] = df.apply(lambda row: 
        f"FIPS: {row['fips']}<br>" +
        f"{hover_label}: {format_hover_value(row['calculated_impact'], metric_option)} {hover_units}<br>" +
        f"Impact Level: {row['category']}", axis=1)
    
    # Create the choropleth map
    fig = px.choropleth(
        df,
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations="fips",
        color="category",
        color_discrete_map={
            "Low Impact": "#2E8B57",      # Sea Green
            "Medium Impact": "#FFD700",    # Gold
            "High Impact": "#DC143C"       # Crimson
        },
        scope="usa",
        labels={"category": "Impact Level"},
        title=f"{metric_option.title()} by County - {state}" + 
              (f" (Based on {power_kwh_per_year:,.0f} kWh/year)" if power_kwh_per_year > 0 else " (Emission Factors)"),
        hover_name="hover_text",
        hover_data={}  # Clear default hover data since we're using custom hover_name
    )
    
    # Update hover template to show only our custom text
    fig.update_traces(
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=df["hover_text"]
    )
    
    # If a specific state is selected, zoom to that state
    if state != "USA":
        # Set the geographic focus to the selected state
        state_centers = {
            "Alabama": {"lat": 32.806671, "lon": -86.791130},
            "Alaska": {"lat": 61.370716, "lon": -152.404419},
            "Arizona": {"lat": 33.729759, "lon": -111.431221},
            "Arkansas": {"lat": 34.969704, "lon": -92.373123},
            "California": {"lat": 36.116203, "lon": -119.681564},
            "Colorado": {"lat": 39.059811, "lon": -105.311104},
            "Connecticut": {"lat": 41.597782, "lon": -72.755371},
            "Delaware": {"lat": 39.318523, "lon": -75.507141},
            "Florida": {"lat": 27.766279, "lon": -81.686783},
            "Georgia": {"lat": 33.040619, "lon": -83.643074},
            "Idaho": {"lat": 44.240459, "lon": -114.478828},
            "Illinois": {"lat": 40.349457, "lon": -88.986137},
            "Indiana": {"lat": 39.849426, "lon": -86.258278},
            "Iowa": {"lat": 42.011539, "lon": -93.210526},
            "Kansas": {"lat": 38.526600, "lon": -96.726486},
            "Kentucky": {"lat": 37.668140, "lon": -84.670067},
            "Louisiana": {"lat": 31.169546, "lon": -91.867805},
            "Maine": {"lat": 44.693947, "lon": -69.381927},
            "Maryland": {"lat": 39.063946, "lon": -76.802101},
            "Massachusetts": {"lat": 42.230171, "lon": -71.530106},
            "Michigan": {"lat": 43.326618, "lon": -84.536095},
            "Minnesota": {"lat": 45.694454, "lon": -93.900192},
            "Mississippi": {"lat": 32.741646, "lon": -89.678696},
            "Missouri": {"lat": 38.456085, "lon": -92.288368},
            "Montana": {"lat": 47.040182, "lon": -109.633837},
            "Nebraska": {"lat": 41.125370, "lon": -98.268082},
            "Nevada": {"lat": 38.313515, "lon": -117.055374},
            "New Hampshire": {"lat": 43.452492, "lon": -71.563896},
            "New Jersey": {"lat": 40.298904, "lon": -74.756138},
            "New Mexico": {"lat": 34.840515, "lon": -106.248482},
            "New York": {"lat": 42.165726, "lon": -74.948051},
            "North Carolina": {"lat": 35.630066, "lon": -79.806419},
            "North Dakota": {"lat": 47.528912, "lon": -99.784012},
            "Ohio": {"lat": 40.388783, "lon": -82.764915},
            "Oklahoma": {"lat": 35.565342, "lon": -96.928917},
            "Oregon": {"lat": 44.572021, "lon": -122.070938},
            "Pennsylvania": {"lat": 40.590752, "lon": -77.209755},
            "Rhode Island": {"lat": 41.680893, "lon": -71.511780},
            "South Carolina": {"lat": 33.856892, "lon": -80.945007},
            "South Dakota": {"lat": 44.299782, "lon": -99.438828},
            "Tennessee": {"lat": 35.747845, "lon": -86.692345},
            "Texas": {"lat": 31.054487, "lon": -97.563461},
            "Utah": {"lat": 40.150032, "lon": -111.862434},
            "Vermont": {"lat": 44.045876, "lon": -72.710686},
            "Virginia": {"lat": 37.769337, "lon": -78.169968},
            "Washington": {"lat": 47.400902, "lon": -121.490494},
            "West Virginia": {"lat": 38.491226, "lon": -80.954570},
            "Wisconsin": {"lat": 44.268543, "lon": -89.616508},
            "Wyoming": {"lat": 42.755966, "lon": -107.302490}
        }
        
        if state in state_centers:
            center = state_centers[state]
            fig.update_layout(
                geo=dict(
                    center=dict(lat=center["lat"], lon=center["lon"]),
                    projection_scale=6  # Zoom in on the state
                )
            )
    
    # Customize the map appearance
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Display the map
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics
    st.subheader(f"📊 Statistics for {state}")
    
    # Display appropriate statistics based on whether facility data was provided
    if power_kwh_per_year > 0:
        impact_range_text = f"({impact_values.min():.2f} - {impact_values.max():.2f} {hover_units})"
    else:
        impact_range_text = f"({impact_values.min():.4f} - {impact_values.max():.4f} {hover_units})"
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        st.metric(
            "Low Impact Counties",
            f"{len(df[df['category'] == 'Low Impact'])} counties",
            f"≤ {low_percentile:.4f}"
        )
    
    with stat_col2:
        st.metric(
            "Medium Impact Counties",
            f"{len(df[df['category'] == 'Medium Impact'])} counties",
            f"{low_percentile:.4f} - {high_percentile:.4f}"
        )
    
    with stat_col3:
        st.metric(
            "High Impact Counties",
            f"{len(df[df['category'] == 'High Impact'])} counties",
            f"> {high_percentile:.4f}"
        )
    
    # Additional information about the data being displayed
    if power_kwh_per_year > 0:
        st.info(f"📌 **Note**: The map shows calculated {metric_option} values for your facility consuming {power_kwh_per_year:,.0f} kWh/year. Hover over counties to see specific impact values.")
    else:
        st.info(f"📌 **Note**: The map shows emission factors per kWh. Enter your facility's power consumption to see calculated environmental impacts.")

def calculate_facility_impact(power_value: float, power_unit: str, water_value: float, water_unit: str, metric_option: str):
    """
    Calculate and display the environmental impact of the user's facility.
    
    Args:
        power_value: Power consumption value
        power_unit: Power consumption unit
        water_value: Water consumption value
        water_unit: Water consumption unit
        metric_option: Selected environmental metric
    """
    # Convert to standard units
    power_kwh_per_year = convert_power_to_kwh_per_year(power_value, power_unit)
    water_liters_per_year = convert_water_to_liters_per_year(water_value, water_unit)
    
    st.subheader("🏭 Your Facility's Impact")
    
    # Display converted values
    impact_col1, impact_col2 = st.columns(2)
    
    with impact_col1:
        st.metric(
            "Annual Power Consumption",
            f"{power_kwh_per_year:,.0f} kWh/year",
            f"From {power_value} {power_unit}"
        )
    
    with impact_col2:
        st.metric(
            "Annual Water Consumption",
            f"{water_liters_per_year:,.0f} L/year",
            f"From {water_value} {water_unit}"
        )
    
    # Note about impact calculations
    st.info("""
        💡 **How to Use the Map**: 
        - The map now shows your facility's calculated environmental impact for each county
        - Hover over any county to see the specific impact value for your facility in that location
        - Colors represent relative impact levels: Green (lowest 33%), Yellow (middle 33%), Red (highest 33%)
        - Choose locations with green counties to minimize your environmental impact
    """)

# -------------- RUN THE APP --------------
if __name__ == "__main__":
    main()
