import streamlit as st
import math


def compute_coordinates(x_gtf, y_gtf, z_gtf, arc_deg, collar_deg, z=100, d_0=55):
    """
    Compute coordinates based on GTF parameters.

    Args:
        x_gtf, y_gtf, z_gtf: GTF coordinates
        arc_deg: Arc angle in degrees
        collar_deg: Collar angle in degrees
        z: Fixed z value (default 100 for M1 Frame)
        d_0: Initial distance (default 55)

    Returns:
        dict: Computed coordinates and distances
    """
    # Convert degrees to radians
    arc_rad = math.radians(arc_deg)
    collar_rad = math.radians(collar_deg)

    D = d_0 - ((z - z_gtf) / math.sin(collar_rad))
    x_new = x_gtf + ((D - 55) * math.cos(arc_rad))
    y_new = y_gtf + ((55 - D) * math.cos(collar_rad))
    z_new = z_gtf + ((55 - D) * math.sin(collar_rad))

    return {
        "x": x_new,
        "y": y_new,
        "z": z_new,
        "D": D,
        "L": 205 - D
    }


# Page config
st.set_page_config(
    page_title="GTF Coordinate Calculator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling and no scrolling
st.markdown("""
    <style>
    /* Disable scrolling */
    html, body {
        overflow: hidden !important;
    }
    
    [data-testid="stAppViewContainer"], 
    [data-testid="stApp"] {
        overflow: hidden !important;
    }
    
    .main {
        overflow: hidden !important;
    }
    
    .block-container {
        overflow: hidden !important;
        max-height: 100vh !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    
    /* Hide all scrollbars */
    ::-webkit-scrollbar {
        display: none !important;
    }
    
    * {
        -ms-overflow-style: none !important;
        scrollbar-width: none !important;
    }
    
    /* Compact header */
    .main-header {
        text-align: center;
        padding: 0.5rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 1.5rem;
        line-height: 1.2;
    }
    
    .main-header p {
        margin: 0.2rem 0 0 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    
    .section-header {
        background-color: #f0f2f6;
        padding: 0.4rem 0.8rem;
        border-radius: 5px;
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
    }
    
    .section-header h3 {
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Compact metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.8rem;
    }
    
    /* Compact info boxes */
    [data-testid="stAlert"] {
        padding: 0.4rem 0.8rem;
        margin-bottom: 0.5rem;
    }
    
    /* Compact markdown sections */
    h4 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    hr {
        margin: 0.5rem 0;
    }
    
    .metric-container {
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.3rem 0;
    }
    
    .stNumberInput > div > div > input {
        font-weight: 500;
    }
    
    /* Compact buttons */
    .stButton button {
        padding: 0.4rem 1rem;
    }
    
    /* Reduce column gaps */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Fixed z value for M1 Frame
z = 100.0
d_0 = 55.0
# Main layout with better spacing
col_input, col_spacer, col_output = st.columns([2, 0.3, 2], gap="large")

with col_input:
    st.markdown('<div class="section-header"><h3 style="margin: 0;">📝 Input Parameters</h3></div>', unsafe_allow_html=True)

    st.info("ℹ️ **z is fixed at 100 mm for M1 Frame**")
    st.info("ℹ️ **d0 is fixed at 55 mm for current probes**")

    # GTF Coordinates section
    st.markdown("#### GTF Coordinates")
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        x_gtf = st.number_input("**X (mm)**", value=140.0, format="%.2f", help="X coordinate in GTF system")
    with row1_col2:
        y_gtf = st.number_input("**Y (mm)**", value=140.0, format="%.2f", help="Y coordinate in GTF system")
    with row1_col3:
        z_gtf = st.number_input("**Z (mm)**", value=110.0, format="%.2f", help="Z coordinate in GTF system")

    st.markdown("---")

    # Angle Parameters section
    st.markdown("#### Angle Parameters")
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        arc_deg = st.number_input("**Arc (°)**", value=90.0, format="%.2f", help="Arc angle in degrees")
    with row2_col2:
        collar_deg = st.number_input("**Collar (°)**", value=75.0, format="%.2f", help="Collar angle in degrees")

    # Compute button for better UX
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    compute_btn = st.button("🔄 Recalculate", type="primary", use_container_width=True)

with col_spacer:
    st.markdown("")

with col_output:
    st.markdown('<div class="section-header"><h3 style="margin: 0;">📊 Results</h3></div>', unsafe_allow_html=True)

    # Compute results
    results = compute_coordinates(x_gtf, y_gtf, z_gtf, arc_deg, collar_deg, z, d_0)

    # Computed Coordinates
    st.markdown("#### Computed Coordinates")
    coord_col1, coord_col2, coord_col3 = st.columns(3)

    with coord_col1:
        st.metric(
            label="X",
            value=f"{results['x']:.4f}",
            delta=f"{results['x'] - x_gtf:+.4f} mm",
            help="Computed X coordinate"
        )

    with coord_col2:
        st.metric(
            label="Y",
            value=f"{results['y']:.4f}",
            delta=f"{results['y'] - y_gtf:+.4f} mm",
            help="Computed Y coordinate"
        )

    with coord_col3:
        st.metric(
            label="Z",
            value=f"{results['z']:.4f}",
            delta=f"{results['z'] - z_gtf:+.4f} mm",
            help="Computed Z coordinate"
        )

    st.markdown("---")

    # Distance Measurements
    st.markdown("#### Distance Measurements")
    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        st.metric(
            label="D (Distance)",
            value=f"{results['D']:.4f} mm",
            help="Computed distance D"
        )

    with dist_col2:
        st.metric(
            label="L (Target Distance)",
            value=f"{results['L']:.4f} mm",
            help="Distance to target (205 - D)"
        )


