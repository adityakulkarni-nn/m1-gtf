import streamlit as st
import math
import pandas as pd
from datetime import datetime
import pytz
import os


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

    D = d_0 - ((z_gtf - z) / math.sin(collar_rad))
    delta_D = d_0 - D
    x_new = x_gtf - (delta_D * math.cos(arc_rad))
    y_new = y_gtf + (delta_D * math.cos(collar_rad))
    z_new = z_gtf - (delta_D * math.sin(collar_rad))

    return {
        "x": x_new,
        "y": y_new,
        "z": z_new,
        "D": D,
        "L": 205 - D
    }


def compute_with_angles(x_gtf, y_gtf, z_gtf, arc_deg, collar_deg, z=100, d_0=55):
    """
    Compute coordinates based on GTF parameters with angles.

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
    arc1 = arc_deg
    collar1 = collar_deg
    D = d_0 - ((z_gtf - z) / math.sin(collar_rad))
    delta_D = d_0 - D
    x_new = x_gtf - (delta_D * math.cos(arc_rad))
    if x_new <  50:
        x_new = 50.0
        arc1 = math.degrees(math.acos((x_gtf - x_new) / delta_D))
    elif x_new > 150:
        x_new = 150.0
        arc1 = math.degrees(math.acos((x_gtf - x_new) / delta_D))

    y_new = y_gtf + (delta_D * math.cos(collar_rad))
    if y_new < 70:
        y_new = 70.0
        collar1 = math.degrees(math.acos((y_new - y_gtf) / delta_D))
    elif y_new > 170:
        y_new = 170.0
        collar1 = math.degrees(math.acos((y_new - y_gtf) / delta_D))
    z_new = z_gtf - (delta_D * math.sin(collar_rad))

    return {
        "x": x_new,
        "y": y_new,
        "z": z_new,
        "D": D,
        "L": 205 - D,
        "ARC": arc1,
        "COLLAR": collar1
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
    # Fixed parameters info
    st.markdown("#### Fixed Parameters")
    fixed_col1, fixed_col2, fixed_col3 = st.columns(3)
    with fixed_col1:
        st.metric(label="Z (M1 Frame)", value="100 mm")
    with fixed_col2:
        st.metric(label="D₀ (Initial)", value="55 mm")
    with fixed_col3:
        st.metric(label="L₀ (Reference)", value="150 mm")

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
    compute_btn = st.button("🔄 Recalculate", type="primary", width='stretch')

with col_spacer:
    st.markdown("")

with col_output:

    # Compute results
    results = compute_coordinates(x_gtf, y_gtf, z_gtf, arc_deg, collar_deg, z, d_0)
    results1 = compute_with_angles(x_gtf, y_gtf, z_gtf, arc_deg, collar_deg, z, d_0)

    # Create tabs for Normal and Free Angle modes
    tab1, tab2 = st.tabs(["Normal", "Free Angle"])

    with tab1:
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

    with tab2:
        # Computed Coordinates
        st.markdown("#### Computed Coordinates")
        coord_col1, coord_col2, coord_col3 = st.columns(3)

        with coord_col1:
            st.metric(
                label="X",
                value=f"{results1['x']:.4f}",
                delta=f"{results1['x'] - x_gtf:+.4f} mm",
                help="Computed X coordinate"
            )

        with coord_col2:
            st.metric(
                label="Y",
                value=f"{results1['y']:.4f}",
                delta=f"{results1['y'] - y_gtf:+.4f} mm",
                help="Computed Y coordinate"
            )

        with coord_col3:
            st.metric(
                label="Z",
                value=f"{results1['z']:.4f}",
                delta=f"{results1['z'] - z_gtf:+.4f} mm",
                help="Computed Z coordinate"
            )

        st.markdown("---")

        # Distance Measurements
        st.markdown("#### Distance Measurements")
        dist_col1, dist_col2 = st.columns(2)

        with dist_col1:
            st.metric(
                label="D (Distance)",
                value=f"{results1['D']:.4f} mm",
                help="Computed distance D"
            )

        with dist_col2:
            st.metric(
                label="L (Target Distance)",
                value=f"{results1['L']:.4f} mm",
                help="Distance to target (205 - D)"
            )

        st.markdown("---")

        # Adjusted Angles (only in Free Angle mode)
        st.markdown("#### Adjusted Angles")
        angle_col1, angle_col2 = st.columns(2)

        with angle_col1:
            st.metric(
                label="Arc (Adjusted)",
                value=f"{results1['ARC']:.4f}°",
                delta=f"{results1['ARC'] - arc_deg:+.4f}°",
                help="Adjusted Arc angle"
            )

        with angle_col2:
            st.metric(
                label="Collar (Adjusted)",
                value=f"{results1['COLLAR']:.4f}°",
                delta=f"{results1['COLLAR'] - collar_deg:+.4f}°",
                help="Adjusted Collar angle"
            )

    st.markdown("---")

    # Action Buttons

    # Clear comment if flag is set
    if 'clear_comment' in st.session_state and st.session_state.clear_comment:
        st.session_state.comment_input = ""
        st.session_state.clear_comment = False

    # Comment input with key for clearing
    comment = st.text_input("**Comment (Optional)**", key="comment_input", placeholder="Add a note about this measurement...", help="Optional comment to store with this result")

    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("💾 Record Result", type="secondary", width='stretch'):
            # Get current time in Central timezone
            central_tz = pytz.timezone('America/Chicago')
            now = datetime.now(central_tz)
            
            # Prepare data to append
            new_data = {
                'Date': now.strftime('%Y-%m-%d'),
                'Time': now.strftime('%H:%M:%S'),
                'Updated X': results['x'],
                'Updated Y': results['y'],
                'M1 Z': results['z'],
                'D': results['D'],
                'Distance to Target': results['L'],
                'Comment': comment if comment else ''
            }
            
            excel_file = 'data.xlsx'
            
            # Check if file exists
            if os.path.exists(excel_file):
                # Read existing data
                df = pd.read_excel(excel_file)
                # Append new row
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            else:
                # Create new dataframe
                df = pd.DataFrame([new_data])
            
            # Sort by Date and Time in reverse chronological order (newest first)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by=['Date', 'Time'], ascending=[False, False]).reset_index(drop=True)

            # Replace NaN values with empty strings before saving
            df = df.fillna('')

            # Save to Excel
            df.to_excel(excel_file, index=False)
            st.success(f"✅ Result recorded at {now.strftime('%Y-%m-%d %H:%M:%S')} CST")

            # Set flag to clear comment input on next rerun
            st.session_state.clear_comment = True
            st.rerun()

    with btn_col2:
        if st.button("📋 Show Past Experiments", type="secondary", width='stretch'):
            excel_file = 'data.xlsx'
            if os.path.exists(excel_file):
                st.session_state.show_modal = True
            else:
                st.warning("⚠️ No past experiments found. Record a result first!")

# Modal for showing past experiments
if 'show_modal' in st.session_state and st.session_state.show_modal:
    @st.dialog("Past Experiments", width="large")
    def show_experiments():
        excel_file = 'data.xlsx'
        df = pd.read_excel(excel_file)

        # Replace NaN values with empty strings (especially for Comment column)
        df = df.fillna('')

        # Sort by Date and Time in reverse chronological order (newest first)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by=['Date', 'Time'], ascending=[False, False]).reset_index(drop=True)

        # Action buttons
        btn_col1, btn_col2, = st.columns(2)

        with btn_col1:
            # Also offer Excel download
            from io import BytesIO
            excel_buffer = BytesIO()
            df.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)

            st.download_button(
                label="📥 Download as Excel",
                data=excel_buffer,
                file_name=f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch'
            )

        with btn_col2:
            if st.button("🗑️ Delete Selected", type="secondary", width='stretch'):
                # Get selected row indices
                selected_indices = [i for i, selected in enumerate(st.session_state.selected_experiments) if selected]
                if selected_indices:
                    df_updated = df.drop(selected_indices).reset_index(drop=True)
                    # Sort by Date and Time in reverse chronological order before saving
                    df_updated['Date'] = pd.to_datetime(df_updated['Date'])
                    df_updated = df_updated.sort_values(by=['Date', 'Time'], ascending=[False, False]).reset_index(drop=True)
                    # Replace NaN values with empty strings before saving
                    df_updated = df_updated.fillna('')
                    df_updated.to_excel(excel_file, index=False)
                    st.success(f"✅ Deleted {len(selected_indices)} experiment(s)")
                    # Reset all selections after deletion
                    st.session_state.selected_experiments = [False] * len(df_updated)
                    st.rerun()
                else:
                    st.warning("⚠️ Please select rows to delete")

        # Clear all button
        if st.button("🔄 Clear All", type="secondary", width='stretch'):
            st.session_state.confirm_clear = True

        # Clear all confirmation
        if 'confirm_clear' in st.session_state and st.session_state.confirm_clear:
            st.warning("⚠️ This will delete all experiments. Are you sure?")
            conf_col1, conf_col2 = st.columns(2)
            with conf_col1:
                if st.button("✅ Yes, delete all", type="primary", width='stretch'):
                    # Create empty dataframe with same columns and save
                    df_empty = pd.DataFrame(columns=df.columns)
                    df_empty.to_excel(excel_file, index=False)
                    st.success("✅ All experiments cleared")
                    st.session_state.confirm_clear = False
                    st.session_state.selected_experiments = []
                    st.rerun()
            with conf_col2:
                if st.button("❌ Cancel", width='stretch'):
                    st.session_state.confirm_clear = False
                    st.rerun()

        st.write(f"**Total Experiments:** {len(df)}")

        # Initialize selected rows in session state with correct length
        if 'selected_experiments' not in st.session_state or len(st.session_state.selected_experiments) != len(df):
            st.session_state.selected_experiments = [False] * len(df)

        # Display dataframe with checkboxes in first column
        df_with_checkbox = df.copy()

        # Create header with checkbox
        col_headers = st.columns([0.5] + [1.5] * len(df.columns))
        with col_headers[0]:
            st.write("")
        for idx, col_name in enumerate(df.columns):
            with col_headers[idx + 1]:
                st.write(f"**{col_name}**")

        # Display each row with checkbox
        for row_idx in range(len(df)):
            cols = st.columns([0.5] + [1.5] * len(df.columns))
            with cols[0]:
                st.session_state.selected_experiments[row_idx] = st.checkbox(
                    f"Row {row_idx}",
                    value=st.session_state.selected_experiments[row_idx],
                    key=f"checkbox_{row_idx}",
                    label_visibility="collapsed"
                )
            for col_idx, col_name in enumerate(df.columns):
                with cols[col_idx + 1]:
                    st.write(str(df.iloc[row_idx, col_idx]))


    show_experiments()

    # Reset modal state after dialog is closed (X button or click outside)
    st.session_state.show_modal = False





