import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from reserving_engine import (
    LossTriangle, ReservingAnalyser, 
    ChainLadderMethod, BornhuetterFergusonMethod,
    generate_sample_triangle
)

# Page configuration
st.set_page_config(
    page_title="Bornhuetter-Ferguson Based Reserving Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #6865F2;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #6865F2;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #6865F2;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown(
    '<h1 class="main-header">📊 Bornhuetter-Ferguson Based Reserving Engine</h1>', 
    unsafe_allow_html=True
)

st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        This interactive dashboard provides comprehensive loss reserving analysis 
        using industry-standard methods including Chain Ladder and Bornhuetter-Ferguson.
        Upload your loss triangle data or use the sample data generator to explore 
        the tool's capabilities.
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar for inputs
with st.sidebar:
    st.header("📋 Data Input")
    
    data_source = st.radio(
        "Data Source",
        ["Sample Data Generator", "Upload CSV File"],
        help="Choose to generate sample data or upload your own"
    )
    
    if data_source == "Sample Data Generator":
        n_periods = st.slider(
            "Number of Periods",
            min_value=5,
            max_value=15,
            value=10,
            help="Number of accident and development periods"
        )
        
        base_claims = st.number_input(
            "Base Claims Amount (£)",
            min_value=1000.0,
            max_value=100000.0,
            value=10000.0,
            step=1000.0,
            format="%.0f"
        )
        
        volatility = st.slider(
            "Volatility",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Claim development volatility"
        )
        
        if st.button("🔄 Generate Sample Data", type="primary"):
            # Generate sample triangle
            triangle_data = generate_sample_triangle(
                n_periods, base_claims, volatility
            )
            st.session_state['triangle_data'] = triangle_data
            st.session_state['n_periods'] = n_periods
            st.session_state['data_source'] = 'generated'
            st.success("Sample data generated successfully!")
    
    else:
        uploaded_file = st.file_uploader(
            "Upload Loss Triangle CSV",
            type=['csv'],
            help="CSV file with loss triangle data (square matrix)"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, index_col=0)
                triangle_data = df.values
                if triangle_data.shape[0] != triangle_data.shape[1]:
                    st.error("Triangle must be square (equal rows and columns)")
                else:
                    st.session_state['triangle_data'] = triangle_data
                    st.session_state['n_periods'] = triangle_data.shape[0]
                    st.session_state['data_source'] = 'uploaded'
                    st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    st.markdown("---")
    st.header("⚙️ Method Selection")
    
    run_chain_ladder = st.checkbox("Chain Ladder Method", value=True)
    run_bf = st.checkbox("Bornhuetter-Ferguson Method", value=False)
    
    if run_bf:
        bf_prior_type = st.selectbox(
            "Prior Ultimates Source",
            ["Chain Ladder Ultimates", "Scaled Chain Ladder", "Manual Input"],
            help="Source for prior ultimate estimates"
        )
        
        if bf_prior_type == "Scaled Chain Ladder":
            scale_factor = st.slider(
                "Scale Factor",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Multiplier for Chain Ladder ultimates"
            )
        elif bf_prior_type == "Manual Input":
            st.text_area(
                "Enter prior ultimates (comma-separated)",
                help="Enter values separated by commas, e.g., 15000, 16000, 17000"
            )

# Main analysis area
if 'triangle_data' in st.session_state:
    triangle_data = st.session_state['triangle_data']
    n_periods = st.session_state['n_periods']
    
    triangle = LossTriangle(triangle_data, is_cumulative=True)
    analyser = ReservingAnalyser(triangle)
    
    # Key metrics row
    st.subheader("📊 Input Loss Triangle")
    
    # Create heatmap of input triangle
    fig_triangle = go.Figure(data=go.Heatmap(
        z=triangle_data,
        colorscale='Blues',
        text=triangle_data,
        texttemplate='%{text:,.0f}',
        textfont={"size": 10},
        showscale=True,
        hovertemplate='Accident: %{y}<br>Development: %{x}<br>Value: £%{z:,.2f}<extra></extra>'
    ))
    
    fig_triangle.update_layout(
        title="Cumulative Loss Triangle",
        xaxis_title="Development Period",
        yaxis_title="Accident Period",
        height=500,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig_triangle, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_observed = triangle.get_latest_observed()
        st.metric(
            label="Total Latest Observed",
            value=f"£{np.sum(latest_observed):,.2f}"
        )
    
    with col2:
        st.metric(
            label="Number of Periods",
            value=f"{n_periods}"
        )
    
    with col3:
        st.metric(
            label="Average Latest Observed",
            value=f"£{np.mean(latest_observed):,.2f}"
        )
    
    with col4:
        st.metric(
            label="Max Latest Observed",
            value=f"£{np.max(latest_observed):,.2f}"
        )
    
    st.markdown("---")
    
    # Run analyses
    results_available = False
    cl_results = None
    bf_results = None
    
    if run_chain_ladder:
        st.subheader("🔗 Chain Ladder Analysis")
        cl_results = analyser.run_chain_ladder()
        results_available = True
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Reserve (CL)",
                value=f"£{cl_results['total_reserve']:,.2f}",
                delta=f"{cl_results['total_reserve']/np.sum(latest_observed)*100:.1f}% of observed"
            )
        
        with col2:
            st.metric(
                label="Total Ultimate (CL)",
                value=f"£{np.sum(cl_results['ultimates']):,.2f}"
            )
        
        with col3:
            st.metric(
                label="Average Development Factor",
                value=f"{np.mean(cl_results['development_factors']):.4f}"
            )
        
        # Development factors visualisation
        col1, col2 = st.columns(2)
        
        with col1:
            fig_factors = go.Figure()
            fig_factors.add_trace(go.Bar(
                x=[f"Period {i+1}→{i+2}" for i in range(len(cl_results['development_factors']))],
                y=cl_results['development_factors'],
                marker_color='#6865F2',
                text=[f"{x:.4f}" for x in cl_results['development_factors']],
                textposition='outside'
            ))
            fig_factors.update_layout(
                title="Development Factors (Age-to-Age)",
                xaxis_title="Development Period Transition",
                yaxis_title="Development Factor",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_factors, use_container_width=True)
        
        with col2:
            # Reserve by accident period
            fig_reserves = go.Figure()
            fig_reserves.add_trace(go.Bar(
                x=[f"Acc {i}" for i in range(n_periods)],
                y=cl_results['reserves'],
                marker_color='#5DFFBC',
                text=[f"£{x:,.0f}" for x in cl_results['reserves']],
                textposition='outside'
            ))
            fig_reserves.update_layout(
                title="Reserves by Accident Period (Chain Ladder)",
                xaxis_title="Accident Period",
                yaxis_title="Reserve Amount (£)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_reserves, use_container_width=True)
        
        # Projected triangle heatmap
        st.subheader("📈 Projected Triangle (Chain Ladder)")
        fig_projected = go.Figure(data=go.Heatmap(
            z=cl_results['projected_triangle'],
            colorscale='Viridis',
            text=cl_results['projected_triangle'],
            texttemplate='%{text:,.0f}',
            textfont={"size": 9},
            showscale=True,
            hovertemplate='Accident: %{y}<br>Development: %{x}<br>Value: £%{z:,.2f}<extra></extra>'
        ))
        fig_projected.update_layout(
            title="Full Projected Cumulative Triangle",
            xaxis_title="Development Period",
            yaxis_title="Accident Period",
            height=500
        )
        st.plotly_chart(fig_projected, use_container_width=True)
    
    if run_bf:
        st.markdown("---")
        st.subheader("📈 Bornhuetter-Ferguson Analysis")
        
        # Determine prior ultimates
        if 'bf_prior_type' in locals():
            if bf_prior_type == "Chain Ladder Ultimates":
                if not run_chain_ladder:
                    st.warning("Chain Ladder must be run first to use CL ultimates as priors")
                    cl_results = analyser.run_chain_ladder()
                prior_ultimates = cl_results['ultimates']
            elif bf_prior_type == "Scaled Chain Ladder":
                if not run_chain_ladder:
                    cl_results = analyser.run_chain_ladder()
                prior_ultimates = cl_results['ultimates'] * scale_factor
            else:
                # Manual input - use default for now
                prior_ultimates = latest_observed * 1.5
        else:
            if run_chain_ladder:
                prior_ultimates = cl_results['ultimates']
            else:
                prior_ultimates = latest_observed * 1.5
        
        bf_results = analyser.run_bornhuetter_ferguson(prior_ultimates)
        results_available = True
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            delta_value = None
            if cl_results is not None:
                delta_value = f"£{bf_results['total_reserve'] - cl_results['total_reserve']:,.2f} vs CL"
            st.metric(
                label="Total Reserve (BF)",
                value=f"£{bf_results['total_reserve']:,.2f}",
                delta=delta_value
            )
        
        with col2:
            st.metric(
                label="Total Ultimate (BF)",
                value=f"£{np.sum(bf_results['ultimates']):,.2f}"
            )
        
        with col3:
            avg_dev_ratio = np.mean(bf_results['development_ratios'])
            st.metric(
                label="Average Development Ratio",
                value=f"{avg_dev_ratio:.2%}"
            )
        
        # BF vs CL comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bf_reserves = go.Figure()
            fig_bf_reserves.add_trace(go.Bar(
                x=[f"Acc {i}" for i in range(n_periods)],
                y=bf_results['reserves'],
                marker_color='#FF6B6B',
                text=[f"£{x:,.0f}" for x in bf_results['reserves']],
                textposition='outside',
                name='BF Reserve'
            ))
            fig_bf_reserves.update_layout(
                title="Reserves by Accident Period (Bornhuetter-Ferguson)",
                xaxis_title="Accident Period",
                yaxis_title="Reserve Amount (£)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_bf_reserves, use_container_width=True)
        
        with col2:
            # Development ratios
            fig_dev_ratios = go.Figure()
            fig_dev_ratios.add_trace(go.Bar(
                x=[f"Acc {i}" for i in range(n_periods)],
                y=bf_results['development_ratios'] * 100,
                marker_color='#4ECDC4',
                text=[f"{x:.1f}%" for x in bf_results['development_ratios'] * 100],
                textposition='outside'
            ))
            fig_dev_ratios.update_layout(
                title="Development Ratios (Observed/CL Ultimate)",
                xaxis_title="Accident Period",
                yaxis_title="Development Ratio (%)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_dev_ratios, use_container_width=True)
    
    # Method comparison
    if run_chain_ladder and run_bf and cl_results is not None:
        st.markdown("---")
        st.subheader("📉 Method Comparison")
        
        comparison_df = pd.DataFrame({
            'Accident Period': [f"Acc {i}" for i in range(n_periods)],
            'Latest Observed': latest_observed,
            'Chain Ladder Reserve': cl_results['reserves'],
            'Bornhuetter-Ferguson Reserve': bf_results['reserves'],
            'CL Ultimate': cl_results['ultimates'],
            'BF Ultimate': bf_results['ultimates']
        })
        
        # Comparison chart
        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(
            name='Chain Ladder',
            x=comparison_df['Accident Period'],
            y=comparison_df['Chain Ladder Reserve'],
            marker_color='#6865F2'
        ))
        fig_comparison.add_trace(go.Bar(
            name='Bornhuetter-Ferguson',
            x=comparison_df['Accident Period'],
            y=comparison_df['Bornhuetter-Ferguson Reserve'],
            marker_color='#FF6B6B'
        ))
        
        fig_comparison.update_layout(
            title="Reserve Comparison by Accident Period",
            xaxis_title="Accident Period",
            yaxis_title="Reserve Amount (£)",
            barmode='group',
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Ultimate comparison
        fig_ultimate_comp = go.Figure()
        fig_ultimate_comp.add_trace(go.Scatter(
            x=comparison_df['Accident Period'],
            y=comparison_df['CL Ultimate'],
            mode='lines+markers',
            name='Chain Ladder Ultimate',
            line=dict(color='#6865F2', width=3),
            marker=dict(size=10)
        ))
        fig_ultimate_comp.add_trace(go.Scatter(
            x=comparison_df['Accident Period'],
            y=comparison_df['BF Ultimate'],
            mode='lines+markers',
            name='Bornhuetter-Ferguson Ultimate',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=10)
        ))
        fig_ultimate_comp.add_trace(go.Scatter(
            x=comparison_df['Accident Period'],
            y=comparison_df['Latest Observed'],
            mode='lines+markers',
            name='Latest Observed',
            line=dict(color='#5DFFBC', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        fig_ultimate_comp.update_layout(
            title="Ultimate Claims Comparison",
            xaxis_title="Accident Period",
            yaxis_title="Amount (£)",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_ultimate_comp, use_container_width=True)
        
        # Reserve development pattern
        if run_chain_ladder:
            st.subheader("📊 Incremental Development Pattern")
            
            incremental_factors = analyser.calculate_incremental_factors(
                cl_results['ultimates']
            )
            
            fig_development = go.Figure()
            fig_development.add_trace(go.Scatter(
                x=list(range(len(incremental_factors))),
                y=incremental_factors * 100,
                mode='lines+markers',
                name='Incremental Development %',
                line=dict(color='#6865F2', width=3),
                marker=dict(size=12, color='#6865F2'),
                fill='tozeroy',
                fillcolor='rgba(104, 101, 242, 0.2)'
            ))
            
            fig_development.update_layout(
                title="Incremental Development Factors (Proportion of Ultimate)",
                xaxis_title="Development Period",
                yaxis_title="Percentage of Ultimate (%)",
                height=450,
                showlegend=False
            )
            
            st.plotly_chart(fig_development, use_container_width=True)
    
    # Detailed results table
    if results_available:
        st.markdown("---")
        st.subheader("📋 Detailed Results Table")
        
        results_df = pd.DataFrame({
            'Accident Period': [f"Acc {i}" for i in range(n_periods)],
            'Latest Observed': latest_observed
        })
        
        if run_chain_ladder:
            results_df['CL Ultimate'] = cl_results['ultimates']
            results_df['CL Reserve'] = cl_results['reserves']
            results_df['CL Reserve %'] = (
                cl_results['reserves'] / cl_results['ultimates'] * 100
            )
        
        if run_bf:
            results_df['BF Ultimate'] = bf_results['ultimates']
            results_df['BF Reserve'] = bf_results['reserves']
            results_df['BF Reserve %'] = (
                bf_results['reserves'] / bf_results['ultimates'] * 100
            )
            results_df['Prior Ultimate'] = bf_results['prior_ultimates']
            results_df['Development Ratio'] = bf_results['development_ratios']
        
        # Format the dataframe
        format_dict = {}
        for col in results_df.columns:
            if 'Latest Observed' in col or 'Ultimate' in col or 'Reserve' in col:
                format_dict[col] = '{:,.2f}'
            elif '%' in col:
                format_dict[col] = '{:.2f}%'
            elif 'Ratio' in col:
                format_dict[col] = '{:.2%}'
        
        styled_df = results_df.style.format(format_dict)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="reserving_results.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Please configure data input in the sidebar to begin analysis.")
    st.markdown("""
    ### Getting Started:
    1. Choose a data source (Sample Data Generator or Upload CSV)
    2. Configure your data parameters
    3. Select reserving methods to run
    4. Explore the interactive visualisations and results
    """)

