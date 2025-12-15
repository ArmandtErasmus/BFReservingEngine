# Actuarial Reserving Tool

A Bornhuetter-Ferguson based reserving engine implementing Chain Ladder and Bornhuetter-Ferguson methods with an interactive Streamlit dashboard.

![dashboard](https://github.com/ArmandtErasmus/BFReservingEngine/blob/main/reserving.png)

[Try it out!](https://bornhuetter-ferguson-reserving-engine.streamlit.app/)

## Features

- **Chain Ladder Method**: Industry-standard reserving method based on development factor analysis
- **Bornhuetter-Ferguson Method**: Combines prior expectations with observed development patterns
- **Interactive Dashboard**: Professional Streamlit interface with advanced visualisations
- **Object-Oriented Design**: Clean, extensible code architecture
- **Comprehensive Analysis**: Development factors, incremental patterns, and method comparison

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Dashboard

To launch the interactive Streamlit dashboard:

```bash
streamlit run streamlit_dashboard.py
```

The dashboard will open in your default web browser.

### Using the Python API

```python
from reserving_engine import (
    LossTriangle, ReservingAnalyser, 
    generate_sample_triangle
)

# Generate or load your loss triangle data
triangle_data = generate_sample_triangle(n_periods=10, base_claims=10000)

# Create triangle object
triangle = LossTriangle(triangle_data, is_cumulative=True)

# Create analyser
analyser = ReservingAnalyser(triangle)

# Run Chain Ladder
cl_results = analyser.run_chain_ladder()
print(f"Total Reserve: £{cl_results['total_reserve']:,.2f}")

# Run Bornhuetter-Ferguson
prior_ultimates = cl_results['ultimates'] * 1.1
bf_results = analyser.run_bornhuetter_ferguson(prior_ultimates)
print(f"BF Reserve: £{bf_results['total_reserve']:,.2f}")
```


This will run comprehensive tests on all components and provide a detailed report.

## Project Structure

- `reserving_engine.py`: Core OOP implementation of reserving methods
- `streamlit_dashboard.py`: Interactive Streamlit dashboard
- `requirements.txt`: Python package dependencies


