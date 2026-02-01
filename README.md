# GoKart Lap Analyzer

An interactive Streamlit application to analyze karting sessions recorded with Garmin devices (`.FIT` files).

The app detects laps using modular, GPS-based methods and provides rich, interactive visualizations including:
- Lap time metrics
- Distance-to-gate analysis
- Interactive GPS track plots (cartesian or map background)
- Per-lap data export

## Features
- Upload Garmin `.FIT` activity files
- Modular lap detection methods (pluggable architecture to explore accuracy on other methods)
- Interactive Plotly visualizations
- Map-based track visualization (OpenStreetMap)
- Auto-generated parameter controls from dataclasses
- Export lap metrics and labeled samples to CSV

## Tech stack
- Python
- Streamlit
- Plotly
- pandas / numpy
- fitparse
- scipy

## Project structure

gokart-lap-analyzer/
├── app.py
├── lap_methods/
│ ├── init.py
│ ├── gps_gate.py
│ └── ui.py
├── requirements.txt
└── .venv/ (ignored)


## Getting started
```bash
git clone <repo-url>
cd gokart-lap-analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
