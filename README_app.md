# Incident Anomalies Alerts — app.py

## Overview
This file is the main backend application for the Incident Anomalies Alerts project. It is responsible for serving the web interface, handling user requests, and connecting to the anomaly detection pipeline.

## Features
- Serves the web interface using Flask
- Handles video uploads and processing
- Displays detected incidents and anomalies
- Provides access to processed video results

## Requirements
- Python 3.8 or higher
- Flask
- opencv-python
- numpy
- Pillow
- scikit-image
- torch
- transformers
- matplotlib

## Installation
1. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install flask opencv-python numpy Pillow scikit-image torch transformers matplotlib
   ```

## Running the App
1. Make sure you are in the project directory.
2. Run the application:
   ```bash
   python app.py
   ```
3. Open your browser and go to `http://localhost:5000` to access the web interface.

## Notes
- Ensure your input videos are placed in the correct folder as specified in the app configuration.
- Output results will be saved in the designated output directory.
- For advanced anomaly detection, make sure all required machine learning packages are installed.
