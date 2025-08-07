#!/usr/bin/env python3
"""
Script to run the Streamlit GNN Inference Explorer app
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Streamlit is not installed. Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
    
    # Check if required data file exists
    if not os.path.exists('./preprocessed_fraud_test.csv'):
        print("Warning: preprocessed_fraud_test.csv not found.")
        print("Please ensure the data file is in the current directory.")
        print("The app will attempt to load the data file when started.")
    
    # Run the Streamlit app
    print("Starting Streamlit GNN Inference Explorer...")
    print("The app will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the app.")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_inference_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nApp stopped by user.")
    except Exception as e:
        print(f"Error running Streamlit app: {e}")

if __name__ == "__main__":
    main() 