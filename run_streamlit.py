#!/usr/bin/env python3
"""
Launcher script for the Streamlit Graph Visualization App
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app"""
    print("🚀 Launching Graph Neural Network vs XGBoost Comparison App")
    print("=" * 60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Check if required directories exist
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./models/xgb_models", exist_ok=True)
    os.makedirs("./models/gnn_models", exist_ok=True)
    
    print("📁 Required directories created/verified")
    print("🌐 Starting Streamlit app...")
    print("📱 The app will open in your browser automatically")
    print("🔗 If it doesn't open, go to: http://localhost:8501")
    print("=" * 60)
    
    # Launch Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == "__main__":
    main() 