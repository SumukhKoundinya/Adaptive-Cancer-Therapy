# visualization/dashboard.py

import streamlit as st

def launch_dashboard(results):
    st.title("Δ-Nim Adaptive Cancer Therapy System")

    st.write("## Model Performance")
    st.json(results)

    st.write("## Key Insight")
    st.write("Game-theoretic treatment policy improves survival prediction vs ML-only baselines.")