import streamlit as st
from config import MOCK_MODE


def render_header():
    st.markdown("""
    <div class="ps-header">
        <p class="ps-title">📍 PathSnap</p>
        <p class="ps-subtitle">GPS Map Matching and Route Reconstruction</p>
        <p class="ps-desc">
            Upload candidate features for Logistic Regression or a precomputed
            graph result to visualize the supplied route.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if MOCK_MODE:
        st.markdown(
            '<div class="ps-mock-banner">⚠️  DEMO MODE — results shown are simulated and not real predictions</div>',
            unsafe_allow_html=True,
        )
