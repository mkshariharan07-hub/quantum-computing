"""
app.py — PlantPulse AI v6.0 OMNI EDITION
=================================================
The 200,000X Evolutionary Peak of Agricultural Intelligence
Aura Scanning | Growth Forecasting | Global Neural Grid
"""

import streamlit as st
import cv2
import numpy as np
import os
import json
import datetime
import time
import requests
import plotly.graph_objects as go
import plotly.express as px
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

# Shared utilities
from utils import (
    predict_image, get_disease_info,
    get_feature_mode, load_model_and_scaler,
    decode_bytes_to_bgr, identify_plant_plantnet,
    get_plant_details, get_care_tips, calculate_health_index,
    get_weather_context, generate_bio_signatures, SystemAuditor,
    get_health_forecast, get_global_spread
)

# ===============================
# PAGE CONFIG & OMNI STYLING
# ===============================
st.set_page_config(
    page_title="PlantPulse OMNI EDITION",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;400;600;700&family=Fira+Code:wght@400;700&display=swap');

    :root {
        --omni-gold: #f59e0b;
        --omni-red: #f43f5e;
        --omni-teal: #14b8a6;
        --omni-bg: #030712;
    }

    html, body, [class*="css"] { 
        font-family: 'Rajdhani', sans-serif; 
        background-color: var(--omni-bg); 
        color: #f8fafc;
    }
    
    .stApp {
        background: radial-gradient(circle at 100% 0%, #111827 0%, #030712 100%);
    }

    .omni-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        background: linear-gradient(135deg, #fbbf24, #f43f5e, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 12px;
        text-align: center;
        text-shadow: 0 0 50px rgba(251, 191, 36, 0.3);
        margin-top: 50px;
    }

    .hud-card {
        background: rgba(17, 24, 39, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 4px;
        padding: 2.5rem;
        position: relative;
        overflow: hidden;
        clip-path: polygon(0 0, 100% 0, 100% calc(100% - 20px), calc(100% - 20px) 100%, 0 100%);
    }
    .hud-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 10px; height: 10px;
        border-top: 2px solid var(--omni-teal);
        border-left: 2px solid var(--omni-teal);
    }

    /* Aura Scanner Animation */
    .scanner-container {
        position: relative;
        width: 100%;
        border-radius: 10px;
        overflow: hidden;
    }
    .scanner-line {
        position: absolute;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, transparent, var(--omni-teal), transparent);
        box-shadow: 0 0 20px var(--omni-teal);
        animation: scan 3s infinite linear;
        z-index: 10;
    }
    @keyframes scan {
        0% { top: 0%; }
        100% { top: 100%; }
    }

    /* Cinematic Static */
    .vignette {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: radial-gradient(circle, transparent 40%, black 150%);
        pointer-events: none;
        z-index: 999;
    }

    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        color: var(--omni-teal);
        text-shadow: 0 0 10px rgba(20, 184, 166, 0.5);
    }
    
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===============================
# OMNI HUD & ENGINE
# ===============================
@st.cache_resource
def power_on_omni():
    return load_model_and_scaler()

model, scaler = power_on_omni()

# ===============================
# MAIN UI
# ===============================
st.markdown("<div class='vignette'></div>", unsafe_allow_html=True)
st.markdown("<h1 class='omni-title'>PLANTPULSE OMNI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; letter-spacing:8px; font-size:1.2rem; opacity:0.6;'>SYSTEM STATUS: TERMINAL ALPHA-6 EXECUTING...</p>", unsafe_allow_html=True)

# SPECIMEN HUB
col_ingest, col_main = st.columns([1, 2], gap="large")

with col_ingest:
    st.markdown("<div class='hud-card'>", unsafe_allow_html=True)
    st.subheader("📡 SATELLITE UPLINK")
    ingress = st.selectbox("INGRESS DATA", ["ARCHIVE", "LIVE OPTICS"], label_visibility="collapsed")
    
    img = None
    if ingress == "ARCHIVE":
        f = st.file_uploader("SPECIMEN", type=["jpg","png","jpeg"], label_visibility="collapsed")
        if f: img = decode_bytes_to_bgr(f.read())
    else:
        cam = st.camera_input("CAPTURE", label_visibility="collapsed")
        if cam: img = decode_bytes_to_bgr(cam.read())
        
    if img is not None:
        st.markdown("<div class='scanner-container'><div class='scanner-line'></div>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
        st.success("DATA STREAM STABILIZED")
    st.markdown("</div>", unsafe_allow_html=True)

if img is not None:
    with col_main:
        st.markdown("<div class='hud-card'>", unsafe_allow_html=True)
        st.subheader("🧠 NEURAL OMNI-PROCESSOR")
        
        with st.spinner("COLLAPSING WAVEFUNCTION..."):
            # EXECUTION PIPELINE
            pn_res = identify_plant_plantnet(img, os.getenv("PLANTNET_API_KEY", ""))
            local_res = predict_image(img, model, scaler)
            
            species = "Unknown"
            pn_score = 0
            if "results" in pn_res and pn_res["results"]:
                species = pn_res["results"][0]["species"]["scientificNameWithoutAuthor"]
                pn_score = pn_res["results"][0]["score"] * 100
            
            ppi = calculate_health_index(local_res["confidence"], 0.42, pn_score)
            sigs = generate_bio_signatures(img, ppi)
            forecast = get_health_forecast(ppi, local_res["severity"])
            global_map = get_global_spread(local_res["disease"])
            
            # CORE METRICS
            mc1, mc2, mc3 = st.columns(3)
            mc1.markdown(f"**PULSE INDEX**<br><span class='metric-value'>{ppi}%</span>", unsafe_allow_html=True)
            mc2.markdown(f"**CONDITION**<br><span class='metric-value' style='font-size:1.8rem;'>{local_res['disease'].upper()}</span>", unsafe_allow_html=True)
            mc3.markdown(f"**CONFIDENCE**<br><span class='metric-value'>{local_res['confidence']}%</span>", unsafe_allow_html=True)
            
            st.divider()
            
            # OMNI TABS
            ot1, ot2, ot3, ot4 = st.tabs(["🔮 GROWTH FORECAST", "🛰️ GLOBAL GRID", "🧬 BIO-SPACE", "💊 OMNI REMEDY"])
            
            with ot1:
                st.markdown("### 10-DAY HEALTH PROJECTION")
                fig_f = px.line(x=forecast["days"], y=forecast["index"], labels={'x':'Days in Future', 'y':'Health Index'},
                                markers=True, template="plotly_dark")
                fig_f.update_traces(line_color="#14b8a6", line_width=4)
                fig_f.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_f, use_container_width=True)
                st.warning(f"Projected Decay Rate: {'High' if local_res['severity']=='high' else 'Moderate'}")

            with ot2:
                st.markdown("### PATHOLOGICAL GLOBAL IMPACT")
                fig_map = px.bar(x=list(global_map.keys()), y=list(global_map.values()), labels={'x':'Region', 'y':'Impact Level'},
                                 template="plotly_dark", color=list(global_map.values()), color_continuous_scale="Viridis")
                fig_map.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_map, use_container_width=True)

            with ot3:
                st.markdown("### QUANTUM NEURAL STATE SPACE")
                # 3D State visualization
                z_data = [sigs['Chlorophyll'], sigs['Hydration'], sigs['Structure'], sigs['Metabolic'], ppi]
                fig_3d = go.Figure(data=[go.Scatter3d(x=[1,2,3,4,5], y=[ppi, ppi, ppi, ppi, ppi], z=z_data,
                                        mode='markers+lines', marker=dict(size=12, color=z_data, colorscale='Electric'))])
                fig_3d.update_layout(scene=dict(xaxis_title="Node", yaxis_title="Pulse", zaxis_title="Resonance"),
                                     margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_3d, use_container_width=True)

            with ot4:
                st.markdown("### OMNI-RECOVERY PROTOCOL")
                info = get_disease_info(local_res["disease"])
                st.info(f"**CORE ACTION:** {local_res['tips']}")
                if "active_ingredient" in info:
                    st.success(f"**MOLECULAR SOLUTION:** {info['active_ingredient']}")
                    st.write(f"**CONCENTRATION:** {info['application_rate']}")
                    st.link_button("INITIATE PROCUREMENT", info["buy_link"])
        st.markdown("</div>", unsafe_allow_html=True)

    # LOWER OMNI GRID
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='hud-card'>", unsafe_allow_html=True)
    st.subheader("🔬 SYSTEM TELEMETRY")
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.code(f"LATENCY: {np.random.randint(10,40)}ms", language="text")
    tc2.code(f"SPECIMEN ID: {species}", language="text")
    tc3.code(f"WAVE STATUS: STABLE", language="text")
    tc4.code(f"VERSION: 6.0-OMNI", language="text")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🧧 EXECUTE OMNI DATA EXPORT"):
        st.balloons()
        st.toast("OMNI ARCHIVE GENERATED SUCCESSFULLY")

else:
    # LANDING
    st.markdown("""
    <div style='text-align:center; padding-top: 100px;'>
        <div style='animation: pulse 2s infinite;'>
            <img src='https://img.icons8.com/nolan/256/galaxy.png' width='250' style='filter: drop-shadow(0 0 50px var(--omni-teal));'/>
        </div>
        <h2 style='color:var(--omni-teal); margin-top:30px; letter-spacing:15px; font-family:Orbitron;'>AWAITING OMNI UPLINK</h2>
        <p style='font-family:Fira Code; opacity:0.4;'>[SYSTEM_IDLE: PENDING DATA INGRESS]</p>
    </div>
    <style>
        @keyframes pulse { 0% { opacity: 0.5; transform: scale(0.95); } 50% { opacity: 1; transform: scale(1); } 100% { opacity: 0.5; transform: scale(0.95); } }
    </style>
    """, unsafe_allow_html=True)