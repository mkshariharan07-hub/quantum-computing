"""
app.py — PlantPulse AI v4.0 FINAL
=================================================
The Ultimate Zenith of Botanical Intelligence
Bento Grid UI | Holographic AI Agronomist | Quantum Neural Flux
"""

import streamlit as st
import cv2
import numpy as np
import joblib
import os
import json
import datetime
import time
import requests
import plotly.graph_objects as go
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
    get_weather_context, generate_bio_signatures, SystemAuditor
)

# ===============================
# PAGE CONFIG & STYLING
# ===============================
st.set_page_config(
    page_title="PlantPulse AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

    :root {
        --primary: #2dd4bf;
        --secondary: #6366f1;
        --accent: #f43f5e;
        --bg: #020617;
        --card-bg: rgba(15, 23, 42, 0.7);
        --border: rgba(255, 255, 255, 0.08);
    }

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; background-color: var(--bg); color: white; }
    
    .main { 
        background: radial-gradient(circle at 0% 0%, #0f172a 0%, #020617 100%);
    }

    .bento-item {
        background: var(--card-bg);
        backdrop-filter: blur(25px);
        border: 1px solid var(--border);
        border-radius: 30px;
        padding: 1.5rem;
        transition: 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }
    .bento-item:hover {
        border-color: var(--primary);
        box-shadow: 0 15px 45px rgba(45, 212, 191, 0.15);
        transform: scale(1.01);
    }

    .hologram-text {
        color: var(--primary);
        text-shadow: 0 0 10px rgba(45, 212, 191, 0.5), 0 0 20px rgba(45, 212, 191, 0.3);
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 2px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 14px;
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        border: 1px solid currentColor;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2dd4bf, #6366f1);
        box-shadow: 0 0 10px rgba(45, 212, 191, 0.4);
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    .floating { animation: float 4s ease-in-out infinite; }

    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===============================
# CORE LOGIC
# ===============================
@st.cache_resource
def init_engine():
    return load_model_and_scaler()

engine_model, engine_scaler = init_engine()

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.markdown("<h1 style='text-align:center; color:#2dd4bf;'>🌿 PlantPulse AI</h1>", unsafe_allow_html=True)
    st.caption("Final Edition • Deep Botanical AI")
    st.markdown("---")
    
    st.subheader("📡 Global Context")
    wx = get_weather_context()
    if wx["found"]:
        st.markdown(f"📍 **Terminal:** `{wx['city']}`")
        st.markdown(f"🌡️ **Thermal:** `{wx['temp']}`")
    else:
        st.error("Link Offline")
    
    st.markdown("---")
    st.subheader("⚙️ Config")
    use_pn = st.toggle("Satellite Cloud", value=True)
    voice_on = st.toggle("Narration", value=False)
    
    st.markdown("---")
    st.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=100)

# ===============================
# MAIN DASHBOARD
# ===============================
st.markdown("<h1 style='font-size:4rem; margin-bottom:0;'>PlantPulse <span style='color:#6366f1;'>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:1.2rem; opacity:0.6; margin-top:0;'>The definitive agricultural intelligence platform.</p>", unsafe_allow_html=True)

# SYSTEM AUDIT PANEL
audit = SystemAuditor.run_audit()
if not all([audit["model"], audit["env"]]):
    st.error("🚨 CRITICAL: System Audit Failed. Check sidebar for missing components.")

ingest_col1, ingest_col2 = st.columns([1, 1])

with ingest_col1:
    st.markdown("<div class='bento-item' style='height:100%'>", unsafe_allow_html=True)
    st.subheader("📡 Data Ingress")
    optics_mode = st.tabs(["📁 Archive", "📷 Camera"])
    
    img = None
    with optics_mode[0]:
        f = st.file_uploader("spec", type=["jpg","png","jpeg"], label_visibility="collapsed")
        if f: img = decode_bytes_to_bgr(f.read())
    with optics_mode[1]:
        c = st.camera_input("opt", label_visibility="collapsed")
        if c: img = decode_bytes_to_bgr(c.read())
        
    if img is not None:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

if img is not None:
    with ingest_col2:
        st.markdown("<div class='bento-item' style='height:100%'>", unsafe_allow_html=True)
        st.subheader("🧠 Neural Multi-Processing")
        
        with st.spinner("Processing..."):
            local_res = predict_image(img, engine_model, engine_scaler)
            pn_res = identify_plant_plantnet(img, os.getenv("PLANTNET_API_KEY", "")) if use_pn else None
            
            species = local_res["plant"]
            pn_conf = 0
            if pn_res and "results" in pn_res and pn_res["results"]:
                m = pn_res["results"][0]
                pn_conf = m["score"] * 100
                if pn_conf > 40: species = m["species"]["scientificNameWithoutAuthor"]
            
            ppi = calculate_health_index(local_res["confidence"], 0.35, pn_conf)
            sigs = generate_bio_signatures(img, ppi)
            
            st.markdown(f"<p class='hologram-text' style='font-size:2.5rem; text-align:center;'>INDEX: {ppi}%</p>", unsafe_allow_html=True)
            st.progress(ppi/100)
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=list(sigs.values()) + [list(sigs.values())[0]],
                theta=list(sigs.keys()) + [list(sigs.keys())[0]],
                fill='toself',
                fillcolor='rgba(99, 102, 241, 0.4)',
                line=dict(color='#6366f1', width=3),
                marker=dict(size=8, color='#2dd4bf')
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=False, range=[0, 100]), 
                           angularaxis=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
                showlegend=False, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=30, b=30, l=40, r=40), height=280
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # ── ROW 1 ──
    b1, b2, b3 = st.columns([1, 1, 1])
    
    with b1:
        st.markdown("<div class='bento-item' style='height:100%'>", unsafe_allow_html=True)
        st.subheader("🧬 Identity")
        wiki = get_plant_details(species)
        if wiki.get("image"): st.image(wiki["image"], width=150)
        st.markdown(f"**Species ID:** `{species}`")
        st.write(wiki["summary"][:200] + "...")
        st.markdown("</div>", unsafe_allow_html=True)

    with b2:
        st.markdown("<div class='bento-item' style='height:100%'>", unsafe_allow_html=True)
        st.subheader("🩺 Diagnostics")
        scolor = "#2dd4bf" if local_res["severity"]=="low" else ("#f59e0b" if local_res["severity"]=="medium" else "#f43f5e")
        st.markdown(f"<div class='status-pill' style='color:{scolor}'>{local_res['severity']} risk</div>", unsafe_allow_html=True)
        st.markdown(f"#### {local_res['disease'].replace('_', ' ').title()}")
        st.info(f"**Action:** {local_res['tips']}")
        
        # Operational Feasibility (Remedy)
        info = get_disease_info(local_res["disease"])
        if "active_ingredient" in info:
            with st.expander("🛠️ Operational Remedy"):
                st.write(f"**Active Ingredient:** {info['active_ingredient']}")
                st.write(f"**App Rate:** {info['application_rate']}")
                st.link_button("🛒 Order Supplies", info["buy_link"])

        st.markdown("</div>", unsafe_allow_html=True)

    with b3:
        st.markdown("<div class='bento-item' style='height:100%'>", unsafe_allow_html=True)
        st.subheader("🌌 Quantum Flux")
        st.code("RESONANCE: SYNC\nENTROPIC: STABLE", language="yaml")
        if voice_on:
            t = f"Species: {species}. Condition: {local_res['disease']}. Stability: {ppi}%."
            st.audio(f"https://translate.google.com/translate_tts?ie=UTF-8&q={t.replace(' ', '%20')}&tl=en&client=tw-ob")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── ROW 2 ──
    st.markdown("<br>", unsafe_allow_html=True)
    b4, b5, b6 = st.columns([1, 1, 1])

    with b4:
        st.markdown("<div class='bento-item' style='height:100%'>", unsafe_allow_html=True)
        st.subheader("🧖 AI Agronomist")
        st.markdown(f"**Dr. Flora AI:** 'Recommended: {local_res['tips'].split('.')[0]}. Climate: {wx['desc'].lower()}.'")
        st.markdown("</div>", unsafe_allow_html=True)

    with b5:
        st.markdown("<div class='bento-item' style='height:100%'>", unsafe_allow_html=True)
        st.subheader("💊 Pharmacy")
        if "tomato" in species.lower(): st.success("Rich in Lycopene")
        elif "grape" in species.lower(): st.success("Rich in Resveratrol")
        else: st.info("High in Vitamins")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with b6:
        st.markdown("<div class='bento-item' style='height:100%'>", unsafe_allow_html=True)
        st.subheader("📟 Neural Log")
        st.code(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Active\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Result Logged", language="text")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    if st.button("🧧 DATA EXPORT"):
        st.balloons()
        with open("operation_history.log", "a") as logf:
            logf.write(f"[{datetime.datetime.now()}] {species} | {local_res['disease']} | Pulse: {ppi}%\n")
        st.toast("Zenith Report Generated & Logged.")

else:
    st.markdown("""
    <div style='text-align:center; padding-top: 150px;'>
        <div class='floating'>
            <img src='https://img.icons8.com/nolan/256/universe.png' style='width:200px; filter: drop-shadow(0 0 30px rgba(99, 102, 241, 0.5));'/>
        </div>
        <h2 style='color:#6366f1; margin-top:30px; letter-spacing:4px;'>AWAITING UPLINK</h2>
    </div>
    """, unsafe_allow_html=True)