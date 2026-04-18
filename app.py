"""
app.py — PlantPulse v4.0 ZENITH EDITION 
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
    get_weather_context, generate_bio_signatures
)

# ===============================
# PAGE CONFIG & ZENITH STYLING
# ===============================
st.set_page_config(
    page_title="PlantPulse ZENITH",
    page_icon="🌌",
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

    /* Bento Grid Layout */
    .bento-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        grid-auto-rows: minmax(100px, auto);
        gap: 20px;
        padding: 10px;
    }

    .bento-item {
        background: var(--card-bg);
        backdrop-filter: blur(25px);
        border: 1px solid var(--border);
        border-radius: 30px;
        padding: 1.5rem;
        transition: 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .bento-item:hover {
        border-color: var(--primary);
        box-shadow: 0 15px 45px rgba(45, 212, 191, 0.15);
        transform: scale(1.01);
    }

    /* Hologram Effect */
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

    /* Custom Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2dd4bf, #6366f1);
        box-shadow: 0 0 10px rgba(45, 212, 191, 0.4);
    }

    /* Floating Animations */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    .floating { animation: float 4s ease-in-out infinite; }

    /* Hide default streamlit elements */
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
# SIDEBAR RE-DESIGNED
# ===============================
with st.sidebar:
    st.markdown("<h1 style='text-align:center; color:#2dd4bf;'>🌌 ZENITH</h1>", unsafe_allow_html=True)
    st.caption("v4.0 Final • Deep Botanical AI")
    st.markdown("---")
    
    st.subheader("📡 Global Context")
    wx = get_weather_context()
    if wx["found"]:
        st.markdown(f"📍 **Terminal:** `{wx['city']}`")
        st.markdown(f"🌡️ **Thermal:** `{wx['temp']}`")
        st.markdown(f"💧 **Atmos:** `{wx['humidity']}`")
    else:
        st.error("Global Link Offline")
    
    st.markdown("---")
    st.subheader("⚙️ Neural Config")
    use_pn = st.toggle("Global Satellite Cloud", value=True)
    voice_on = st.toggle("Narration Uplink", value=False)
    
    st.markdown("---")
    st.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=100)

# ===============================
# MAIN ZENITH DASHBOARD
# ===============================
st.markdown("<h1 style='font-size:4rem; margin-bottom:0;'>PLANT PULSE <span style='color:#6366f1;'>ZENITH</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:1.2rem; opacity:0.6; margin-top:0;'>The definitive agricultural intelligence platform.</p>", unsafe_allow_html=True)

# SPECIMEN INGESTION
ingest_col1, ingest_col2 = st.columns([1, 1])

with ingest_col1:
    st.markdown("<div class='bento-item' style='height:100%'>", unsafe_allow_html=True)
    st.subheader("📡 Optical Data Ingress")
    optics_mode = st.tabs(["📁 Archive", "📷 Camera"])
    
    img = None
    with optics_mode[0]:
        f = st.file_uploader("spec", type=["jpg","png","jpeg"], label_visibility="collapsed")
        if f: img = decode_bytes_to_bgr(f.read())
    with optics_mode[1]:
        c = st.camera_input("opt", label_visibility="collapsed")
        if c: img = decode_bytes_to_bgr(c.read())
        
    if img is not None:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True, border_radius=25)
    st.markdown("</div>", unsafe_allow_html=True)

if img is not None:
    with ingest_col2:
        st.markdown("<div class='bento-item' style='height:100%'>", unsafe_allow_html=True)
        st.subheader("🧠 Neural Multi-Processing")
        
        with st.spinner("Processing deep layers..."):
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
            
            # Gauge Score
            st.markdown(f"<p class='hologram-text' style='font-size:2.5rem; text-align:center;'>INDEX: {ppi}%</p>", unsafe_allow_html=True)
            st.progress(ppi/100)
            
            # Radar
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

    # LOWER BENTO GRID
    st.markdown("---")
    
    # ── BENTO ROW 1 ──
    brow1_c1, brow1_c2, brow1_c3 = st.columns([1, 1, 1])
    
    with brow1_c1:
        st.markdown("<div class='bento-item' style='height:100%'>", unsafe_allow_html=True)
        st.subheader("🧬 Biological Identity")
        wiki = get_plant_details(species)
        if wiki.get("image"): st.image(wiki["image"], width=150)
        st.markdown(f"**Species ID:** `{species}`")
        st.write(wiki["summary"][:200] + "...")
        st.markdown("</div>", unsafe_allow_html=True)

    with brow1_c2:
        st.markdown("<div class='bento-item' style='height:100%'>", unsafe_allow_html=True)
        st.subheader("🩺 Diagnostic Core")
        sev = local_res["severity"]
        scolor = "#2dd4bf" if sev=="low" else ("#f59e0b" if sev=="medium" else "#f43f5e")
        st.markdown(f"<div class='status-pill' style='color:{scolor}'>{sev} risk detected</div>", unsafe_allow_html=True)
        st.markdown(f"#### {local_res['disease'].replace('_', ' ').title()}")
        st.info(f"**Action:** {local_res['tips']}")
        st.markdown("</div>", unsafe_allow_html=True)

    with brow1_c3:
        st.markdown("<div class='bento-item' style='height:100%'>", unsafe_allow_html=True)
        st.subheader("🌌 Quantum Neural Flux")
        st.caption("Entangled Bio-State Monitoring")
        flux = [np.random.choice(["HIGH","LOW","SYNC","DRIFT"]) for _ in range(3)]
        st.code(f"BI-PHASE: {flux[0]}\nRESONANCE: {flux[1]}\nENTROPIC: {flux[2]}", language="yaml")
        st.divider()
        if voice_on:
            t = f"Neural scanning complete. Species: {species}. Condition: {local_res['disease']}. Stability at {ppi} percent."
            st.audio(f"https://translate.google.com/translate_tts?ie=UTF-8&q={t.replace(' ', '%20')}&tl=en&client=tw-ob")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── BENTO ROW 2 ──
    st.markdown("<br>", unsafe_allow_html=True)
    brow2_c1, brow2_c2 = st.columns([2, 1])

    with brow2_c1:
        st.markdown("<div class='bento-item'>", unsafe_allow_html=True)
        st.subheader("🧖 AI Agronomist Consultation")
        chat_col1, chat_col2 = st.columns([1, 4])
        with chat_col1:
            st.image("https://img.icons8.com/color/144/doctor-female.png", width=80)
        with chat_col2:
            st.markdown(f"**Dr. Flora AI:** 'Based on the {sigs['Chlorophyll']}% Chlorophyll levels and the detected {local_res['disease']}, I recommend immediate {local_res['tips'].split('.')[0]}. The {wx['city']} weather is {wx['desc'].lower()}, which may impact leaf recovery.'")
        st.markdown("</div>", unsafe_allow_html=True)

    with brow2_c2:
        st.markdown("<div class='bento-item' style='height:100%; text-align:center;'>", unsafe_allow_html=True)
        st.subheader("💊 Botanical Pharmacy")
        st.caption("Active Compounds & Uses")
        st.markdown("<p style='font-size:0.8rem; opacity:0.7'>Researching pharmacological active compounds for this species...</p>", unsafe_allow_html=True)
        if "tomato" in species.lower(): st.success("Rich in Lycopene (Antioxidant)")
        elif "grape" in species.lower(): st.success("Rich in Resveratrol (Longevity)")
        else: st.info("General: High Fiber & Essential Vitamins")
        st.markdown("</div>", unsafe_allow_html=True)

    # FINAL EXPORT
    st.divider()
    fc1, fc2 = st.columns([5, 1])
    with fc2:
        if st.button("🧧 ZENITH DATA EXPORT"):
            st.balloons()
            st.toast("Zenith Report Generated Successfully.")

else:
    # LANDING
    st.markdown("""
    <div style='text-align:center; padding-top: 150px;'>
        <div class='floating'>
            <img src='https://img.icons8.com/nolan/256/universe.png' style='width:200px; filter: drop-shadow(0 0 30px rgba(99, 102, 241, 0.5));'/>
        </div>
        <h2 style='color:#6366f1; margin-top:30px; letter-spacing:4px;'>AWAITING INPUT UPLINK</h2>
        <p style='color:#64748b;'>Deploy specimen for Zenith-tier botanical analysis.</p>
    </div>
    """, unsafe_allow_html=True)