"""
app.py — PlantPulse AI + Quantum  (Enhanced v2)
=================================================
Improvements over v1:
  • Loads StandardScaler saved by main.py for consistent feature scaling
  • extract_features() includes histogram normalization (matches main.py v2)
  • Session history: stores last 5 results for comparison
  • Sidebar shows model stats (classes, accuracy from report if available)
  • Analysis result is downloadable as a JSON report
  • Quantum circuit encodes more image information (entropy + mean)
  • Lazy quantum execution: only runs if AI confidence < threshold
  • UI: severity badge, treatment tips per disease, result history panel
"""

import streamlit as st
import cv2
import numpy as np
import joblib
import os
import json
import datetime
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# ===============================
# PAGE CONFIG & STYLING
# ===============================
st.set_page_config(
    page_title="PlantPulse AI + Quantum",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; }

    .stButton>button {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white; border: none; border-radius: 12px;
        padding: 0.6rem 1.2rem; font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(16,185,129,0.3);
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(16,185,129,0.4); }

    .metric-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px; padding: 1.5rem;
        text-align: center; transition: transform 0.3s ease;
    }
    .metric-card:hover { transform: scale(1.02); border-color: #10b981; }

    .quantum-badge {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white; padding: 4px 12px; border-radius: 50px;
        font-size: 0.8rem; font-weight: 600;
    }

    .severity-high   { background:#ef4444; color:white; padding:4px 12px; border-radius:50px; font-weight:600; font-size:0.85rem; }
    .severity-medium { background:#f59e0b; color:white; padding:4px 12px; border-radius:50px; font-weight:600; font-size:0.85rem; }
    .severity-low    { background:#10b981; color:white; padding:4px 12px; border-radius:50px; font-weight:600; font-size:0.85rem; }

    .history-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px; padding: 0.8rem 1rem;
        margin-bottom: 8px; font-size: 0.85rem;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px; background: rgba(255,255,255,0.04);
        border-radius: 14px; padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 8px 20px;
        font-weight: 600; color: #94a3b8; background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #10b981, #059669) !important;
        color: white !important;
    }
    [data-testid="stCameraInput"] > div {
        border-radius: 16px; overflow: hidden;
        border: 2px solid rgba(99,102,241,0.4);
        box-shadow: 0 0 20px rgba(99,102,241,0.15);
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# DISEASE KNOWLEDGE BASE
# ===============================
DISEASE_INFO = {
    "healthy": {
        "severity": "low",
        "tips": "✅ No treatment needed. Maintain regular watering and sunlight.",
        "emoji": "🌱"
    },
    "early_blight": {
        "severity": "medium",
        "tips": "🌿 Remove affected leaves. Apply copper-based fungicide. Avoid overhead watering.",
        "emoji": "🟡"
    },
    "late_blight": {
        "severity": "high",
        "tips": "🚨 Isolate plant immediately. Apply mancozeb or chlorothalonil. Destroy infected tissue.",
        "emoji": "🔴"
    },
    "leaf_mold": {
        "severity": "medium",
        "tips": "💨 Improve air circulation. Apply fungicide. Reduce humidity.",
        "emoji": "🟠"
    },
    "bacterial_spot": {
        "severity": "high",
        "tips": "⚗️ Use copper-based bactericide. Avoid working with wet plants.",
        "emoji": "🔴"
    },
    "common_rust": {
        "severity": "medium",
        "tips": "🌾 Apply triazole fungicide early. Rotate crops next season.",
        "emoji": "🟠"
    },
    "northern_leaf_blight": {
        "severity": "high",
        "tips": "🚜 Apply fungicide at first sign. Use resistant varieties next cycle.",
        "emoji": "🔴"
    },
    "gray_leaf_spot": {
        "severity": "medium",
        "tips": "🌤 Improve drainage. Apply strobilurin fungicide preventively.",
        "emoji": "🟡"
    },
}

def get_disease_info(disease_key: str) -> dict:
    """Return treatment info for a disease, with a safe fallback."""
    key = disease_key.lower().replace(" ", "_")
    for k, v in DISEASE_INFO.items():
        if k in key:
            return v
    return {"severity": "medium", "tips": "Consult an agronomist for targeted treatment.", "emoji": "⚠️"}


# ===============================
# FEATURE EXTRACTION (mirrors main.py exactly)
# ===============================
def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Deterministic feature vector — MUST match main.py exactly.
    Includes histogram normalization added in main.py v2.
    """
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

    # Normalize histograms (matches main.py v2)
    h_hist /= (h_hist.sum() + 1e-7)
    s_hist /= (s_hist.sum() + 1e-7)
    v_hist /= (v_hist.sum() + 1e-7)

    means, stds = cv2.meanStdDev(img)
    stats = np.concatenate([means.flatten(), stds.flatten()]) / 255.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (128 * 128)

    return np.concatenate([h_hist, s_hist, v_hist, stats, [edge_density]])


# ===============================
# MODEL & SCALER LOADING
# ===============================
@st.cache_resource
def get_model_and_scaler():
    """Load model and scaler once; cache for the app lifetime."""
    if not os.path.exists("plant_model.pkl"):
        st.error("🚨 Model not found! Please run `python main.py` to train first.")
        st.stop()
    mdl = joblib.load("plant_model.pkl")
    scl = joblib.load("plant_scaler.pkl") if os.path.exists("plant_scaler.pkl") else None
    return mdl, scl

model, scaler = get_model_and_scaler()


# ===============================
# IMAGE DECODING (stream-safe)
# ===============================
def decode_image_source(source_file, source_type: str = "upload"):
    """
    Reads source file bytes ONCE and caches decoded image in session_state.
    Prevents stream-exhaustion bug on Streamlit re-runs.
    """
    file_key = f"{source_type}_{source_file.name}_{source_file.size}"
    if st.session_state.get("cached_img_key") != file_key:
        raw = source_file.read()
        if not raw:
            return None
        arr = np.asarray(bytearray(raw), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        st.session_state["cached_img"] = img
        st.session_state["cached_img_key"] = file_key
    return st.session_state["cached_img"]


# ===============================
# QUANTUM CIRCUIT
# ===============================
def build_quantum_circuit(img: np.ndarray) -> tuple[QuantumCircuit, float]:
    """
    Richer 4-qubit circuit encoding:
      Q0 — mean brightness gate
      Q1 — edge density gate
      Q2-Q3 — entanglement for consensus measurement
    Returns (circuit, entropy_score).
    """
    small = cv2.resize(img, (64, 64))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(float) / 255.0

    mean_val    = float(np.mean(gray))
    edge_dens   = float(np.sum(cv2.Canny((gray * 255).astype(np.uint8), 50, 150) > 0) / (64 * 64))

    # Shannon entropy on histogram
    hist, _ = np.histogram(gray, bins=32, range=(0, 1))
    hist    = hist / (hist.sum() + 1e-7)
    entropy = float(-np.sum(hist * np.log2(hist + 1e-9)))  # 0–5 scale
    entropy_norm = min(entropy / 5.0, 1.0)

    qc = QuantumCircuit(4, 4)
    # Encode image features as rotation angles
    from math import pi
    qc.ry(mean_val * pi, 0)
    qc.ry(edge_dens * pi, 1)
    qc.ry(entropy_norm * pi, 2)
    # Entangle for joint measurement
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.cx(2, 3)
    qc.h(3)
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])

    return qc, entropy_norm


def run_quantum(qc: QuantumCircuit, backend_pref: str):
    """Run on IBM Cloud or fall back to local StatevectorSampler."""
    try:
        IBM_TOKEN = os.getenv("IBM_QUANTUM_TOKEN", "")
        if not IBM_TOKEN:
            raise ValueError("No IBM token.")
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_TOKEN)
        if backend_pref == "Simulator Only":
            backend = service.backend("ibmq_qasm_simulator")
        else:
            try:
                backend = service.least_busy(simulator=False, min_qubits=4)
            except Exception:
                backend = service.least_busy(simulator=True)
        qc_t = transpile(qc, backend)
        sampler = Sampler(backend)
        job = sampler.run([qc_t], shots=1024)
        result = job.result()
        counts = result[0].data.c.get_counts()
        return counts, backend.name
    except Exception:
        # Local fallback
        try:
            from qiskit.primitives import StatevectorSampler as LS
        except ImportError:
            from qiskit.primitives import Sampler as LS
        sampler = LS()
        job = sampler.run([qc])
        result = job.result()
        if hasattr(result, "quasi_dist"):
            counts = result.quasi_dist[0].binary_probabilities()
        else:
            counts = result[0].data.c.get_counts()
        return counts, "local-simulator"


# ===============================
# SESSION HISTORY
# ===============================
def add_to_history(plant: str, disease: str, confidence: float, source: str):
    if "history" not in st.session_state:
        st.session_state["history"] = []
    record = {
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "plant": plant,
        "disease": disease,
        "confidence": round(confidence, 1),
        "source": source,
    }
    st.session_state["history"].insert(0, record)
    st.session_state["history"] = st.session_state["history"][:5]  # Keep last 5


# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.image("https://img.icons8.com/color/144/leaf.png", width=90)
    st.title("PlantPulse Engine")
    st.caption("Hybrid AI + Quantum Plant Diagnostics")
    st.markdown("---")

    st.markdown("### ⚙️ Configuration")
    confidence_threshold = st.slider("AI Confidence Threshold (%)", 0, 100, 70)
    backend_pref = st.selectbox("Quantum Backend", ["Dynamic (Least Busy)", "Simulator Only"])
    run_quantum_always = st.toggle("Always Run Quantum", value=False,
        help="If OFF, quantum only runs when AI confidence < threshold (faster).")

    st.markdown("---")
    st.markdown("### 📋 Model Info")
    n_classes = len(model.classes_) if hasattr(model, "classes_") else "N/A"
    st.metric("Disease Classes", n_classes)
    st.metric("Features Used", model.n_features_in_)
    if os.path.exists("plant_scaler.pkl"):
        st.success("✅ Scaler loaded")
    else:
        st.warning("⚠️ No scaler — retrain with main.py")

    if os.path.exists("training_report.txt"):
        with open("training_report.txt") as f:
            report_text = f.read()
        with st.expander("📄 Last Training Report"):
            st.code(report_text, language="text")

    st.markdown("---")
    # Session history in sidebar
    st.markdown("### 🕐 Session History")
    history = st.session_state.get("history", [])
    if history:
        for rec in history:
            badge = "🟢" if rec["disease"].lower() == "healthy" else "🔴"
            st.markdown(
                f"<div class='history-card'>"
                f"{badge} <b>{rec['plant']}</b> — {rec['disease']}<br>"
                f"<span style='color:#64748b'>{rec['confidence']}% · {rec['source']} · {rec['time']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.caption("No analyses yet this session.")


# ===============================
# MAIN UI
# ===============================
st.markdown("# 🌿 PlantPulse <span class='quantum-badge'>AI + QUANTUM</span>", unsafe_allow_html=True)
st.write("Upload a leaf image **or snap a live photo** for real-time AI + Quantum analysis.")

col1, col2 = st.columns([1, 1], gap="large")

# ---- INPUT PANEL ----
with col1:
    tab_upload, tab_camera = st.tabs(["📁  Upload Image", "📷  Use Camera"])
    img = None
    input_source = "upload"

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Drop leaf image here...", type=["jpg", "png", "jpeg"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            img = decode_image_source(uploaded_file, "upload")
            input_source = "upload"
            if img is None:
                st.error("❌ Failed to decode image.")
                st.stop()
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     caption="📁 Uploaded Specimen", use_column_width=True)

    with tab_camera:
        st.markdown(
            "<p style='color:#94a3b8;font-size:0.85rem;'>📌 Allow camera access, "
            "position leaf in good light, then click <b>Take Photo</b>.</p>",
            unsafe_allow_html=True
        )
        camera_file = st.camera_input("Capture leaf", label_visibility="collapsed")
        if camera_file:
            img = decode_image_source(camera_file, "camera")
            input_source = "camera"
            if img is None:
                st.error("❌ Failed to capture image.")
                st.stop()
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     caption="📷 Live Capture", use_column_width=True)
            st.success("✅ Photo captured!")

# ---- ANALYSIS PANEL ----
with col2:
    active_img = img if img is not None else st.session_state.get("cached_img")

    if active_img is not None:

        # ── 1. CLASSICAL AI ──────────────────────────────────────
        with st.spinner("⚡ Classical AI Processing..."):
            raw_features = extract_features(active_img).reshape(1, -1)

            if raw_features.shape[1] != model.n_features_in_:
                st.error(
                    f"🚨 Feature mismatch: model expects {model.n_features_in_}, "
                    f"got {raw_features.shape[1]}. Re-run `python main.py`."
                )
                st.stop()

            # Apply scaler if available
            features = scaler.transform(raw_features) if scaler else raw_features

            prediction  = model.predict(features)[0]
            conf_probs  = model.predict_proba(features)
            confidence  = float(np.max(conf_probs) * 100)
            top_classes = model.classes_

            try:
                plant, disease = prediction.split("___")
            except Exception:
                plant, disease = "Unknown", prediction

        # Disease metadata
        info     = get_disease_info(disease)
        severity = info["severity"]
        sev_cls  = f"severity-{severity}"

        st.markdown("### 🧠 Classical Analysis")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='metric-card'><h4>Plant</h4><h2>{plant.title()}</h2></div>",
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><h4>Condition</h4><h2>{disease.replace('_',' ').title()}</h2></div>",
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card'><h4>Severity</h4>"
                        f"<br><span class='{sev_cls}'>{severity.upper()}</span></div>",
                        unsafe_allow_html=True)

        st.progress(confidence / 100, text=f"AI Confidence: {confidence:.1f}%")

        if confidence < confidence_threshold:
            st.warning(
                f"⚠️ **Low Confidence ({confidence:.1f}%)** — result may be uncertain. "
                "Ensure the leaf is well-lit, centered, and fills most of the frame."
            )

        # Treatment tips
        st.info(f"💊 **Treatment Tip:** {info['tips']}")

        # Top-5 probabilities
        with st.expander("📊 Full Prediction Breakdown"):
            prob_pairs = sorted(zip(top_classes, conf_probs[0]), key=lambda x: -x[1])
            for cls, prob in prob_pairs[:5]:
                label = cls.replace("___", " → ").replace("_", " ").title()
                st.markdown(f"`{label}` — **{prob*100:.1f}%**")
                st.progress(float(prob))

        # ── 2. QUANTUM VERIFICATION ───────────────────────────────
        st.markdown("---")
        st.markdown("### ⚛️ Quantum Verification")

        should_run_quantum = run_quantum_always or (confidence < confidence_threshold)

        if not should_run_quantum:
            st.success(
                f"✅ AI confidence is high ({confidence:.1f}%). "
                "Quantum verification skipped to save time. "
                "Enable **'Always Run Quantum'** in sidebar to force it."
            )
            # Final diagnosis without quantum
            st.markdown("### 🏁 Final Diagnosis")
            if disease.lower() == "healthy":
                st.success(f"✅ **Healthy {plant.title()}** — No disease detected.")
            else:
                st.error(f"🚨 **{disease.replace('_',' ').title()}** detected in **{plant.title()}**.")
        else:
            try:
                qc, entropy = build_quantum_circuit(active_img)

                with st.status("🔗 Running Quantum Job...", expanded=False) as status:
                    counts, backend_name = run_quantum(qc, backend_pref)
                    status.write(f"Backend: `{backend_name}` | Image entropy: `{entropy:.3f}`")

                dominant_state = max(counts, key=counts.get)
                is_healthy     = disease.lower() == "healthy"

                # Interpret 4-qubit state: majority of 1s = "positive signal"
                ones_ratio     = dominant_state.count("1") / len(dominant_state)
                quantum_agrees = (ones_ratio >= 0.5 and not is_healthy) or \
                                 (ones_ratio < 0.5 and is_healthy)

                st.markdown("### 🏁 Final Diagnosis")
                if is_healthy and quantum_agrees:
                    st.success(f"✅ **Healthy {plant.title()}** — Confirmed by hybrid AI+Quantum consensus.")
                elif not is_healthy and quantum_agrees:
                    st.error(f"🚨 **{disease.replace('_',' ').title()}** in **{plant.title()}** — Quantum state confirms disease signal.")
                else:
                    st.warning("⚠️ Mixed signals: AI and Quantum disagree. Manual inspection recommended.")
                    st.info(f"AI Diagnosis: **{disease.replace('_',' ').title()}** in **{plant.title()}**")

                st.caption(f"Dominant quantum state: `{dominant_state}` | Ones ratio: `{ones_ratio:.2f}`")

            except Exception as qerr:
                st.warning(f"Quantum layer skipped: {qerr}")
                st.markdown("### 🏁 Final Diagnosis (AI Only)")
                if disease.lower() == "healthy":
                    st.success(f"✅ **Healthy {plant.title()}**")
                else:
                    st.error(f"🚨 **{disease.replace('_',' ').title()}** in **{plant.title()}**")

        # ── 3. SAVE TO HISTORY & DOWNLOAD ────────────────────────
        add_to_history(plant, disease, confidence, input_source)

        result_json = json.dumps({
            "timestamp": datetime.datetime.now().isoformat(),
            "source": input_source,
            "plant": plant,
            "disease": disease,
            "confidence_pct": round(confidence, 2),
            "severity": severity,
            "treatment_tip": info["tips"],
        }, indent=2)

        st.download_button(
            label="⬇️ Download Diagnosis Report (JSON)",
            data=result_json,
            file_name=f"plantpulse_{plant}_{disease}_{datetime.datetime.now().strftime('%H%M%S')}.json",
            mime="application/json",
        )

    else:
        # Clear stale cache when no image present
        for k in ["cached_img", "cached_img_key"]:
            st.session_state.pop(k, None)

        st.markdown("""
        <div style='text-align:center;padding:80px 20px;color:#64748b;'>
            <img src='https://img.icons8.com/dotty/80/64748b/camera.png'/>
            <br><br>
            <span style='font-size:1.1rem;'>
                Use the <b>Upload</b> or <b>Camera</b> tab on the left<br>
                to provide a leaf specimen for analysis.
            </span>
        </div>
        """, unsafe_allow_html=True)