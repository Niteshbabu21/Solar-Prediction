import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Powered Adaptive Solar Energy Predictor",
    page_icon="⚡",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: radial-gradient(circle at top, #071c3b 0, #020617 55%, #000000 100%);
        color: #e5e7eb;
        font-family: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Glassmorphism cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.80);
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(22px);
    }

    /* Hero title */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        background: linear-gradient(120deg, #38bdf8, #a855f7, #facc15);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: 0.98rem;
        color: #cbd5f5;
        max-width: 620px;
    }

    /* Section headings */
    .section-title {
        font-size: 1.1rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9ca3af;
        margin-bottom: 0.4rem;
    }

    /* Input labels tweak */
    label {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #e5e7eb !important;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 999px;
        padding: 0.7rem 1rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border: 1px solid rgba(248, 250, 252, 0.15);
        background: linear-gradient(135deg, #22c55e, #16a34a);
        transition: all 0.16s ease-in-out;
    }

    .stButton>button:hover {
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 18px 40px rgba(34, 197, 94, 0.45);
        border-color: rgba(248, 250, 252, 0.50);
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: rgba(15,23,42,0.92);
        padding: 1rem 1.2rem;
        border-radius: 1rem;
        border: 1px solid rgba(148, 163, 184, 0.5);
    }

    /* Footer text */
    .footer-text {
        font-size: 0.75rem;
        color: #6b7280;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- MODEL LOADER ----------
@st.cache_resource
def load_model(model_path: str = "system_production_model.pkl"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. "
            "Make sure it is in the same folder as app.py."
        )
    model = joblib.load(path)
    return model


# ---------- HERO SECTION ----------
with st.container():
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="hero-title">
                AI Powered Adaptive<br/>Solar Energy Predictor
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <p class="hero-subtitle">
                Feed in real-time environmental parameters and let the model
                estimate your <b>system production</b>.
                Designed for smart grids, research projects, and next-gen
                sustainable energy dashboards.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-title">Model Snapshot</div>
            """,
            unsafe_allow_html=True,
        )
        st.write("⚙️ **Model:** system_production_model.pkl")
        st.write("🧠 **Engine:** Machine Learning regression")
        st.write("🔋 **Output:** Predicted Solar / System Production")
        st.write("🌐 **Use case:** AI-powered solar forecasting for your project")
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")  # spacing

# ---------- INPUT + PREDICTION LAYOUT ----------
input_col, output_col = st.columns([1.3, 1], gap="large")

with input_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Input Parameters</div>',
        unsafe_allow_html=True
    )
    st.caption("Enter the 7 input features below:")

    # 1. Date-Hour (NMT) - numeric encoding as in your training
    date_hour = st.number_input(
        "Date-Hour (NMT)",
        help="Numeric encoded Date-Hour used during training (for reference only).",
        format="%.2f",
        value=1.00
    )

    # 2. Wind Speed (m/s)
    wind_speed = st.number_input(
        "Wind Speed (m/s)",
        min_value=0.0,
        max_value=50.0,
        value=3.5,
        step=0.1
    )

    # 3. Sunshine (hours)
    sunshine = st.number_input(
        "Sunshine (hours)",
        min_value=0.0,
        max_value=24.0,
        value=6.0,
        step=0.1
    )

    # 4. Air Pressure (hPa)
    air_pressure = st.number_input(
        "Air Pressure (hPa)",
        min_value=800.0,
        max_value=1100.0,
        value=1013.0,
        step=0.5
    )

    # 5. Radiation (W/m²)
    radiation = st.number_input(
        "Radiation (W/m²)",
        min_value=0.0,
        max_value=1500.0,
        value=650.0,
        step=1.0
    )

    # 6. Air Temperature (°C)
    air_temperature = st.number_input(
        "Air Temperature (°C)",
        min_value=-20.0,
        max_value=60.0,
        value=28.0,
        step=0.1
    )

    # 7. Relative Air Humidity (%)
    relative_humidity = st.number_input(
        "Relative Air Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=45.0,
        step=0.5
    )

    st.markdown("---")

    # Predict button
    predict_clicked = st.button("⚡ Predict Solar Energy")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- OUTPUT AREA ----------
with output_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Prediction Output</div>',
        unsafe_allow_html=True
    )

    if predict_clicked:
        try:
            model = load_model()
            features = np.array(
                [[
                    date_hour,
                    wind_speed,
                    sunshine,
                    air_pressure,
                    radiation,
                    air_temperature,
                    relative_humidity
                ]]
            )
            prediction = model.predict(features).item()

            st.metric(
                label="Predicted System / Solar Energy",
                value=f"{prediction:,.2f}",
                delta=None
            )

            st.success(
                f"✅ Prediction generated successfully for Date-Hour (NMT): **{date_hour:.2f}**"
            )

            with st.expander("View raw input vector", expanded=False):
                st.write(features)

        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error("🚨 An error occurred while generating prediction.")
            st.exception(e)
    else:
        st.info("Click **Predict Solar Energy** after entering all inputs.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown(
    """
    <div class="footer-text">
        Built for the project: <b>AI Powered Adaptive Solar Energy Predictor</b> ⚡☀️
    </div>
    """,
    unsafe_allow_html=True,
)
