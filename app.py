import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import date

# 1. Page setup
st.set_page_config(page_title="Auto Evaluate", page_icon="🚘", layout="wide")

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "Car_Raw_Data.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "best_used_car_price_pipeline.joblib"
SCORES_PATH = BASE_DIR / "artifacts" / "Model_Comparison_Results.csv"
HERO_PATH = BASE_DIR / "assets" / "purple_car_hero.png"

# 2. Load the dataset, saved model and model scores
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_scores():
    if SCORES_PATH.exists():
        scores = pd.read_csv(SCORES_PATH)
        if "Model" in scores.columns and "R2" in scores.columns:
            return scores[["Model", "R2"]].dropna()
    return pd.DataFrame({
        "Model": ["Linear Regression", "Decision Tree", "Random Forest", "KNN Regressor"],
        "R2": [0.637, 0.754, 0.848, 0.903]
    })

try:
    df = load_data()
    model = load_model()
    scores = load_scores()
except Exception:
    st.error("Could not load the dataset or saved model. Check the data and artifacts folders.")
    st.stop()

# 3. Black and purple design with subtle 3D effects
st.markdown("""
<style>
.stApp {
 background: radial-gradient(circle at 15% 0%,rgba(147,51,234,.20),transparent 30%),
 linear-gradient(135deg,#08070d,#130d20 55%,#090811);
 color:#f8f6ff;
}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:#100b1a;border-right:1px solid #352044;}
.hero {
 position:relative; overflow:hidden; padding:28px; min-height:260px; border-radius:24px;
 background:linear-gradient(110deg,rgba(13,10,22,.98),rgba(36,13,55,.85));
 border:1px solid rgba(192,132,252,.30);
 box-shadow:0 22px 55px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.08);
 margin-bottom:20px;
}
.hero-copy {position:relative;z-index:2;max-width:58%;}
.eyebrow {color:#d58aff;text-transform:uppercase;letter-spacing:2px;font-size:.72rem;font-weight:800;}
.hero h1 {font-size:2.3rem;line-height:1.08;letter-spacing:-1px;margin:12px 0;}
.hero h1 span {color:#d46aff;}
.hero p,.muted {color:#bdb3d2;}
.hero-car {
 width:100%;height:260px;object-fit:cover;border-radius:24px;
 border:1px solid rgba(192,132,252,.22);
 box-shadow:0 20px 45px rgba(111,35,180,.20);
 animation:floatcar 5s ease-in-out infinite;
}
@keyframes floatcar {0%,100%{transform:translateY(2px)}50%{transform:translateY(-7px)}}
.panel {
 padding:19px;border-radius:21px;
 background:linear-gradient(145deg,rgba(29,21,45,.95),rgba(12,10,20,.96));
 border:1px solid rgba(192,132,252,.22);
 box-shadow:0 16px 36px rgba(0,0,0,.28),inset 0 1px 0 rgba(255,255,255,.06);
 margin-bottom:15px;transition:transform .2s ease,box-shadow .2s ease;
}
.panel:hover {transform:translateY(-3px);box-shadow:0 22px 42px rgba(0,0,0,.32),0 0 22px rgba(168,85,247,.10);}
.result-number {font-size:2.35rem;font-weight:850;color:#df8aff;letter-spacing:-1px;}
.score-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;}
.score-card {padding:12px;border-radius:13px;background:rgba(255,255,255,.035);border:1px solid rgba(192,132,252,.17);}
.score-name {color:#c9c0df;font-size:.78rem;min-height:30px;}
.score-val {color:#e2a1ff;font-size:1.35rem;font-weight:800;}
.score-track {background:#30213e;border-radius:999px;height:6px;margin-top:8px;overflow:hidden;}
.score-fill {height:100%;border-radius:999px;background:linear-gradient(90deg,#994bff,#ec68ef);}
div[data-testid="stForm"] {border:1px solid rgba(192,132,252,.22);border-radius:20px;padding:18px;background:rgba(20,16,34,.78);box-shadow:0 18px 36px rgba(0,0,0,.22);}
.stFormSubmitButton>button {border:0;border-radius:12px;padding:.75rem 1.1rem;font-weight:800;color:white;background:linear-gradient(115deg,#7c2be8,#c83fe8);box-shadow:0 9px 24px rgba(151,58,235,.30);}
.stFormSubmitButton>button:hover {transform:translateY(-2px);box-shadow:0 13px 28px rgba(151,58,235,.4);}
label,.stMarkdown,p,h2,h3 {color:#f8f6ff;}
footer {visibility:hidden;}
@media(max-width:800px){.hero-copy{max-width:100%;}.hero h1{font-size:1.8rem;}}
</style>
""", unsafe_allow_html=True)

# 4. Header and car hero
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #352044;padding:5px 0 15px;margin-bottom:18px;">
 <div><div style="font-size:1.3rem;font-weight:800;">🚘 Auto <span style="color:#d46aff;">Evaluate</span></div>
 <div class="muted" style="font-size:.82rem;">Used Car Price Prediction</div></div>
 <div class="muted">Machine Learning · Price Estimator</div>
</div>
""", unsafe_allow_html=True)

hero_text, hero_image = st.columns([1, 1], gap="medium")
with hero_text:
    st.markdown("""
    <div class="hero">
      <div class="hero-copy" style="max-width:100%;">
        <div class="eyebrow">Auto Evaluate · ML Lab Project</div>
        <h1>Find the true value of your <span>next car.</span></h1>
        <p>Get a used-car price estimate from a trained machine-learning model.</p>
        <p style="margin-top:18px;">✦ Data-driven &nbsp;&nbsp; ✦ Simple to use &nbsp;&nbsp; ✦ Model-powered</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
with hero_image:
    if HERO_PATH.exists():
        st.image(str(HERO_PATH), use_container_width=True)

# 5. Create dropdown choices from the dataset
def options_for(column):
    return sorted(df[column].dropna().astype(str).unique().tolist())

brands = options_for("Brand")
models_list = options_for("Model")
bodies = options_for("Body")
engine_types = options_for("Engine Type")
registrations = options_for("Registration")

min_year = int(df["Year"].dropna().min())
max_year = max(date.today().year, int(df["Year"].dropna().max()))
default_year = min(2018, max_year)

# 6. Simple form on the left, result and R2 panel on the right
left, right = st.columns([1.05, .95], gap="large")

with left:
    st.markdown("### 🚘 Enter car details")
    st.markdown('<p class="muted">Choose the details below and estimate the price.</p>', unsafe_allow_html=True)
    with st.form("car_details"):
        c1, c2 = st.columns(2)
        with c1:
            brand = st.selectbox("Brand", brands, index=brands.index("Toyota") if "Toyota" in brands else 0)
            year = st.number_input("Year", min_value=min_year, max_value=max_year, value=default_year, step=1)
            mileage = st.number_input("Mileage (km)", min_value=0, max_value=1000000, value=100000, step=5000)
            engine_volume = st.number_input("Engine volume (L)", min_value=.1, max_value=10.0, value=2.0, step=.1)
        with c2:
            model_name = st.selectbox("Model", models_list, index=models_list.index("Corolla") if "Corolla" in models_list else 0)
            body = st.selectbox("Body style", bodies, index=bodies.index("sedan") if "sedan" in bodies else 0)
            engine_type = st.selectbox("Engine type", engine_types, index=engine_types.index("Petrol") if "Petrol" in engine_types else 0)
            registration = st.selectbox("Registration", registrations, index=registrations.index("yes") if "yes" in registrations else 0)
        submitted = st.form_submit_button("✦  Estimate price", use_container_width=True)

with right:
    if "predicted_price" not in st.session_state:
        st.markdown("""
        <div class="panel"><div class="eyebrow">Your estimate</div><h3>Estimated car price</h3>
        <div class="muted">Your prediction will appear here after you submit the form.</div></div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="eyebrow">Model performance</div><h3>R² score comparison</h3><p class="muted">Higher R² means the model explains more variation in test prices.</p>', unsafe_allow_html=True)
    score_html = '<div class="score-grid">'
    for _, row in scores.iterrows():
        value = max(0.0, min(1.0, float(row["R2"])))
        score_html += (
            '<div class="score-card"><div class="score-name">' + str(row["Model"]) +
            '</div><div class="score-val">' + f'{value:.3f}' +
            '</div><div class="score-track"><div class="score-fill" style="width:' +
            f'{value*100:.1f}%' + '"></div></div></div>'
        )
    score_html += '</div></div>'
    st.markdown(score_html, unsafe_allow_html=True)

# 7. Use the actual saved model to predict the price
if submitted:
    car_details = pd.DataFrame([{
        "Brand": brand,
        "Body": body,
        "Mileage": mileage,
        "EngineV": engine_volume,
        "Engine Type": engine_type,
        "Registration": registration,
        "Year": year,
        "Model": model_name
    }])
    try:
        predicted_price = float(model.predict(car_details)[0])
        st.session_state["predicted_price"] = predicted_price
        st.session_state["last_car_details"] = car_details
    except Exception:
        st.error("The model could not predict this car. Check that the saved model matches the notebook.")

if "predicted_price" in st.session_state:
    st.markdown("---")
    result_col, summary_col = st.columns([1, 1], gap="large")
    with result_col:
        st.markdown(
            '<div class="panel"><div class="eyebrow">Prediction complete</div><h3>Estimated price</h3><div class="result-number">' +
            f'{st.session_state["predicted_price"]:,.0f}' +
            '</div><p class="muted">Shown in the same price units as the training dataset. This is an estimate, not a guaranteed selling price.</p></div>',
            unsafe_allow_html=True
        )
    with summary_col:
        st.markdown('<div class="panel"><h3>🚗 Car summary</h3>', unsafe_allow_html=True)
        details = st.session_state["last_car_details"].iloc[0]
        summary = pd.DataFrame({
            "Detail": ["Brand", "Model", "Year", "Mileage", "Engine volume", "Engine type", "Body style", "Registration"],
            "Selected value": [details["Brand"], details["Model"], int(details["Year"]), f'{int(details["Mileage"]):,} km',
                               f'{details["EngineV"]:.1f} L', details["Engine Type"], details["Body"], details["Registration"]]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<p class="muted" style="text-align:center;margin:24px 0 8px;">Auto Evaluate · Student ML project · Estimates are for educational use</p>', unsafe_allow_html=True)
