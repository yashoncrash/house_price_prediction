import  streamlit as st
import joblib
import pandas as pd


# ---------- Page setup ----------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.markdown(
    """
    <style>
    .big-title {font-size: 2.6rem; font-weight: 800; text-align: center;
                background: linear-gradient(90deg, #4f8bf9, #7b2ff7);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .sub-title {text-align: center; color: gray; margin-bottom: 2rem;}
    .price-box {padding: 1.5rem; border-radius: 15px; text-align: center;
                background: linear-gradient(135deg, #4f8bf9, #7b2ff7); color: white;}
    .price-box h1 {margin: 0; font-size: 2.8rem; color: white;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Load model once ----------
@st.cache_resource
def load_model():
    return joblib.load("linear_pipeline.pkl")


model = load_model()

# ---------- Header ----------
st.markdown('<div class="big-title">🏠 House Price Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter the house details and get an instant price estimate</div>', unsafe_allow_html=True)

# ---------- Sidebar inputs ----------
st.sidebar.header("🔧 House Details")

area_sqft = st.sidebar.slider("Area (sq ft)", 200, 10000, 1500, step=50)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=0, max_value=20, value=3)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=0, max_value=20, value=2)
age_years = st.sidebar.slider("Age of house (years)", 0, 100, 5)
distance_city_km = st.sidebar.slider("Distance from city (km)", 0, 100, 10)
has_parking = st.sidebar.radio("Parking available?", ["Yes", "No"], horizontal=True)

predict_btn = st.sidebar.button("🔮 Predict Price", use_container_width=True)

# ---------- Summary cards ----------
c1, c2, c3 = st.columns(3)
c1.metric("📐 Area", f"{area_sqft} sq ft")
c2.metric("🛏️ Bedrooms / 🚿 Bathrooms", f"{bedrooms} / {bathrooms}")
c3.metric("📍 Distance from city", f"{distance_city_km} km")

st.divider()

# ---------- Prediction ----------
if predict_btn:
    input_df = pd.DataFrame([{
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "distance_city_km": distance_city_km,
        "has_parking": 1 if has_parking == "Yes" else 0,
    }])

    with st.spinner("Calculating..."):
        prediction = float(model.predict(input_df)[0])

    st.markdown(
        f'<div class="price-box"><p>Estimated Price</p><h1>{prediction:,.2f}</h1></div>',
        unsafe_allow_html=True,
    )
    st.balloons()

    with st.expander("📄 See input data sent to the model"):
        st.dataframe(input_df, use_container_width=True)
else:
    st.info("👈 Fill in the details in the sidebar and click **Predict Price**.")
