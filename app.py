import streamlit as st
import numpy as np
import pickle

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="Customer Clustering App",
    page_icon="🧩",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------- Custom CSS ----------------------
st.markdown(
    """
    <style>
    /* Background Gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e3f2fd, #fce4ec);
    }

    /* Header Style */
    .title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a237e;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Input container styling */
    .stNumberInput label {
        font-weight: 600 !important;
        color: #1565c0 !important;
    }

    /* Button styling */
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        background: linear-gradient(90deg, #1976d2, #42a5f5);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #1565c0, #1e88e5);
        transform: scale(1.03);
        transition: all 0.3s ease-in-out;
    }

    /* Success message styling */
    .stSuccess {
        background-color: #e8f5e9 !important;
        border-radius: 10px !important;
        border: 1px solid #66bb6a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- App Header ----------------------
st.markdown("<div class='title'>🧩 Customer Clustering App</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Predict your customer's behavior using K-Means Clustering</div>",
    unsafe_allow_html=True,
)

# ---------------------- Load Model ----------------------
try:
    kmeans = pickle.load(open("kmeans.pkl", "rb"))
except FileNotFoundError:
    st.error("⚠️ Model file 'kmeans.pkl' not found. Please make sure it's in the same folder.")
    st.stop()

# ---------------------- Prediction Function ----------------------
def clustering(age, avg_spend, visit_per_week, promotion_interest):
    new_customer = np.array([[age, avg_spend, visit_per_week, promotion_interest]])
    predicted_cluster = kmeans.predict(new_customer)
    labels = ["🛍️ Daily Customer", "🎉 Weekend Customer", "💸 Promotion Seeker"]
    return labels[predicted_cluster[0]]

# ---------------------- Input Section ----------------------
st.markdown("### 🧾 Enter Customer Details")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Customer Age", min_value=18, max_value=100, value=40)
        visit_per_week = st.number_input("Visits per Week", min_value=0, max_value=20, value=4)
    with col2:
        avg_spend = st.number_input("Average Spend ($)", min_value=0.0, max_value=1000.0, value=30.0)
        promotion_interest = st.number_input("Promotion Interest (0-10)", min_value=0, max_value=10, value=7)

# ---------------------- Predict Button ----------------------
st.markdown(" ")
if st.button("🔮 Predict Customer Cluster"):
    cluster_label = clustering(age, avg_spend, visit_per_week, promotion_interest)
    st.success(f"✅ Prediction: The customer belongs to the **{cluster_label}** cluster.")

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True,
)
