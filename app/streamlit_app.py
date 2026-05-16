import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zomato Rating Predictor",
    page_icon="🍽️",
    layout="wide"
)

# ── Colour palette ─────────────────────────────────────────────────────────
TEAL  = '#0F6E56'
CORAL = '#D85A30'
NAVY  = '#1B3A5C'
GOLD  = '#C8960C'

# ── Paths ──────────────────────────────────────────────────────────────────
BASE        = '/content/drive/MyDrive/Colab Notebooks/zomato-rating-predictor'
MODEL_PATH  = f'{BASE}/models/xgb_model.pkl'
COLS_PATH   = f'{BASE}/models/feature_columns.pkl'
DATA_PATH   = f'{BASE}/data/zomato_clean.csv'
RAW_PATH    = f'{BASE}/data/zomato.csv'

# ── Load model & metadata ──────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(COLS_PATH, 'rb') as f:
        feature_columns = pickle.load(f)
    return model, feature_columns

@st.cache_data
def load_data():
    df_clean = pd.read_csv(DATA_PATH)
    df_raw   = pd.read_csv(RAW_PATH, encoding='latin1')
    df_raw   = df_raw[df_raw['Aggregate rating'] > 0].drop_duplicates()
    df_raw['primary_cuisine'] = (
        df_raw['Cuisines'].fillna('Unknown')
        .str.split(',').str[0].str.strip()
    )
    return df_clean, df_raw

@st.cache_data
def get_encoding_maps():
    _, df_raw = load_data()
    city_freq    = df_raw['City'].value_counts(normalize=True).to_dict()
    cuisine_freq = df_raw['primary_cuisine'].value_counts(normalize=True).to_dict()
    cities       = sorted(df_raw['City'].unique().tolist())
    cuisines     = sorted(df_raw['primary_cuisine'].unique().tolist())
    return city_freq, cuisine_freq, cities, cuisines

model, feature_columns = load_model()
df_clean, df_raw       = load_data()
city_freq, cuisine_freq, cities, cuisines = get_encoding_maps()

# Country one-hot columns in the feature set
country_cols = [c for c in feature_columns if c.startswith('country_')]
country_names = [c.replace('country_', '') for c in country_cols]

COUNTRY_MAP = {
    'India': 1, 'Australia': 14, 'Brazil': 30, 'Canada': 37,
    'Indonesia': 94, 'New Zealand': 148, 'Philippines': 162,
    'Qatar': 166, 'Singapore': 184, 'South Africa': 189,
    'Sri Lanka': 191, 'Turkey': 208, 'UAE': 214,
    'United Kingdom': 215, 'United States': 216
}

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #FAFAF8; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #E8E6E0;
        text-align: center;
    }
    .rating-display {
        font-size: 4rem;
        font-weight: 700;
        text-align: center;
        line-height: 1.1;
    }
    .rating-label {
        font-size: 1rem;
        text-align: center;
        margin-top: 0.2rem;
        font-weight: 500;
    }
    .insight-box {
        background: #F0F7F4;
        border-left: 4px solid #0F6E56;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        color: #1B3A5C;
    }
    .warning-box {
        background: #FEF6EE;
        border-left: 4px solid #D85A30;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        color: #7A3010;
    }
    .section-header {
        color: #1B3A5C;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #0F6E56;
    }
    div[data-testid="stSidebar"] {
        background-color: #F4F2EC;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='background: linear-gradient(135deg, {NAVY} 0%, {TEAL} 100%);
            padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;'>
    <h1 style='color: white; margin: 0; font-size: 2rem;'>🍽️ Zomato Rating Predictor</h1>
    <p style='color: rgba(255,255,255,0.8); margin: 0.4rem 0 0; font-size: 1rem;'>
        Enter restaurant attributes to predict its Zomato rating — powered by XGBoost + SHAP
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Inputs
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<p class='section-header'>Restaurant Details</p>", unsafe_allow_html=True)

    selected_city = st.selectbox("City", options=cities, index=cities.index('New Delhi') if 'New Delhi' in cities else 0)
    selected_cuisine = st.selectbox("Primary Cuisine", options=cuisines, index=cuisines.index('North Indian') if 'North Indian' in cuisines else 0)
    selected_country = st.selectbox("Country", options=sorted(COUNTRY_MAP.keys()), index=sorted(COUNTRY_MAP.keys()).index('India'))

    st.markdown("---")
    st.markdown(f"<p class='section-header'>Pricing</p>", unsafe_allow_html=True)

    price_range = st.select_slider(
        "Price Range",
        options=[1, 2, 3, 4],
        value=2,
        format_func=lambda x: {1: "1 — Budget", 2: "2 — Moderate", 3: "3 — Expensive", 4: "4 — Luxury"}[x]
    )
    avg_cost = st.number_input(
        "Average Cost for Two (local currency)",
        min_value=0, max_value=500000, value=800, step=100
    )

    st.markdown("---")
    st.markdown(f"<p class='section-header'>Services</p>", unsafe_allow_html=True)

    has_online  = st.toggle("Has Online Delivery", value=False)
    has_booking = st.toggle("Has Table Booking",   value=False)

    st.markdown("---")
    st.markdown(f"<p class='section-header'>Menu</p>", unsafe_allow_html=True)

    cuisine_count = st.slider("Number of Cuisines Offered", min_value=1, max_value=10, value=1)

    st.markdown("---")
    st.markdown(f"<p class='section-header'>Engagement</p>", unsafe_allow_html=True)

    votes = st.number_input(
        "Number of Votes / Reviews",
        min_value=0, max_value=100000, value=100, step=10,
        help="For a new restaurant, enter 0 or a small estimate."
    )

    predict_btn = st.button("🔮 Predict Rating", use_container_width=True, type="primary")

# ══════════════════════════════════════════════════════════════════════════════
# BUILD INPUT VECTOR
# ══════════════════════════════════════════════════════════════════════════════
def build_input_row():
    row = {col: 0 for col in feature_columns}

    # Core numeric
    row['Average Cost for two']  = avg_cost
    row['Has Table booking']     = int(has_booking)
    row['Has Online delivery']   = int(has_online)
    row['Price range']           = price_range
    row['Votes']                 = votes
    row['cuisine_count']         = cuisine_count
    row['is_multi_cuisine']      = int(cuisine_count > 1)
    row['log_votes']             = np.log1p(votes)
    row['cost_per_person']       = avg_cost / 2
    row['has_both_services']     = int(has_online and has_booking)

    # Frequency encodings
    row['city_freq_enc']    = city_freq.get(selected_city, 0.0)
    row['cuisine_freq_enc'] = cuisine_freq.get(selected_cuisine, 0.0)

    # Country one-hot
    country_col = f'country_{selected_country}'
    if country_col in row:
        row[country_col] = 1

    return pd.DataFrame([row])[feature_columns]

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW STATS (always visible)
# ══════════════════════════════════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size:1.8rem; font-weight:700; color:{TEAL};'>{len(df_clean):,}</div>
        <div style='color:#888; font-size:0.85rem; margin-top:4px;'>Restaurants in Dataset</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size:1.8rem; font-weight:700; color:{NAVY};'>{df_clean['Aggregate rating'].mean():.2f}</div>
        <div style='color:#888; font-size:0.85rem; margin-top:4px;'>Average Rating</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size:1.8rem; font-weight:700; color:{GOLD};'>0.6442</div>
        <div style='color:#888; font-size:0.85rem; margin-top:4px;'>Model R² Score</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='font-size:1.8rem; font-weight:700; color:{CORAL};'>±0.33</div>
        <div style='color:#888; font-size:0.85rem; margin-top:4px;'>Avg Prediction Error</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
if predict_btn:
    X_input = build_input_row()
    predicted_rating = float(model.predict(X_input)[0])
    predicted_rating = np.clip(predicted_rating, 1.0, 5.0)

    # Rating tier
    if predicted_rating >= 4.5:
        tier, tier_color, tier_emoji = "Excellent",    TEAL,  "🌟"
    elif predicted_rating >= 4.0:
        tier, tier_color, tier_emoji = "Very Good",    TEAL,  "✅"
    elif predicted_rating >= 3.5:
        tier, tier_color, tier_emoji = "Good",         GOLD,  "👍"
    elif predicted_rating >= 3.0:
        tier, tier_color, tier_emoji = "Average",      GOLD,  "😐"
    elif predicted_rating >= 2.0:
        tier, tier_color, tier_emoji = "Below Average", CORAL, "⚠️"
    else:
        tier, tier_color, tier_emoji = "Poor",          CORAL, "❌"

    # ── Layout: rating left, insights right ───────────────────────────────
    left, right = st.columns([1, 2])

    with left:
        st.markdown(f"""
        <div style='background:white; border-radius:16px; padding:2rem;
                    border: 2px solid {tier_color}; text-align:center;'>
            <div style='font-size:0.9rem; color:#888; margin-bottom:0.5rem;'>
                Predicted Rating
            </div>
            <div class='rating-display' style='color:{tier_color};'>
                {predicted_rating:.1f}
            </div>
            <div style='font-size:2rem; margin: 0.3rem 0;'>{tier_emoji}</div>
            <div class='rating-label' style='color:{tier_color};'>{tier}</div>
            <div style='font-size:0.8rem; color:#aaa; margin-top:1rem;'>
                Scale: 1.8 – 4.9
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Context: where does this sit relative to dataset average
        avg = df_clean['Aggregate rating'].mean()
        diff = predicted_rating - avg
        direction = "above" if diff > 0 else "below"
        st.markdown(f"""
        <div style='margin-top:1rem; background:#F4F2EC; border-radius:10px;
                    padding:0.8rem 1rem; font-size:0.88rem; color:{NAVY};'>
            This restaurant is predicted to score
            <strong>{abs(diff):.2f} points {direction}</strong>
            the dataset average of <strong>{avg:.2f}</strong>.
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown(f"<p class='section-header'>What's Driving This Prediction</p>", unsafe_allow_html=True)

        # ── SHAP for this prediction ───────────────────────────────────────
        explainer   = shap.Explainer(model, X_input)
        shap_values = explainer(X_input)

        sv   = shap_values.values[0]
        base = shap_values.base_values[0]

        shap_df = pd.DataFrame({
            'Feature':    feature_columns,
            'SHAP Value': sv
        }).reindex(pd.Series(sv).abs().sort_values(ascending=False).index)
        shap_df = shap_df.head(10)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [TEAL if v > 0 else CORAL for v in shap_df['SHAP Value']]

        bars = ax.barh(
            shap_df['Feature'][::-1],
            shap_df['SHAP Value'][::-1],
            color=colors[::-1], edgecolor='white', linewidth=0.4, alpha=0.9
        )
        for bar, val in zip(bars, shap_df['SHAP Value'][::-1]):
            ax.text(
                val + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}',
                va='center',
                ha='left' if val >= 0 else 'right',
                fontsize=8.5, color='#444'
            )

        ax.axvline(0, color='#ccc', lw=0.8)
        ax.set_title(
            f'Feature Contributions  |  Base: {base:.2f}  →  Prediction: {predicted_rating:.2f}',
            fontsize=10, color=NAVY, pad=10
        )
        ax.set_xlabel('SHAP Value (teal = pushes rating up, coral = pushes rating down)', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Plain-English Insights ─────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<p class='section-header'>Key Insights</p>", unsafe_allow_html=True)

        top_positive = shap_df[shap_df['SHAP Value'] > 0].head(2)
        top_negative = shap_df[shap_df['SHAP Value'] < 0].head(2)

        for _, row in top_positive.iterrows():
            feat = row['Feature']
            val  = row['SHAP Value']
            if feat == 'Votes':
                msg = f"High vote count is your biggest asset — it adds <strong>+{val:.2f}</strong> to the predicted rating. Popularity builds trust."
            elif feat == 'log_votes':
                msg = f"Strong review volume is boosting the prediction by <strong>+{val:.2f}</strong>."
            elif feat == 'city_freq_enc':
                msg = f"Being in a high-density restaurant city adds <strong>+{val:.2f}</strong> — competitive markets tend to push quality up."
            elif feat == 'cuisine_freq_enc':
                msg = f"Your primary cuisine is well-represented on Zomato, contributing <strong>+{val:.2f}</strong> to the prediction."
            elif feat == 'Price range':
                msg = f"Your price positioning is working in your favour (+{val:.2f}). Higher-tier restaurants tend to attract more deliberate diners."
            elif feat == 'Average Cost for two':
                msg = f"Your cost point is positively associated with rating (+{val:.2f})."
            elif feat in ('Has Table booking', 'has_both_services'):
                msg = f"Offering table booking is positively correlated with higher ratings (+{val:.2f})."
            elif feat == 'Has Online delivery':
                msg = f"Online delivery availability is contributing positively (+{val:.2f})."
            else:
                msg = f"<strong>{feat}</strong> is contributing <strong>+{val:.2f}</strong> to the prediction."
            st.markdown(f"<div class='insight-box'>✅ {msg}</div>", unsafe_allow_html=True)

        for _, row in top_negative.iterrows():
            feat = row['Feature']
            val  = row['SHAP Value']
            if feat == 'Votes':
                msg = f"Low vote count is the main drag on this prediction ({val:.2f}). Getting more reviews should be the priority."
            elif feat == 'log_votes':
                msg = f"Limited review volume is pulling the rating down ({val:.2f}). Encouraging customers to leave reviews could improve this significantly."
            elif feat == 'city_freq_enc':
                msg = f"Being in a less-dense city is reducing the prediction ({val:.2f}) — fewer comparisons, less visibility."
            elif feat == 'cuisine_freq_enc':
                msg = f"This cuisine type is less common on Zomato, which is weighing on the prediction ({val:.2f})."
            elif feat == 'Price range':
                msg = f"Budget pricing is associated with lower ratings in this dataset ({val:.2f}). Consider whether the value perception matches the experience."
            elif feat == 'Average Cost for two':
                msg = f"The cost point is negatively affecting the prediction ({val:.2f})."
            elif feat in ('Has Table booking', 'has_both_services'):
                msg = f"Not offering table booking is holding back the rating ({val:.2f})."
            elif feat == 'Has Online delivery':
                msg = f"No online delivery option is a slight negative ({val:.2f}) — competitors in this segment likely offer it."
            else:
                msg = f"<strong>{feat}</strong> is reducing the prediction by <strong>{val:.2f}</strong>."
            st.markdown(f"<div class='warning-box'>⚠️ {msg}</div>", unsafe_allow_html=True)

    # ── Model disclaimer ───────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        f"**Model accuracy note:** This XGBoost model has an R² of 0.64 and RMSE of ±0.33 "
        f"on held-out test data. Predictions should be interpreted as directional estimates, "
        f"not exact ratings. Restaurant ratings are inherently subjective — the model captures "
        f"structural patterns but cannot account for chef quality, décor, or service."
    )

else:
    # ── Placeholder when no prediction yet ────────────────────────────────
    st.markdown(f"""
    <div style='background:white; border-radius:16px; padding:3rem 2rem;
                border: 1px dashed #ccc; text-align:center; color:#aaa;'>
        <div style='font-size:3rem;'>🍽️</div>
        <div style='font-size:1.1rem; margin-top:1rem; color:#888;'>
            Fill in the restaurant details in the sidebar and click
            <strong style='color:{TEAL};'>Predict Rating</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET EXPLORER (always visible at bottom)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📊 Dataset Explorer", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Rating Distribution", "Top Cities", "Top Cuisines"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df_clean['Aggregate rating'], bins=30, color=TEAL, edgecolor='white', linewidth=0.5)
        ax.axvline(df_clean['Aggregate rating'].mean(), color=CORAL, lw=2, linestyle='--',
                   label=f"Mean: {df_clean['Aggregate rating'].mean():.2f}")
        ax.set_xlabel('Aggregate Rating'); ax.set_ylabel('Count')
        ax.set_title('Rating Distribution (rated restaurants only)')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend()
        st.pyplot(fig); plt.close()

    with tab2:
        city_counts = df_raw['City'].value_counts().head(15).reset_index()
        city_counts.columns = ['City', 'Count']
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(city_counts['City'][::-1], city_counts['Count'][::-1],
                color=NAVY, edgecolor='white', linewidth=0.4, alpha=0.85)
        ax.set_title('Top 15 Cities by Restaurant Count')
        ax.set_xlabel('Count')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()

    with tab3:
        cuisine_stats = (
            df_raw.groupby('primary_cuisine')['Aggregate rating']
            .agg(['mean', 'count'])
            .reset_index()
        )
        cuisine_stats = cuisine_stats[cuisine_stats['count'] >= 30].sort_values('mean', ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [TEAL if m >= df_raw['Aggregate rating'].mean() else '#ccc' for m in cuisine_stats['mean']]
        ax.barh(cuisine_stats['primary_cuisine'][::-1], cuisine_stats['mean'][::-1],
                color=colors[::-1], edgecolor='white', linewidth=0.4, alpha=0.9)
        ax.axvline(df_raw['Aggregate rating'].mean(), color=CORAL, lw=1.5, linestyle='--',
                   label=f"Overall mean: {df_raw['Aggregate rating'].mean():.2f}")
        ax.set_title('Top 15 Cuisines by Average Rating (min 30 restaurants)')
        ax.set_xlabel('Mean Rating')
        ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        st.pyplot(fig); plt.close()

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center; color:#aaa; font-size:0.8rem; margin-top:3rem; padding-top:1rem;
            border-top: 1px solid #eee;'>
    Zomato Rating Predictor · XGBoost + SHAP · Built by Gedion Leslie Kweya Odera
</div>
""", unsafe_allow_html=True)
